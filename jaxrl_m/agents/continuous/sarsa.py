import copy
from functools import partial
from typing import Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.encoding import EncodingWrapper
from jaxrl_m.common.optimizers import make_optimizer
from jaxrl_m.common.typing import Batch, Data, Params, PRNGKey
from jaxrl_m.networks.actor_critic_nets import Critic, ensemblize
from jaxrl_m.networks.mlp import MLP


def sarsa_loss(q, q_target):
    """MSE loss between Q and SARSA target Q"""
    loss = jnp.square(q - q_target)
    return loss.mean(), {
        "sarsa_loss": loss.mean(),
        "q_mean": q.mean(),
        "q_std": q.std(),
        "q_target_mean": q_target.mean(),
        "q_target_std": q_target.std(),
        "td_error": jnp.abs(q - q_target).mean(),
    }


class SARSAAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def forward_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """Forward pass for critic network."""
        if train:
            assert rng is not None, "Must specify rng when training"
        qs = self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )
        return qs

    def forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        """Forward pass for target critic network."""
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params
        )

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        rng, new_rng = jax.random.split(self.state.rng)

        def critic_loss_fn(params, rng):
            """
            SARSA critic loss: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
            Uses the actual next actions from the dataset, not max Q-values
            """
            rng, key = jax.random.split(rng)
            
            # Get Q-values for current state-action pairs
            q = self.forward_critic(
                batch["observations"], batch["actions"], key, grad_params=params
            )
            
            # For SARSA, we need the next actions that were actually taken
            # TODO: These should be in the batch as "next_actions"
            if "next_actions" not in batch:
                raise KeyError("SARSA requires 'next_actions' in batch. "
                             "Make sure your dataset includes the actual next actions taken.")
            
            rng, key = jax.random.split(rng)
            # Get Q-values for next state-action pairs using target network
            next_q = self.forward_target_critic(
                batch["next_observations"], batch["next_actions"], key
            )
            
            # SARSA target: r + γ * Q(s', a')
            # Take minimum over ensemble if using multiple Q-functions
            if next_q.ndim > 1 and self.config["critic_ensemble_size"] > 1:
                next_q = jnp.min(next_q, axis=0)
                if q.ndim > 1:
                    q = jnp.min(q, axis=0)
            
            target_q = batch["rewards"] + self.config["discount"] * next_q * batch["masks"]
            
            return sarsa_loss(q, target_q)

        loss_fns = {
            "critic": critic_loss_fn,
        }

        # Compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # Update target params with soft update
        new_state = new_state.target_update(self.config["target_update_rate"])

        # Update RNG
        new_state = new_state.replace(rng=new_rng)

        # Log learning rates
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    def get_q_values(
        self,
        observations: Data,
        actions: jax.Array,
        rng: Optional[PRNGKey] = None,
    ) -> jnp.ndarray:
        """Get Q-values for given state-action pairs"""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        q_values = self.forward_critic(observations, actions, rng, train=False)
        
        # If ensemble, take minimum or mean based on config
        if q_values.ndim > 1 and self.config["critic_ensemble_size"] > 1:
            if self.config.get("use_min_q", True):
                q_values = jnp.min(q_values, axis=0)
            else:
                q_values = jnp.mean(q_values, axis=0)
        
        return q_values

    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        """Get debugging metrics"""
        # Get Q-values for current state-action pairs
        q = self.state.apply_fn(
            {"params": self.state.params},
            batch["observations"],
            batch["actions"],
            name="critic",
        )
        
        metrics = {
            "q": q,
        }
        
        # Get next Q-values and targets if next_actions available
        if "next_actions" in batch:
            next_q = self.state.apply_fn(
                {"params": self.state.target_params},
                batch["next_observations"],
                batch["next_actions"],
                name="critic",
            )
            
            # Handle ensemble
            if q.ndim > 1 and self.config["critic_ensemble_size"] > 1:
                q_min = jnp.min(q, axis=0)
                next_q_min = jnp.min(next_q, axis=0)
                target_q = batch["rewards"] + self.config["discount"] * next_q_min * batch["masks"]
                
                metrics.update({
                    "q_min": q_min,
                    "q_ensemble": q,
                    "next_q": next_q,
                    "next_q_min": next_q_min,
                    "target_q": target_q,
                    "td_error": jnp.square(q_min - target_q),
                    "reward": batch["rewards"],
                })
            else:
                target_q = batch["rewards"] + self.config["discount"] * next_q * batch["masks"]
                metrics.update({
                    "next_q": next_q,
                    "target_q": target_q,
                    "td_error": jnp.square(q - target_q),
                    "reward": batch["rewards"],
                })

        return metrics

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        actions: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        use_proprio: bool = False,
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        # Optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 2000,
        # Algorithm config
        discount: float = 0.98,
        target_update_rate: float = 0.002,
        critic_ensemble_size: int = 2,
        use_min_q: bool = True,  # Whether to use min over ensemble
    ):
        if encoder_def is not None:
            encoder_def = EncodingWrapper(
                encoder=encoder_def,
                use_proprio=use_proprio,
                stop_gradient=False,
            )

        network_kwargs["activate_final"] = True
        
        # Create only critic network (no actor needed for Q-learning)
        networks = {
            "critic": Critic(
                encoder_def,
                network=ensemblize(partial(MLP, **network_kwargs), critic_ensemble_size)(
                    name="critic_ensemble"
                ),
            ),
        }

        model_def = ModuleDict(networks)

        # Define optimizer (only for critic)
        txs = {
            "critic": make_optimizer(
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
            ),
        }

        # Initialize parameters
        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            critic=[observations, actions],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        config = flax.core.FrozenDict(
            dict(
                discount=discount,
                target_update_rate=target_update_rate,
                critic_ensemble_size=critic_ensemble_size,
                use_min_q=use_min_q,
            )
        )
        
        return cls(state, config)
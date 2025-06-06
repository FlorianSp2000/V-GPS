"""
Minimal SARSA Agent
Compatible with existing CQL/SAC training pipeline
"""
import copy
from functools import partial
from typing import Optional, Union, Tuple
import chex

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.encoding import EncodingWrapper, GCEncodingWrapper, LCEncodingWrapper
from jaxrl_m.common.optimizers import make_optimizer
from jaxrl_m.common.typing import Batch, Data, Params, PRNGKey
from jaxrl_m.networks.actor_critic_nets import Critic, ensemblize
from jaxrl_m.networks.mlp import MLP

# For plotting (optional)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class SARSAAgent(flax.struct.PyTreeNode):
    """
    Minimal SARSA agent for pure Q-learning.
    
    Only learns Q(s,a) from (s, a, r, s', a') tuples.
    Compatible with existing CQL/SAC training pipeline.
    """
    
    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def _include_goals_in_obs(self, batch, which_obs: str):
        """
        Include goals in observations if goal-conditioned.
        
        This is needed because:
        - Goal-conditioned RL requires the critic to see both observation AND goal
        - Creates tuple (obs, goals) that gets passed to networks
        - Allows conditioning Q-values on language instructions
        """
        assert which_obs in ("observations", "next_observations")
        obs = batch[which_obs]
        if self.config["goal_conditioned"]:
            obs = (obs, batch["goals"])
        return obs

    def forward_critic(
        self,
        observations: Union[Data, Tuple[Data, Data]], 
        actions: jax.Array,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """Forward pass for critic network"""
        if train:
            assert rng is not None, "Must specify rng when training"
        # Handle multiple actions per state (for OOD evaluation)
        if jnp.ndim(actions) == 3:
            # 3D case: (batch_size, num_actions, action_dim)
            q_values = jax.vmap(
                lambda a: self.state.apply_fn(
                    {"params": grad_params or self.state.params},
                    observations,
                    a,
                    name="critic",
                    rngs={"dropout": rng} if train else {},
                    train=train,
                ),
                in_axes=1,
                out_axes=-1,
            )(actions)
            
            # Critical assertion for new 3D functionality
            chex.assert_shape(q_values, (self.config["critic_ensemble_size"], actions.shape[0], actions.shape[1]))
            return q_values   
        else:         
            return self.state.apply_fn(
                {"params": grad_params or self.state.params},
                observations,
                actions,
                name="critic",
                rngs={"dropout": rng} if train else {},
                train=train,
            )

    def forward_target_critic(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        actions: jax.Array, 
        rng: PRNGKey,
    ) -> jax.Array:
        """Forward pass for target critic network"""
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params, train=False
        )
    def _sample_ood_actions(self, batch, rng: PRNGKey, num_ood_actions: int = 10):
        """
        Sample Out-of-Distribution (OOD) actions for evaluation.
        Similar to CQL's random action sampling but for SARSA evaluation.
        """
        batch_size = batch["rewards"].shape[0]
        action_dim = batch["actions"].shape[-1]
        
        rng, ood_rng = jax.random.split(rng)
        
        # Sample random actions like CQL does
        if self.config.get("ood_action_sample_method", "uniform") == "uniform":
            ood_actions = jax.random.uniform(
                ood_rng,
                shape=(batch_size, num_ood_actions, action_dim),
                minval=-1.0,
                maxval=1.0,
            )
        elif self.config.get("ood_action_sample_method", "uniform") == "normal":
            ood_actions = jax.random.normal(
                ood_rng,
                shape=(batch_size, num_ood_actions, action_dim),
            )
        else:
            # Default to uniform
            ood_actions = jax.random.uniform(
                ood_rng,
                shape=(batch_size, num_ood_actions, action_dim),
                minval=-1.0,
                maxval=1.0,
            )
        
        chex.assert_shape(ood_actions, (batch_size, num_ood_actions, action_dim))
        return ood_actions

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self, 
        batch: Batch, 
        *,
        pmap_axis: str = None,
        networks_to_update: frozenset[str] = frozenset({"critic"}),
    ):
        """Update Q-function using SARSA target"""
        
        def loss_fn(params, rng):
            batch_size = batch["rewards"].shape[0]
            
            # Current Q-values: Q(s, a)
            current_q = self.forward_critic(
                self._include_goals_in_obs(batch, "observations"),
                batch["actions"],
                rng=rng,
                grad_params=params,
                train=True,
            )  # (ensemble_size, batch_size)
            
            # Target Q-values: Q_target(s', a') using ACTUAL next actions
            rng, target_rng = jax.random.split(rng)
            target_next_q = self.forward_target_critic(
                self._include_goals_in_obs(batch, "next_observations"),
                batch["next_actions"],  # 🎯 KEY: Use actual next actions!
                rng=target_rng,
            )  # (ensemble_size, batch_size)
            
            # Subsample ensemble if requested
            if self.config["critic_subsample_size"] is not None:
                rng, subsample_key = jax.random.split(rng)
                subsample_idcs = jax.random.randint(
                    subsample_key,
                    (self.config["critic_subsample_size"],),
                    0,
                    self.config["critic_ensemble_size"],
                )
                target_next_q = target_next_q[subsample_idcs]
            
            # Take minimum across ensemble for target (reduce overestimation)
            if self.config.get("use_min_q", True):
                target_next_min_q = target_next_q.min(axis=0)
            else:
                target_next_min_q = target_next_q.mean(axis=0)
            
            # SARSA target: r + γ * mask * Q_target(s', a')
            targets = (
                batch["rewards"] 
                + self.config["discount"] * batch["masks"] * target_next_min_q
            )
            chex.assert_shape(targets, (batch_size,))
            # Broadcast targets to match current_q shape
            targets = targets[None].repeat(current_q.shape[0], axis=0)
            
            # MSE loss
            td_error = current_q - targets
            critic_loss = jnp.mean(td_error ** 2)
            
            info = {
                "critic_loss": critic_loss,
                "online_q": jnp.mean(current_q), # called q_values
                "target_q": jnp.mean(targets),  # target_values
                "td_err": jnp.mean(jnp.abs(td_error)), # td_error, cql uses mse instead of mae for td error
                "rewards": jnp.mean(batch["rewards"]),
            }
            
            return critic_loss, info
        
        # Setup loss functions (follow SAC pattern)
        loss_fns = {"critic": partial(loss_fn)}
        
        # Only compute gradients for specified networks
        assert networks_to_update.issubset(
            loss_fns.keys()
        ), f"Invalid networks: {networks_to_update}"
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        # Update RNG (follow SAC pattern)
        rng, new_rng = jax.random.split(self.state.rng)
        
        # Apply loss functions
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )
        
        # Update target network (follow SAC pattern)
        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])
        
        # Update RNG (follow SAC pattern)
        new_state = new_state.replace(rng=new_rng)
        
        # Log learning rates (follow SAC pattern - fixed the issue!)
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]
        
        return self.replace(state=new_state), info

    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        """Debug metrics for validation"""
        rng = jax.random.PRNGKey(0)
        
        # Current Q-values
        current_q = self.forward_critic(
            self._include_goals_in_obs(batch, "observations"),
            batch["actions"],
            rng=rng,
            train=False,
        )
        
        # Target Q-values  
        target_q = self.forward_target_critic(
            self._include_goals_in_obs(batch, "next_observations"),
            batch["next_actions"],
            rng=rng,
        )
        # --- OOD Q-values evaluation ---
        rng, ood_rng = jax.random.split(rng)
        ood_actions = self._sample_ood_actions(batch, ood_rng, num_ood_actions=self.config.get("num_ood_actions", 10))
        ood_q = self.forward_critic(
            self._include_goals_in_obs(batch, "observations"),
            ood_actions,
            rng=rng,
            train=False,
        )

        return {
            "online_q": jnp.mean(current_q), # called q_values
            "target_q": jnp.mean(target_q),
            "q_std": jnp.std(current_q),
            "rewards": batch["rewards"],
            "ood_q": jnp.mean(ood_q),
        }

    def get_q_values(self, observations, goals, actions):
        """Get Q-values for given state-action pairs"""
        obs = (observations, goals) if self.config["goal_conditioned"] else observations
        q_values = self.state.apply_fn(
            {"params": self.state.target_params},
            obs,
            actions,
            name="critic",
        )
        return jnp.min(q_values, axis=0)  # Take min across ensemble

    def get_eval_values(self, traj, seed, goals):
        """Evaluate Q-values over trajectory"""
        obs = (traj["observations"], goals) if self.config["goal_conditioned"] else traj["observations"]
        
        # Current Q-values
        q_values = self.forward_critic(obs, traj["actions"], seed, train=False)
        q_values = jnp.min(q_values, axis=0)
        
        # Target Q-values
        target_q_values = self.forward_target_critic(obs, traj["actions"], seed)
        target_q_values = jnp.min(target_q_values, axis=0)
        
        # OOD Q-values - sample random actions and evaluate
        rng, ood_rng = jax.random.split(seed) if seed is not None else (jax.random.PRNGKey(42), jax.random.PRNGKey(43)) #TODO: seed is passed in train, but is there better fallback?
        batch_like = {"rewards": traj["rewards"], "actions": traj["actions"]} # for shape
        ood_actions = self._sample_ood_actions(batch_like, ood_rng, num_ood_actions=self.config.get("num_ood_actions", 10)) 
        
        # Get OOD Q-values
        ood_q = self.forward_critic(obs, ood_actions, rng, train=False)
        ood_q_values = jnp.mean(jnp.min(ood_q, axis=0), axis=-1)  # Mean over OOD actions, min over ensemble
        
        # OOD Target Q-values
        ood_target_q = self.forward_target_critic(obs, ood_actions, rng)
        ood_target_q_values = jnp.mean(jnp.min(ood_target_q, axis=0), axis=-1)

        return {
            "q": q_values,
            "target_q": target_q_values,
            "q_ood": ood_q_values,
            "target_q_ood": ood_target_q_values,
            "rewards": traj["rewards"],
            "masks": traj["masks"],
        }
        
    def plot_values(self, traj, seed=None, goals=None):
        """Plot Q-values over trajectory (for compatibility with training pipeline)"""
        from absl import logging
        if goals is None:
            goals = traj["goals"]
        else:
            # Handle goal length mismatch
            traj_len = traj["observations"]["image"].shape[0]
            if goals["language"].shape[0] != traj_len:
                if goals["language"].shape[0] > traj_len:
                    goals = {k: v[:traj_len] for k, v in goals.items()}
                else:
                    num_repeat = traj_len - goals["language"].shape[0]
                    for k, v in goals.items():
                        rep = jnp.repeat(v[-1:], num_repeat, axis=0)
                        goals[k] = jnp.concatenate([v, rep], axis=0)
        logging.info(f"goals['language'].shape: {goals['language'].shape}")
        metrics = self.get_eval_values(traj, seed, goals)
        images = traj["observations"]["image"].squeeze()
        
        num_rows = len(metrics.keys()) + 1 # image without: + language prompt + summary
        fig, axs = plt.subplots(num_rows, 1, figsize=(10, 16)) # 8, 16 was before 
        canvas = FigureCanvas(fig)
        
        current_row = 0
        # Row 0: Plot images
        interval = max(1, images.shape[0] // 8)
        sel_images = images[::interval]
        sel_images = np.split(sel_images, sel_images.shape[0], 0)
        sel_images = [a.squeeze() for a in sel_images]
        sel_images = np.concatenate(sel_images, axis=1)
        axs[current_row].imshow(sel_images)
        axs[current_row].set_title("Trajectory Images")
        current_row += 1
        
        # # Row 1: Summary statistics
        # summary_text = (
        #     f"Trajectory Length: {len(metrics['rewards'])}\n"
        #     f"Mean Reward: {jnp.mean(metrics['rewards']):.3f}\n"
        #     f"Mean Q-value: {jnp.mean(metrics['q']):.3f}\n"
        #     f"Mean OOD Q-value: {jnp.mean(metrics['q_ood']):.3f}\n"
        #     f"Q vs OOD Q Gap: {jnp.mean(metrics['q']) - jnp.mean(metrics['q_ood']):.3f}"
        # )
        # axs[current_row].text(0.1, 0.5, summary_text, 
        #                       fontsize=11, ha='left', va='center',
        #                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        # axs[current_row].set_xlim(0, 1)
        # axs[current_row].set_ylim(0, 1)
        # axs[current_row].axis('off')
        # axs[current_row].set_title("Summary Statistics", fontweight='bold')
        # current_row += 1

        # Plot metrics
        for i, (key, values) in enumerate(metrics.items()):
            row = i + current_row
            axs[row].plot(values, linestyle='--', marker='o')
            axs[row].set_ylabel(key)
            axs[row].set_title(f"{key} over time")

            # Add some visual enhancements
            if key in ['q', 'target_q', 'q_ood', 'target_q_ood']:
                axs[row].set_xlabel('Time Step')
                # Highlight differences between in-distribution and OOD
                if 'ood' in key:
                    axs[row].set_facecolor('#fff5f5')  # Light red background for OOD
                else:
                    axs[row].set_facecolor('#f5fff5')  # Light green background for in-distribution

        plt.tight_layout()
        canvas.draw()
        out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return out_image

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        # Example arrays for model init
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        shared_encoder: bool = False,
        # Goal conditioning (follow CQL pattern)
        goals: Optional[Data] = None,
        early_goal_concat: bool = False,
        shared_goal_encoder: bool = True,
        language_conditioned: bool = False,
        goal_conditioned: bool = False,
        # Network config (match config naming)
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
            "activate_final": True,
            "use_layer_norm": False,
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        use_min_q: bool = True,
        # OOD evaluation config
        num_ood_actions: int = 10, # NEW
        ood_action_sample_method: str = "uniform",  # "uniform" or "normal" NEW
        # Optimizer config (follow CQL pattern)
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        # Algorithm config
        discount: float = 0.98,
        soft_target_update_rate: Optional[float] = None,
        target_update_rate: Optional[float] = None,  # Support both naming conventions
        **kwargs,
    ):
        """Create SARSA agent (follow CQL/SAC pattern)"""
        
        # Handle naming compatibility
        if soft_target_update_rate is None and target_update_rate is not None:
            soft_target_update_rate = target_update_rate
        elif soft_target_update_rate is None:
            soft_target_update_rate = 5e-3
        
        # Language conditioning requires goal conditioning (follow CQL pattern)
        if language_conditioned:
            assert goal_conditioned or True, "Language conditioning requires goal conditioning"
            goal_conditioned = True  # Auto-enable goal conditioning
        
        # Create encoder (follow CQL _create_encoder_def pattern)
        encoder_def = cls._create_encoder_def(
            encoder_def,
            use_proprio=False,
            enable_stacking=False,
            goal_conditioned=goal_conditioned,
            early_goal_concat=early_goal_concat,
            shared_goal_encoder=shared_goal_encoder,
            language_conditioned=language_conditioned,
        )

        encoders = {
            "critic": encoder_def,
        }

        # Create critic network (follow CQL pattern)
        critic_backbone = partial(MLP, **network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")

        # Only need critic - no actor/policy!
        networks = {"critic": critic_def}
        model_def = ModuleDict(networks)

        # Optimizer (follow CQL pattern with make_optimizer)
        txs = {
            "critic": make_optimizer(
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
            ),
        }

        # Initialize parameters
        rng, init_rng = jax.random.split(rng)
        network_input = (observations, goals) if goal_conditioned else observations
        params = model_def.init(
            init_rng,
            critic=[network_input, actions],
        )["params"]

        # Create training state
        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,  # Initialize target = current
            rng=create_rng,
        )

        # Configuration (follow CQL pattern)
        config = {
            "critic_ensemble_size": critic_ensemble_size,
            "critic_subsample_size": critic_subsample_size,
            "discount": discount,
            "soft_target_update_rate": soft_target_update_rate,
            "goal_conditioned": goal_conditioned,
            "language_conditioned": language_conditioned,
            "use_min_q": use_min_q,
            # OOD evaluation config
            "num_ood_actions": num_ood_actions,
            "ood_action_sample_method": ood_action_sample_method,
        }

        return cls(state=state, config=config)

    @classmethod
    def _create_encoder_def(
        cls,
        encoder_def: nn.Module,
        use_proprio: bool,
        enable_stacking: bool,
        goal_conditioned: bool,
        early_goal_concat: bool,
        shared_goal_encoder: bool,
        language_conditioned: bool,
    ):
        """Create encoder definition (copied from CQL/SAC)"""
        if goal_conditioned and not language_conditioned:
            if early_goal_concat:
                goal_encoder_def = None
            else:
                goal_encoder_def = (
                    encoder_def if shared_goal_encoder else copy.deepcopy(encoder_def)
                )

            encoder_def = GCEncodingWrapper(
                encoder=encoder_def,
                goal_encoder=goal_encoder_def,
                use_proprio=use_proprio,
                stop_gradient=False,
            )
        elif language_conditioned:
            if shared_goal_encoder is not None or early_goal_concat is not None:
                raise ValueError(
                    "If language conditioned, shared_goal_encoder and early_goal_concat must not be set"
                )
            encoder_def = LCEncodingWrapper(
                encoder=encoder_def,
                use_proprio=use_proprio,
                stop_gradient=False,
            )
        else:
            encoder_def = EncodingWrapper(
                encoder_def,
                use_proprio=use_proprio,
                stop_gradient=False,
                enable_stacking=enable_stacking,
            )

        return encoder_def

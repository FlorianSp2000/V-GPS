"""
Ensemble training with single iterator approach - much cleaner and more efficient.
"""
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags, logging
from flax.training import checkpoints
from ml_collections import config_flags

from jaxrl_m.agents import agents
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.vision import encoders
from jaxrl_m.data.text_processing import text_processors
import wandb
from jax.experimental.compilation_cache import compilation_cache

from octo.data.dataset import make_interleaved_dataset_ensemble, make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.utils.train_callbacks import create_validation_dataset
from octo.utils.train_utils import filter_eval_datasets

compilation_cache.initialize_cache("/tmp/jax_compilation_cache")

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_string("project", "jaxrl_m_bridgedata_ensemble", "WandB project name.")
# flags.DEFINE_integer("ensemble_size", None, "Number of ensemble members.")
flags.DEFINE_integer("batch_size_per_member", None, "Batch size per ensemble member.")
flags.DEFINE_string("batching_method", "reshape", "Ensemble batching method: 'reshape' (recommended) or 'nested'.")

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
config_flags.DEFINE_config_file("oxedata_config", None, "Data configuration.", lock_config=False)


def create_ensemble_agents(ensemble_size, rng, template_batch, encoder_def, agent_class, agent_kwargs):
    """
    Create ensemble agents with vectorized creation if available.
    """
    logging.info("Creating ensemble agents...")
    
    # Check if agent class supports vectorized ensemble creation
    if hasattr(agent_class, 'create_ensemble_vectorized'):
        logging.info("Using vectorized ensemble agent creation")
        ensemble_agents = agent_class.create_ensemble_vectorized(
            base_rng=rng,
            ensemble_size=ensemble_size,
            observations=template_batch["observations"],
            actions=template_batch["actions"],
            encoder_def=encoder_def,
            goals=template_batch["goals"],
            **agent_kwargs
        )
    else:
        logging.info("Using standard ensemble agent creation")
        agent_rngs = jax.random.split(rng, ensemble_size)
        
        agents_list = []
        for member_idx, agent_rng in enumerate(agent_rngs):
            agent = agent_class.create(
                rng=agent_rng,
                observations=template_batch["observations"],
                goals=template_batch["goals"],
                actions=template_batch["actions"],
                encoder_def=encoder_def,
                **agent_kwargs,
            )
            agents_list.append(agent)
        
        # Stack into ensemble structure for pmap
        ensemble_agents = jax.tree_map(lambda *args: jnp.stack(args, axis=0), *agents_list)
    
    return ensemble_agents


def get_ensemble_val_batch(val_iterator, ensemble_size):
    """
    Convert single validation batch to ensemble format by duplication.
    This is much more efficient than multiple iterators.
    """
    single_batch = next(val_iterator)
    # Duplicate across ensemble dimension: (batch_size,) → (ensemble_size, batch_size)
    ensemble_batch = jax.tree_map(
        lambda x: jnp.repeat(x[None], ensemble_size, axis=0), 
        single_batch
    )
    return ensemble_batch


def aggregate_ensemble_metrics_cpu(ensemble_metrics):
    """
    Aggregate metrics that are already on CPU after device_get.
    """
    def compute_stats(x):
        return {
            'mean': jnp.mean(x, axis=0),
            'std': jnp.std(x, axis=0), 
            # 'min': jnp.min(x, axis=0),
            # 'max': jnp.max(x, axis=0),
        }
    
    # Apply to all metrics - this runs on CPU so vmap is fine
    aggregated = jax.tree_map(compute_stats, ensemble_metrics)
    
    # Flatten for logging
    flattened = {}
    for metric_name, stats in aggregated.items():
        # logging.info(f"Aggregated metric {metric_name} has stats: {stats}")
        for stat_name, values in stats.items():
            # Create clear names for ensemble disagreement metrics
            if stat_name == 'std' and 'ood_q' in metric_name:
                # This is the key metric for OOD detection: disagreement in OOD Q-values
                flattened[f"ensemble_disagreement_ood_q"] = values
            elif stat_name == 'std' and 'online_q' in metric_name:
                # This is disagreement in online Q-values
                flattened[f"ensemble_disagreement_online_q"] = values
            elif stat_name == 'std' and 'target_q' in metric_name:
                # This is disagreement in target Q-values
                flattened[f"ensemble_disagreement_target_q"] = values
            else:
                # Standard naming
                flattened[f"{stat_name}_{metric_name}"] = values

    return flattened


def prepare_member_metrics_cpu(ensemble_metrics, ensemble_size):
    """
    Extract individual member metrics on CPU.
    """
    member_metrics_list = []
    for member_idx in range(ensemble_size):
        member_metrics = jax.tree_map(lambda x: x[member_idx], ensemble_metrics)
        member_metrics_list.append(member_metrics)
    return member_metrics_list


def batch_log_metrics(wandb_logger, aggregated_metrics, member_metrics_list, step, prefix=""):
    """
    Batch logging to reduce wandb API calls.
    """
    log_dict = {}
    
    # Add aggregated metrics
    for key, value in aggregated_metrics.items():
        # logging.info(f"Logging aggregated metric {prefix}{key} with value {value}")
        log_dict[f"{prefix}{key}"] = value
    
    # Add individual member metrics
    for member_idx, member_metrics in enumerate(member_metrics_list):
        # break up after first 2 members to avoid too many logs
        if member_idx >= 2:
            break
        for key, value in member_metrics.items():
            log_dict[f"{prefix}member_{member_idx}_{key}"] = value
            
    # Single wandb call
    # logging.info(f"log_dict looks: {log_dict}")
    wandb_logger.log(log_dict, step=step)


def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    
    ensemble_size = FLAGS.config.ensemble_size or num_devices
    batch_size_per_member = FLAGS.batch_size_per_member or (FLAGS.config.batch_size // ensemble_size)
    
    logging.info(f"Ensemble training: {ensemble_size} models, {batch_size_per_member} batch size per member")
    
    tf.config.set_visible_devices([], "GPU")

    # Setup wandb
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update({
        "project": FLAGS.project,
        "exp_descriptor": f"{FLAGS.name}",
        "tags": ["ensemble", f"ensemble_size_{ensemble_size}"],
        "group": f"{FLAGS.name}"
    })
    
    variant = FLAGS.config.to_dict()
    variant.update({
        "oxe_config": FLAGS.oxedata_config.to_dict(),
        "ensemble_size": ensemble_size,
        "batch_size_per_member": batch_size_per_member,
    })
    
    wandb_logger = WandBLogger(wandb_config=wandb_config, variant=variant)

    save_dir = tf.io.gfile.join(
        FLAGS.config.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )
    
    # Text processor setup
    text_processor = None
    if FLAGS.config.get("text_processor"):
        text_processor = text_processors[FLAGS.config.text_processor](**FLAGS.config.text_processor_kwargs)

    def process_text(batch, keep_language_str=False):
        decoded_strings = [s.decode("utf-8") for s in batch["goals"]["language"]]
        if text_processor is not None:
            batch["goals"]["language"] = text_processor.encode(decoded_strings)
        if keep_language_str:
            batch["goals"]["language_str"] = decoded_strings
        return batch

    def process_oxe_batch(batch, keep_language_str=False, training=False):
        """Preprocess training or validation batch. Set keep_language_str=True for validation."""
        
        def reshape_to_ensemble(x):
            return x.reshape(ensemble_size, batch_size_per_member, *x.shape[1:])

        pre_batch = {
            "actions": batch["action"].squeeze(),
            "next_actions": batch["next_action"].squeeze(),
            "goals": {
                "language": batch["task"]["language_instruction"]
            },
            "mc_returns": batch["mc_return"],
            "observations": {"image": batch["observation"]["image_primary"].squeeze()},
            "next_observations": {"image": batch["next_observation"]["image_primary"].squeeze()},
            "rewards": batch["reward"],
            "masks": batch["td_mask"],
        }

        processed_batch = process_text(pre_batch, keep_language_str=keep_language_str)

        if training: # in training, we want to reshape (batch_size,) → (ensemble_size, batch_size // ensemble_size)
            ensemble_batch = jax.tree_map(reshape_to_ensemble, processed_batch)
        else: # in validation, we want to keep the batch size as is as we are duplicating the batch
            ensemble_batch = processed_batch 

        return ensemble_batch

    # Dataset creation with OXE preprocessing
    if "oxe_kwargs" in FLAGS.oxedata_config:
        (
            FLAGS.oxedata_config["dataset_kwargs_list"],
            FLAGS.oxedata_config["sample_weights"],
        ) = make_oxe_dataset_kwargs_and_weights(**FLAGS.oxedata_config["oxe_kwargs"])
        oxe_kwargs = FLAGS.oxedata_config["oxe_kwargs"]
        del FLAGS.oxedata_config["oxe_kwargs"]

    # train_data = make_interleaved_dataset_ensemble(
    #     ensemble_size=ensemble_size,
    #     batch_size_per_member=batch_size_per_member,
    #     **dataset_config,
    #     train=True,
    # )
    logging.info(f"FLAGS.oxedata_config is {FLAGS.oxedata_config}")
    train_data = make_interleaved_dataset(
        **FLAGS.oxedata_config, train=True
    )

    if "fractal" in oxe_kwargs.data_mix or "oxe" in oxe_kwargs.data_mix or "rtx" in oxe_kwargs.data_mix:
        val_datasets_kwargs_list, _ = filter_eval_datasets(
            FLAGS.oxedata_config["dataset_kwargs_list"],
            FLAGS.oxedata_config["sample_weights"],
            ["fractal20220817_data"],
        )

        val_data_fractal = create_validation_dataset(
            val_datasets_kwargs_list[0], 
            FLAGS.oxedata_config["traj_transform_kwargs"],
            FLAGS.oxedata_config["frame_transform_kwargs"],
            train=False
        )

        val_traj_data_fractal_iter = map(partial(process_oxe_batch, keep_language_str=True), val_data_fractal.shuffle(1000).repeat().iterator())

        val_data_fractal_iter = (
            val_data_fractal.unbatch()
            .shuffle(1000)
            .repeat()
            .batch(batch_size_per_member)  # Single member batch size
            .iterator(prefetch=0))

        val_data_fractal_iter = map(process_oxe_batch, val_data_fractal_iter)

        prev_val_traj_fractal = next(val_traj_data_fractal_iter)

    else:
        val_data_fractal_iter = None


    # Single training iterator that produces ensemble-shaped batches
    train_iterator = map(partial(process_oxe_batch, training=True), train_data.iterator(prefetch=0))

    # Validation setup - single iterator approach
    val_datasets_kwargs_list, _ = filter_eval_datasets(
        FLAGS.oxedata_config["dataset_kwargs_list"],
        FLAGS.oxedata_config["sample_weights"],
        ["bridge_dataset"],
    )
    val_data = create_validation_dataset(
        val_datasets_kwargs_list[0],
        FLAGS.oxedata_config["traj_transform_kwargs"],
        FLAGS.oxedata_config["frame_transform_kwargs"],
        train=False
    )
    
    # Single validation iterator - we'll duplicate samples for ensemble evaluation
    val_iter = (
        val_data.unbatch()
        .shuffle(1000)
        .repeat()
        .batch(batch_size_per_member)  # Single member batch size
        .iterator(prefetch=0)
    )
    val_iter = map(process_oxe_batch, val_iter)

    # Trajectory data for plotting (single trajectories, not ensemble)
    val_traj_data_iter = map(
        partial(process_oxe_batch, keep_language_str=True), val_data.shuffle(1000).repeat().iterator()
        )
    
    prev_val_traj = next(val_traj_data_iter)
    logging.info(f"Val traj keys: {list(prev_val_traj.keys())}")
    logging.info(f"Val traj actions shape: {prev_val_traj['actions'].shape}")
    
    # Get example batch and create template for agent initialization
    example_batch = next(train_iterator)  # Already ensemble-shaped!
    # logging.info(f"example_batch {example_batch}")
    template_batch = jax.tree_map(lambda x: x[0], example_batch)  # Extract single member for template
    
    logging.info(f"Ensemble batch shape: {jax.tree_map(lambda x: x.shape, example_batch)}")
    logging.info(f"Template batch shape: {jax.tree_map(lambda x: x.shape, template_batch)}")

    # Encoder setup
    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)

    # Create ensemble agents
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    agent_class = agents[FLAGS.config.agent]
    ensemble_agents = create_ensemble_agents(
        ensemble_size, rng, template_batch, encoder_def, agent_class, FLAGS.config.agent_kwargs
    )

    # Checkpoint restoration
    if FLAGS.config.resume_path:
        restored_agents = []
        for member_idx in range(ensemble_size):
            member_checkpoint_path = os.path.join(FLAGS.config.resume_path, f"ensemble_member_{member_idx}")
            if tf.io.gfile.exists(member_checkpoint_path):
                agent = jax.tree_map(lambda x: x[member_idx], ensemble_agents)
                agent = checkpoints.restore_checkpoint(member_checkpoint_path, target=agent)
                logging.info("Restored ensemble member %d from %s", member_idx, member_checkpoint_path)
            else:
                agent = jax.tree_map(lambda x: x[member_idx], ensemble_agents)
            restored_agents.append(agent)
        ensemble_agents = jax.tree_map(lambda *args: jnp.stack(args, axis=0), *restored_agents)

    # Create pmapped functions for cross-device parallelization
    pmapped_update = jax.pmap(lambda agent, batch: agent.update(batch))
    pmapped_debug_metrics = jax.pmap(lambda agent, batch, rng_key: agent.get_debug_metrics(batch, seed=rng_key))

    timer = Timer()
    
    # Training loop - much cleaner now!
    for i in tqdm.tqdm(range(int(FLAGS.config.num_steps))):
        timer.tick("total")

        timer.tick("dataset")
        # Get ensemble batch directly - no complex iterator management!
        ensemble_batch = next(train_iterator)
        timer.tock("dataset")

        timer.tick("train")
        # Parallel training across devices
        ensemble_agents, ensemble_update_infos = pmapped_update(ensemble_agents, ensemble_batch)

        timer.tock("train")

        if (i + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating ensemble...")
            timer.tick("val")
            
            # Validation evaluation - much simpler now
            val_metrics_list = []
            for _ in range(8):
                # Get ensemble validation batch by duplicating single batch
                ensemble_val_batch = get_ensemble_val_batch(val_iter, ensemble_size)
                rng, val_rng = jax.random.split(rng)
                val_rngs = jax.random.split(val_rng, ensemble_size)
                val_metrics = pmapped_debug_metrics(ensemble_agents, ensemble_val_batch, val_rngs)
                val_metrics_list.append(val_metrics)
            
            # Compute validation statistics (this happens on CPU after device_get)
            avg_val_metrics = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *val_metrics_list)

            # Move to CPU for aggregation
            avg_val_metrics_cpu = jax.device_get(avg_val_metrics)
            # logging.info(f"avg_val_metrics_cpu looks: {avg_val_metrics_cpu}")
            # CPU-based aggregation and member extraction
            aggregated_val_metrics = aggregate_ensemble_metrics_cpu(avg_val_metrics_cpu)
            member_val_metrics = prepare_member_metrics_cpu(avg_val_metrics_cpu, ensemble_size)
            
            batch_log_metrics(wandb_logger, aggregated_val_metrics, member_val_metrics, i, "validation/")

            if val_data_fractal_iter is not None:
                val_metrics_list = []
                for _ in range(8):
                    ensemble_val_batch = get_ensemble_val_batch(val_data_fractal_iter, ensemble_size)
                    rng, val_rng = jax.random.split(rng)
                    val_rngs = jax.random.split(val_rng, ensemble_size)
                    val_metrics = pmapped_debug_metrics(ensemble_agents, ensemble_val_batch, val_rngs)
                    val_metrics_list.append(val_metrics)                    
                # take the mean of the metrics across the 8 iterations
                avg_val_metrics = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *val_metrics_list)
                avg_val_metrics_cpu = jax.device_get(avg_val_metrics)

                aggregated_val_metrics = aggregate_ensemble_metrics_cpu(avg_val_metrics_cpu)
                
                member_val_metrics = prepare_member_metrics_cpu(avg_val_metrics_cpu, ensemble_size)
                
                batch_log_metrics(wandb_logger, aggregated_val_metrics, member_val_metrics, i, "validation/fractal/")

            # Simplified plotting (reduced frequency to avoid overhead)
            if "sarsa" in FLAGS.config.agent:
                logging.info("Plotting value functions...")
                for num in range(2):
                    for member_idx in range(min(1, ensemble_size)):
                        traj = next(val_traj_data_iter)
                        rng, val_rng = jax.random.split(rng)
                        
                        single_agent = jax.tree_map(lambda x: x[member_idx], ensemble_agents)
                        # logging.info(f"traj should contain {traj['goals']['language_str']}")
                        plot = single_agent.plot_values(traj, seed=val_rng)
                        plot = wandb.Image(plot)
                        wandb_logger.log({f"value_plots/member_{member_idx}/traj_{num}": plot}, step=i)

                        plot = single_agent.plot_values(traj, seed=val_rng, goals=prev_val_traj["goals"])
                        plot = wandb.Image(plot)
                        wandb_logger.log({f"value_plots/member_{member_idx}/traj_random_lang_{num}": plot}, step=i)
                    # all ensemble members should operate on same trajectory for comparability
                    prev_val_traj = traj
                
                # NEW: Plot ensemble statistics trajectories
                logging.info("Plotting ensemble statistics...")
                for num in range(2):  # Fewer plots for ensemble stats
                    traj = next(val_traj_data_iter)
                    rng, val_rng = jax.random.split(rng)
                    val_rngs = jax.random.split(val_rng, ensemble_size)
                    
                    # Plot with original goals
                    plot = agent_class.plot_ensemble_trajectory_values(ensemble_agents, ensemble_size, traj, val_rngs)
                    plot = wandb.Image(plot)
                    wandb_logger.log({f"ensemble_value_plots/traj_{num}": plot}, step=i)
                    
                    # Plot with random goals (language grounding test)
                    plot = agent_class.plot_ensemble_trajectory_values(ensemble_agents, ensemble_size, traj, val_rngs, goals=prev_val_traj["goals"])
                    plot = wandb.Image(plot)
                    wandb_logger.log({f"ensemble_value_plots/traj_random_lang_{num}": plot}, step=i)
                    prev_val_traj = traj

                if val_data_fractal_iter is not None:
                    logging.info("Plotting value functions for fractal..")
                    for num in range(2):
                        for member_idx in range(min(1, ensemble_size)):
                            traj = next(val_traj_data_fractal_iter)
                            rng, val_rng = jax.random.split(rng)

                            single_agent = jax.tree_map(lambda x: x[member_idx], ensemble_agents)
                            plot = single_agent.plot_values(traj, seed=val_rng)
                            plot = wandb.Image(plot)
                            wandb_logger.log({f"value_plots/member_{member_idx}/fractal/traj_{num}": plot}, step=i)

                            plot = single_agent.plot_values(traj, seed=val_rng, goals=prev_val_traj_fractal["goals"])
                            plot = wandb.Image(plot)
                            wandb_logger.log({f"value_plots/member_{member_idx}/fractal/traj_random_lang_{num}": plot}, step=i)

                        prev_val_traj_fractal = traj
                    # NEW: Fractal ensemble plots  
                    for num in range(2):
                        traj = next(val_traj_data_fractal_iter)
                        rng, val_rng = jax.random.split(rng)
                        val_rngs = jax.random.split(val_rng, ensemble_size)
                        
                        plot = agent_class.plot_ensemble_trajectory_values(ensemble_agents, ensemble_size, traj, val_rngs)
                        plot = wandb.Image(plot)
                        wandb_logger.log({f"ensemble_value_plots/fractal/traj_{num}": plot}, step=i)
                        
                        plot = agent_class.plot_ensemble_trajectory_values(ensemble_agents, ensemble_size, traj, val_rngs, goals=prev_val_traj["goals"])
                        plot = wandb.Image(plot)
                        wandb_logger.log({f"ensemble_value_plots/fractal/traj_random_lang_{num}": plot}, step=i)

                        prev_val_traj_fractal = traj

            timer.tock("val")

        if (i + 1) % FLAGS.config.save_interval == 0:
            logging.info("Saving ensemble checkpoints...")
            for member_idx in range(ensemble_size):
                member_agent = jax.tree_map(lambda x: x[member_idx], ensemble_agents)
                member_save_dir = os.path.join(save_dir, f"ensemble_member_{member_idx}")
                checkpoints.save_checkpoint(member_save_dir, member_agent, step=i + 1, keep=100)

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            # Training metrics logging
            ensemble_update_infos_cpu = jax.device_get(ensemble_update_infos)
            
            # CPU-based aggregation
            aggregated_training_metrics = aggregate_ensemble_metrics_cpu(ensemble_update_infos_cpu)
            member_training_metrics = prepare_member_metrics_cpu(ensemble_update_infos_cpu, ensemble_size)
            
            # Batch log training metrics
            batch_log_metrics(wandb_logger, aggregated_training_metrics, member_training_metrics, i, "training/")
            
            # Log timer info
            wandb_logger.log({"timer": timer.get_average_times()}, step=i)


if __name__ == "__main__":
    app.run(main)
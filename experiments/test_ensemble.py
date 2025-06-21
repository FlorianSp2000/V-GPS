"""
Ensemble vs Single Agent Evaluation Script
Loads checkpoints and compares performance side by side.
"""
import os
import glob
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

from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.utils.train_callbacks import create_validation_dataset
from octo.utils.train_utils import filter_eval_datasets

compilation_cache.initialize_cache("/tmp/jax_compilation_cache")

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_string("project", "ensemble_test_comparison", "WandB project name.")
flags.DEFINE_string("ensemble_checkpoint_path", "", "Path to ensemble checkpoint directory.")
flags.DEFINE_string("single_checkpoint_path", "", "Path to single agent checkpoint directory.")

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
config_flags.DEFINE_config_file("oxedata_config", None, "Data configuration.", lock_config=False)


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in a directory."""
    if not tf.io.gfile.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    # Look for checkpoint_* subdirectories
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_*")
    checkpoint_dirs = tf.io.gfile.glob(checkpoint_pattern)
    
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    # Extract step numbers and find the latest
    steps = []
    for checkpoint_path in checkpoint_dirs:
        try:
            step = int(os.path.basename(checkpoint_path).split("_")[1])
            steps.append((step, checkpoint_path))
        except (ValueError, IndexError):
            continue
    
    if not steps:
        raise ValueError(f"No valid checkpoints found in {checkpoint_dir}")
    
    # Return the path with the highest step number
    latest_step, latest_path = max(steps, key=lambda x: x[0])
    logging.info(f"Found latest checkpoint at step {latest_step}: {latest_path}")
    return latest_path


def load_ensemble_agents(ensemble_checkpoint_path, ensemble_size, checkpoint_step, template_batch, encoder_def, agent_class, seed, agent_kwargs):
    """Load ensemble agents from checkpoints."""
    logging.info(f"Loading ensemble agents from {ensemble_checkpoint_path}")
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    ensemble_agents = agent_class.create_ensemble_vectorized(
        base_rng=rng,
        ensemble_size=ensemble_size,  # Each saved member is a single ensemble member
        observations=template_batch["observations"],
        actions=template_batch["actions"],
        encoder_def=encoder_def,
        goals=template_batch["goals"],
        **agent_kwargs
    )

    restored_agents = []
    for member_idx in range(ensemble_size):
        member_checkpoint_dir = os.path.join(ensemble_checkpoint_path, f"ensemble_member_{member_idx}")
        
        if checkpoint_step is not None:
            member_checkpoint_path = os.path.join(member_checkpoint_dir, f"checkpoint_{checkpoint_step}")
        else:
            member_checkpoint_path = find_latest_checkpoint(member_checkpoint_dir)
        
        if not tf.io.gfile.exists(member_checkpoint_path):
            raise ValueError(f"Ensemble member checkpoint does not exist: {member_checkpoint_path}")
        
        agent = jax.tree_map(lambda x: x[member_idx], ensemble_agents)
        agent = checkpoints.restore_checkpoint(member_checkpoint_path, target=agent)
        restored_agents.append(agent)
        # logging.info(f"member state: {jax.tree_util.tree_structure(member_state)}")
        logging.info(f"Restored ensemble member {member_idx} from {member_checkpoint_path}")

    ensemble_agents = jax.tree_map(lambda *args: jnp.stack(args, axis=0), *restored_agents)  
    logging.info("Successfully stacked all ensemble member states")
    logging.info(f"Full ensemble state shape: {jax.tree_map(lambda x: x.shape, ensemble_agents)}")

    return ensemble_agents

def load_single_agent(single_checkpoint_path, checkpoint_step, template_batch, encoder_def, agent_class, seed, agent_kwargs):
    """Load single agent from checkpoint."""
    logging.info(f"Loading single agent from {single_checkpoint_path}")
    
    if checkpoint_step is not None:
        checkpoint_path = os.path.join(single_checkpoint_path, f"checkpoint_{checkpoint_step}")
    else:
        checkpoint_path = find_latest_checkpoint(single_checkpoint_path)
    
    if not tf.io.gfile.exists(checkpoint_path):
        raise ValueError(f"Single agent checkpoint does not exist: {checkpoint_path}")
    
    # Create dummy agent
    rng = jax.random.PRNGKey(seed)
    agent = agent_class.create(
        rng=rng,
        observations=template_batch["observations"],
        goals=template_batch["goals"],
        actions=template_batch["actions"],
        encoder_def=encoder_def,
        **agent_kwargs,
    )
    
    # Restore from checkpoint
    agent = checkpoints.restore_checkpoint(checkpoint_path, target=agent)
    logging.info(f"Restored single agent from {checkpoint_path}")
    return agent


def get_ensemble_batch(data_iterator, ensemble_size):
    """Convert single data batch to ensemble format by duplication."""
    single_batch = next(data_iterator)
    # Duplicate across ensemble dimension
    ensemble_batch = jax.tree_map(
        lambda x: jnp.repeat(x[None], ensemble_size, axis=0), 
        single_batch
    )
    return ensemble_batch, single_batch


def aggregate_ensemble_metrics_cpu(ensemble_metrics):
    """Aggregate metrics that are already on CPU after device_get."""
    def compute_stats(x):
        return {
            'mean': jnp.mean(x, axis=0),
            'std': jnp.std(x, axis=0),
        }
    
    aggregated = jax.tree_map(compute_stats, ensemble_metrics)
    
    # Flatten for logging
    flattened = {}
    for metric_name, stats in aggregated.items():
        for stat_name, values in stats.items():
            if stat_name == 'std' and 'ood_q' in metric_name:
                flattened[f"ensemble_disagreement_ood_q"] = values
            elif stat_name == 'std' and 'online_q' in metric_name:
                flattened[f"ensemble_disagreement_online_q"] = values
            elif stat_name == 'std' and 'target_q' in metric_name:
                flattened[f"ensemble_disagreement_target_q"] = values
            else:
                flattened[f"{stat_name}_{metric_name}"] = values

    return flattened


def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    
    # Get parameters from config (with command line overrides)
    ensemble_size = FLAGS.config.get("ensemble_size", 8)
    checkpoint_step = FLAGS.config.get("checkpoint_step", None)
    batch_size = FLAGS.config.get("batch_size", 256)
    num_eval_batches = FLAGS.config.get("num_eval_batches", 16)
    num_trajectory_plots = FLAGS.config.get("num_trajectory_plots", 4)
    
    if not FLAGS.ensemble_checkpoint_path:
        raise ValueError("Must specify --ensemble_checkpoint_path")
    if not FLAGS.single_checkpoint_path:
        raise ValueError("Must specify --single_checkpoint_path")
    
    tf.config.set_visible_devices([], "GPU")

    # Setup wandb with test-specific naming
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update({
        "project": FLAGS.project,
        "exp_descriptor": f"ensemble_test_{FLAGS.name}",
        "tags": ["ensemble_test", "comparison", f"ensemble_size_{ensemble_size}"],
        "group": f"ensemble_test_{FLAGS.name}"
    })
    
    variant = FLAGS.config.to_dict()
    variant.update({
        "oxe_config": FLAGS.oxedata_config.to_dict(),
        "ensemble_size": ensemble_size,
        "ensemble_checkpoint_path": FLAGS.ensemble_checkpoint_path,
        "single_checkpoint_path": FLAGS.single_checkpoint_path,
        "checkpoint_step": checkpoint_step,
        "batch_size": batch_size,
        "evaluation_mode": True,
    })
    
    wandb_logger = WandBLogger(wandb_config=wandb_config, variant=variant)

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

    def process_oxe_batch(batch, keep_language_str=False):
        """Preprocess validation batch."""
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
        return process_text(pre_batch, keep_language_str=keep_language_str)

    # Dataset creation
    if "oxe_kwargs" in FLAGS.oxedata_config:
        (
            FLAGS.oxedata_config["dataset_kwargs_list"],
            FLAGS.oxedata_config["sample_weights"],
        ) = make_oxe_dataset_kwargs_and_weights(**FLAGS.oxedata_config["oxe_kwargs"])
        oxe_kwargs = FLAGS.oxedata_config["oxe_kwargs"]
        del FLAGS.oxedata_config["oxe_kwargs"]

    # Create validation datasets
    logging.info("Creating validation datasets...")
    
    # Bridge dataset
    val_datasets_kwargs_list, _ = filter_eval_datasets(
        FLAGS.oxedata_config["dataset_kwargs_list"],
        FLAGS.oxedata_config["sample_weights"],
        ["bridge_dataset"],
    )
    val_data_bridge = create_validation_dataset(
        val_datasets_kwargs_list[0],
        FLAGS.oxedata_config["traj_transform_kwargs"],
        FLAGS.oxedata_config["frame_transform_kwargs"],
        train=False
    )
    
    val_iter_bridge = (
        val_data_bridge.unbatch()
        .shuffle(1000)
        .repeat()
        .batch(batch_size)
        .iterator(prefetch=0)
    )
    val_iter_bridge = map(process_oxe_batch, val_iter_bridge)

    # Trajectory data for plotting
    val_traj_data_iter_bridge = map(
        partial(process_oxe_batch, keep_language_str=True), 
        val_data_bridge.shuffle(1000).repeat().iterator()
    )

    # Fractal dataset (if available)
    val_iter_fractal = None
    val_traj_data_iter_fractal = None
    if "fractal" in oxe_kwargs.data_mix or "oxe" in oxe_kwargs.data_mix or "rtx" in oxe_kwargs.data_mix:
        val_datasets_kwargs_list_fractal, _ = filter_eval_datasets(
            FLAGS.oxedata_config["dataset_kwargs_list"],
            FLAGS.oxedata_config["sample_weights"],
            ["fractal20220817_data"],
        )

        val_data_fractal = create_validation_dataset(
            val_datasets_kwargs_list_fractal[0], 
            FLAGS.oxedata_config["traj_transform_kwargs"],
            FLAGS.oxedata_config["frame_transform_kwargs"],
            train=False
        )

        val_iter_fractal = (
            val_data_fractal.unbatch()
            .shuffle(1000)
            .repeat()
            .batch(batch_size)
            .iterator(prefetch=0)
        )
        val_iter_fractal = map(process_oxe_batch, val_iter_fractal)

        val_traj_data_iter_fractal = map(
            partial(process_oxe_batch, keep_language_str=True), 
            val_data_fractal.shuffle(1000).repeat().iterator()
        )

    # Create template batch for agent initialization
    example_batch = next(val_iter_bridge)
    
    logging.info(f"Example batch shape: {jax.tree_map(lambda x: x.shape, example_batch)}")

    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)

    logging.info("Loading agents...")
    
    # Load ensemble agents (using ensemble agent class)
    ensemble_agent_class = agents["sarsa_ensemble"]  # SARSAEnsembleAgent
    ensemble_agents = load_ensemble_agents(
        FLAGS.ensemble_checkpoint_path,
        ensemble_size,
        checkpoint_step,
        example_batch,
        encoder_def,
        ensemble_agent_class,
        FLAGS.config.seed,
        FLAGS.config.agent_kwargs
    )

    # Load single agent (using single agent class)
    single_agent_class = agents["sarsa"]  # SARSAAgent  
    single_agent = load_single_agent(
        FLAGS.single_checkpoint_path,
        checkpoint_step,
        example_batch,
        encoder_def,
        single_agent_class,
        FLAGS.config.seed,
        FLAGS.config.agent_kwargs
    )

    # Create pmapped functions
    pmapped_ensemble_debug_metrics = jax.pmap(lambda agent, batch, rng_key: agent.get_debug_metrics(batch, seed=rng_key))
    # single_debug_metrics = jax.jit(lambda agent, batch, rng_key: agent.get_debug_metrics(batch, seed=rng_key)) # TODO: redundant

    logging.info("Starting evaluation...")

    # Evaluate on Bridge dataset
    logging.info("Evaluating on Bridge dataset...")
    
    ensemble_val_metrics_list = []
    single_val_metrics_list = []

    rng = jax.random.PRNGKey(FLAGS.config.seed)
    
    for eval_step in range(num_eval_batches):        
        # Create ensemble batch by duplication
        ensemble_val_batch, val_batch = get_ensemble_batch(val_iter_bridge, ensemble_size)

        # Generate random seeds
        rng, eval_rng = jax.random.split(rng)
        ensemble_eval_rngs = jax.random.split(eval_rng, ensemble_size)
        single_eval_rng = jax.random.split(eval_rng, 1)[0]
        
        # Evaluate ensemble
        # logging.info(f"ensemble_agents.state {ensemble_agents.state}")
        # logging.info(f"ensemble_eval_rngs {ensemble_eval_rngs}")
        # logging.info(f"ensemble_val_batch {ensemble_val_batch}")
        ensemble_metrics = pmapped_ensemble_debug_metrics(ensemble_agents, ensemble_val_batch, ensemble_eval_rngs)
        
        # Evaluate single agent
        single_metrics = single_agent.get_debug_metrics(val_batch, seed=single_eval_rng)
        
        ensemble_val_metrics_list.append(ensemble_metrics)
        single_val_metrics_list.append(single_metrics)

    # Aggregate metrics over evaluation batches
    avg_ensemble_metrics = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *ensemble_val_metrics_list)
    avg_single_metrics = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *single_val_metrics_list)

    # Move to CPU for processing
    avg_ensemble_metrics_cpu = jax.device_get(avg_ensemble_metrics)
    avg_single_metrics_cpu = jax.device_get(avg_single_metrics)
    
    # Log ensemble metrics
    aggregated_ensemble_metrics = aggregate_ensemble_metrics_cpu(avg_ensemble_metrics_cpu)
    logging.info(f"Ensemble metrics: {aggregated_ensemble_metrics.keys()}")
    logging.info(f"Single agent metrics: {avg_single_metrics_cpu.keys()}")
    log_dict = {}
    
    # Ensemble metrics
    for key, value in aggregated_ensemble_metrics.items():
        log_dict[f"bridge/ensemble/{key}"] = value
    
    # Single agent metrics
    for key, value in avg_single_metrics_cpu.items():
        log_dict[f"bridge/single/{key}"] = value
    
    # Direct comparison metrics
    log_dict["bridge/comparison/q_diff_mean"] = aggregated_ensemble_metrics["mean_online_q"] - avg_single_metrics_cpu["online_q"]
    log_dict["bridge/comparison/target_q_diff_mean"] = aggregated_ensemble_metrics["mean_target_q"] - avg_single_metrics_cpu["target_q"]
    log_dict["bridge/comparison/ood_q_diff_mean"] = aggregated_ensemble_metrics["mean_ood_q"] - avg_single_metrics_cpu["ood_q"]
    
    wandb_logger.log(log_dict, step=0)

    logging.info(f"Bridge evaluation completed:")
    logging.info(f"  Ensemble mean online_q: {aggregated_ensemble_metrics['mean_online_q']:.3f} ± {aggregated_ensemble_metrics['ensemble_disagreement_online_q']:.3f}")
    logging.info(f"  Single agent online_q: {avg_single_metrics_cpu['online_q']:.3f}")

    # Evaluate on Fractal dataset if available
    if val_iter_fractal is not None:
        logging.info("Evaluating on Fractal dataset...")
        
        ensemble_val_metrics_list = []
        single_val_metrics_list = []
        
        for eval_step in range(num_eval_batches):
            ensemble_val_batch, val_batch = get_ensemble_batch(val_iter_fractal, ensemble_size)
            
            rng, eval_rng = jax.random.split(rng)
            ensemble_eval_rngs = jax.random.split(eval_rng, ensemble_size)
            single_eval_rng = jax.random.split(eval_rng, 1)[0]
            
            ensemble_metrics = pmapped_ensemble_debug_metrics(ensemble_agents, ensemble_val_batch, ensemble_eval_rngs)
            # single_metrics = single_debug_metrics(single_agent, val_batch, single_eval_rng)
            single_metrics = single_agent.get_debug_metrics(val_batch, seed=single_eval_rng)

            ensemble_val_metrics_list.append(ensemble_metrics)
            single_val_metrics_list.append(single_metrics)

        # Aggregate and log fractal metrics
        avg_ensemble_metrics = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *ensemble_val_metrics_list)
        avg_single_metrics = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *single_val_metrics_list)
        
        avg_ensemble_metrics_cpu = jax.device_get(avg_ensemble_metrics)
        avg_single_metrics_cpu = jax.device_get(avg_single_metrics)
        
        aggregated_ensemble_metrics = aggregate_ensemble_metrics_cpu(avg_ensemble_metrics_cpu)
        
        log_dict = {}
        for key, value in aggregated_ensemble_metrics.items():
            log_dict[f"fractal/ensemble/{key}"] = value
        for key, value in avg_single_metrics_cpu.items():
            log_dict[f"fractal/single/{key}"] = value
            
        log_dict["fractal/comparison/q_diff_mean"] = aggregated_ensemble_metrics["mean_online_q"] - avg_single_metrics_cpu["online_q"]
        
        wandb_logger.log(log_dict, step=0)

    # Generate trajectory plots
    logging.info("Generating trajectory plots...")
    
    for plot_idx in range(num_trajectory_plots):
        # Bridge plots
        traj = next(val_traj_data_iter_bridge)
        
        # Single agent plot
        rng, plot_rng = jax.random.split(rng)
        single_plot = single_agent.plot_values(traj, seed=plot_rng)
        single_plot = wandb.Image(single_plot)
        wandb_logger.log({f"trajectory_plots/bridge/single_agent_{plot_idx}": single_plot}, step=0)
        
        # Ensemble plot
        rng, plot_rng = jax.random.split(rng)
        ensemble_plot_rngs = jax.random.split(plot_rng, ensemble_size)
        ensemble_plot = ensemble_agent_class.plot_ensemble_trajectory_values(
            ensemble_agents, ensemble_size, traj, ensemble_plot_rngs
        )
        ensemble_plot = wandb.Image(ensemble_plot)
        wandb_logger.log({f"trajectory_plots/bridge/ensemble_{plot_idx}": ensemble_plot}, step=0)
        
        # Fractal plots if available
        if val_traj_data_iter_fractal is not None:
            traj_fractal = next(val_traj_data_iter_fractal)
            
            rng, plot_rng = jax.random.split(rng)
            single_plot_fractal = single_agent.plot_values(traj_fractal, seed=plot_rng)
            single_plot_fractal = wandb.Image(single_plot_fractal)
            wandb_logger.log({f"trajectory_plots/fractal/single_agent_{plot_idx}": single_plot_fractal}, step=0)
            
            rng, plot_rng = jax.random.split(rng)
            ensemble_plot_rngs = jax.random.split(plot_rng, ensemble_size)
            ensemble_plot_fractal = ensemble_agent_class.plot_ensemble_trajectory_values(
                ensemble_agents, ensemble_size, traj_fractal, ensemble_plot_rngs
            )
            ensemble_plot_fractal = wandb.Image(ensemble_plot_fractal)
            wandb_logger.log({f"trajectory_plots/fractal/ensemble_{plot_idx}": ensemble_plot_fractal}, step=0)

    logging.info("Evaluation completed successfully!")


if __name__ == "__main__":
    app.run(main)
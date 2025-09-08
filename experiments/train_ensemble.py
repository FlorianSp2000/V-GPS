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

from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.utils.train_callbacks import create_validation_dataset, create_deterministic_validation_dataset
from octo.utils.train_utils import filter_eval_datasets

from experiments.utils.finetuning import create_negative_demonstrations
compilation_cache.initialize_cache("/tmp/jax_compilation_cache")

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_string("project", "jaxrl_m_bridgedata_ensemble", "WandB project name.")

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
config_flags.DEFINE_config_file("oxedata_config", None, "Data configuration.", lock_config=False)

def get_latest_checkpoint_step(ckpt_dir, prefix='checkpoint_'):
    """Extract step number from latest checkpoint folder name."""
    if not tf.io.gfile.exists(ckpt_dir):
        return None
    
    try:
        # List all checkpoint folders
        checkpoint_files = tf.io.gfile.listdir(ckpt_dir)
        checkpoint_steps = []
        
        for filename in checkpoint_files:
            if filename.startswith(prefix):
                try:
                    # Extract step number: checkpoint_100000 -> 100000
                    step_str = filename[len(prefix):]
                    step = int(step_str)
                    checkpoint_steps.append(step)
                except ValueError:
                    continue
        
        if checkpoint_steps:
            return max(checkpoint_steps)  # Return highest step number
        else:
            return None
    except Exception as e:
        logging.warning(f"Error reading checkpoint directory {ckpt_dir}: {e}")
        return None

def create_ensemble_agents(ensemble_size, rng, template_batch, encoder_def, agent_class, agent_kwargs):
    """
    Create ensemble agents with vectorized creation if available.
    """
    logging.info("Creating ensemble agents...")
    
    # Check if agent class supports vectorized ensemble creation
    logging.info(f"agent_class {agent_class}")
    # if hasattr(agent_class, 'create_ensemble_vectorized'):
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
    # else:
    #     raise ValueError(f"Agent class {agent_class} does not support vectorized ensemble creation. ")
    return ensemble_agents


def get_ensemble_batch(data_iterator, ensemble_size):
    """
    Convert single data  batch to ensemble format by duplication.
    This is much more efficient than multiple iterators.
    """
    single_batch = next(data_iterator)
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
                # This is the key metric for OOD detection: disagreement in OOD Q-values TODO: does ood here still refer to ood actions / randomly sampled actions?
                flattened[f"ensemble_disagreement_ood_q"] = values
            elif stat_name == 'std' and 'online_q' in metric_name:
                flattened[f"ensemble_disagreement_online_q"] = values
            elif stat_name == 'std' and 'target_q' in metric_name:
                flattened[f"ensemble_disagreement_target_q"] = values
            else:
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
    wandb_logger.log(log_dict, step=step)


def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    
    ensemble_size = FLAGS.config.ensemble_size or num_devices
    batch_size_per_member = FLAGS.config.batch_size // ensemble_size # FLAGS.batch_size_per_member
    # batch_size_per_member = FLAGS.config.batch_size # FLAGS.batch_size_per_member

    logging.info(f"Ensemble training: {ensemble_size} models, batch size {FLAGS.config.batch_size}, {batch_size_per_member} batch size per member")
    logging.info(f"Creating Negative Demos: {FLAGS.config.create_negative_demos}, ratio {FLAGS.config.negative_demo_ratio}, exclude empty language instructions: {FLAGS.config.exclude_empty_lang_instr}")

    tf.config.set_visible_devices([], "GPU")
    tf.random.set_seed(FLAGS.config.seed) # set random tf random seed for create_negative_demonstrations

    resume_step = 0
    if FLAGS.config.resume_path:
        member_checkpoint_path = os.path.join(FLAGS.config.resume_path, "ensemble_member_0") # all members have same latest checkpoint
        resume_step = get_latest_checkpoint_step(member_checkpoint_path) or 0
        logging.info(f"Resuming training from step: {resume_step}")
    # Setup wandb
    if FLAGS.config.get('resume_wandb_id'):
        # Resume existing run - the wandb_id IS the full experiment name
        # Extract unique_identifier from the wandb_id
        wandb_id = FLAGS.config.resume_wandb_id
        if FLAGS.name in wandb_id:
            unique_id = wandb_id.replace(f"{FLAGS.name}_", "", 1)
        else:
            # Fallback: assume the wandb_id is the full experiment_id
            unique_id = wandb_id.split('_')[-1]  # Get timestamp part
        
        wandb_config = WandBLogger.get_default_config()
        wandb_config.update({
            "project": FLAGS.project,
            "exp_descriptor": f"{FLAGS.name}",
            "unique_identifier": unique_id  # This will recreate the same experiment_id
        })
        logging.info(f"Resuming wandb run: {wandb_id} with unique_id: {unique_id}")
    else:
        if FLAGS.config.create_negative_demos:
            # With negative demos: additional tag
            tags = ["ensemble", f"ensemble_size_{ensemble_size}", "negative_demos", f"negdemo_{FLAGS.config.negative_demo_ratio}"]
        else:
            tags = ["ensemble", f"ensemble_size_{ensemble_size}"]

        wandb_config = WandBLogger.get_default_config()
        wandb_config.update({
            "project": FLAGS.project,
            "exp_descriptor": f"{FLAGS.name}",
            "tags": tags,
            "group": f"{FLAGS.name}"
        })
        logging.info("Starting new wandb run")

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
        # Process original language if present
        if "original_language" in batch["goals"]:
            orig_decoded = [s.decode("utf-8") for s in batch["goals"]["original_language"]]
            if text_processor is not None:
                batch["goals"]["original_language"] = text_processor.encode(orig_decoded)
            if keep_language_str:
                batch["goals"]["original_language_str"] = orig_decoded

        return batch

    def process_oxe_batch(batch, keep_language_str=False, training=False):
        """Preprocess training or validation batch. Set keep_language_str=True for validation."""
        
        def reshape_to_ensemble(x):
            return x.reshape(ensemble_size, batch_size_per_member, *x.shape[1:])

        pre_batch = {
            "actions": batch["action"].squeeze(),
            "next_actions": batch["next_action"].squeeze(),
            "goals": {
                "language": batch["task"]["language_instruction"],
                # Add original language if it exists (only for negative demo datasets)
                # Adding additional keys for goals is not an issue despite _include_goals_in_obs because LCEncodingWrapper only looks at the "language" key
                **({"original_language": batch["task"]["original_language_instruction"]} 
                if "original_language_instruction" in batch["task"] else {}) 
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

    logging.info(f"FLAGS.oxedata_config is {FLAGS.oxedata_config}")
    train_data = make_interleaved_dataset(
        **FLAGS.oxedata_config, train=True, 
        exclude_empty_lang_instr=FLAGS.config.exclude_empty_lang_instr, seed=FLAGS.config.seed,
        create_negative_demos=FLAGS.config.create_negative_demos, negative_demo_ratio=FLAGS.config.negative_demo_ratio
    )

    if "fractal" in oxe_kwargs.data_mix or "oxe" in oxe_kwargs.data_mix or "rtx" in oxe_kwargs.data_mix:
        val_datasets_kwargs_list, _ = filter_eval_datasets(
            FLAGS.oxedata_config["dataset_kwargs_list"],
            FLAGS.oxedata_config["sample_weights"],
            ["fractal20220817_data"],
        )

        val_data_fractal = create_deterministic_validation_dataset(
            val_datasets_kwargs_list[0], 
            FLAGS.oxedata_config["traj_transform_kwargs"],
            FLAGS.oxedata_config["frame_transform_kwargs"],
            train=False,
            exclude_empty_lang_instr=FLAGS.config.exclude_empty_lang_instr
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

        if FLAGS.config.create_negative_demos:
            # Create negative demonstrations for validation
            val_data_fractal_filtered = val_data_fractal.filter(lambda traj: traj['task']['language_instruction'][0] != b'')

            val_data_fractal_with_neg, _ = create_negative_demonstrations(
                val_data_fractal_filtered, 
                negative_ratio=1.0,  # 100% negative demos; Then we always have randomly sampled and original prompt per trajectory
                seed=FLAGS.config.seed
            )
            val_data_fractal_negdemo_iter = (
                val_data_fractal_with_neg.unbatch()
                .shuffle(1000)
                .repeat()
                .batch(batch_size_per_member)  # Single member batch size
                .iterator(prefetch=0))
            val_data_fractal_negdemo_iter = map(process_oxe_batch, val_data_fractal_negdemo_iter)

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
    val_data = create_deterministic_validation_dataset(
        val_datasets_kwargs_list[0],
        FLAGS.oxedata_config["traj_transform_kwargs"],
        FLAGS.oxedata_config["frame_transform_kwargs"],
        train=False,
        exclude_empty_lang_instr=FLAGS.config.exclude_empty_lang_instr
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
    
    if FLAGS.config.create_negative_demos:
        # Create negative demonstrations for validation
        val_data_filtered = val_data.filter(lambda traj: traj['task']['language_instruction'][0] != b'')

        val_data_with_neg, _ = create_negative_demonstrations(
            val_data_filtered, 
            negative_ratio=1.0,  # 100% negative demos; Then we always have randomly sampled and original prompt per trajectory
            seed=FLAGS.config.seed
        )
        val_data_negdemo_iter = (
            val_data_with_neg.unbatch()
            .shuffle(1000)
            .repeat()
            .batch(batch_size_per_member)  # Single member batch size
            .iterator(prefetch=0))
        val_data_negdemo_iter = map(process_oxe_batch, val_data_negdemo_iter)
        example_negdemo = next(val_data_negdemo_iter)

    # Get example batch and create template for agent initialization
    example_batch = next(train_iterator)  # Already ensemble-shaped!

    # logging.info(f"example_batch {example_batch}")
    template_batch = jax.tree_map(lambda x: x[0], example_batch)  # Extract single member for template
    
    logging.info(f"Ensemble batch shape: {jax.tree_map(lambda x: x.shape, example_batch)}")
    logging.info(f"Template batch shape: {jax.tree_map(lambda x: x.shape, template_batch)}")
    if example_negdemo:
        logging.info(f"Example negative demo batch shape: {jax.tree_map(lambda x: x.shape, example_negdemo)}")

    # Encoder setup
    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)

    # Create ensemble agents
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    agent_class = agents[FLAGS.config.agent]
    logging.info(f"rng in create_ensemble_agents is {rng}")
    ensemble_agents = create_ensemble_agents(
        ensemble_size, rng, template_batch, encoder_def, agent_class, FLAGS.config.agent_kwargs
    )

    # Checkpoint restoration
    if FLAGS.config.resume_path:
        logging.info("Restoring ensemble agents from checkpoint...")
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
        logging.info(f"RESTORATION: ensemble_agents shape: {jax.tree_map(lambda x: x.shape, ensemble_agents)}")
        # member_agent = jax.tree_map(lambda x: x[0], ensemble_agents)
        # logging.info(f"Single member agent state shape: {jax.tree_map(lambda x: x.shape, member_agent)}")

    # Create pmapped functions for cross-device parallelization
    pmapped_update = jax.pmap(lambda agent, batch: agent.update(batch))
    # broadcast batch data during validation
    pmapped_debug_metrics = jax.pmap(
        lambda agent, batch, rng_key: agent.get_debug_metrics(batch, seed=rng_key),
        in_axes=(0, None, 0),
    )
    # pmapped_debug_metrics = jax.pmap(lambda agent, batch, rng_key: agent.get_debug_metrics(batch, seed=rng_key))
    # pmapped_update = jax.vmap(lambda agent, batch: agent.update(batch))
    # pmapped_debug_metrics = jax.vmap(lambda agent, batch, rng_key: agent.get_debug_metrics(batch, seed=rng_key))

    timer = Timer()
    total_steps = int(FLAGS.config.num_steps)
    logging.info(f"Training from step {resume_step} to {total_steps}")

    for i in tqdm.tqdm(range(resume_step, total_steps)):
        timer.tick("total")

        timer.tick("dataset")
        # Get ensemble batch directly - no complex iterator management!
        ensemble_batch = next(train_iterator)
        timer.tock("dataset")

        timer.tick("train")
        # Parallel training across devices
        ensemble_agents, ensemble_update_infos = pmapped_update(ensemble_agents, ensemble_batch)

        timer.tock("train")

        if i % FLAGS.config.eval_interval == 0: # TODO: removed +1 to check initialization
            logging.info("Evaluating ensemble...")
            timer.tick("val")
            
            # Validation evaluation
            val_metrics_list = []
            # noise_metrics_list = []  # Collect noise metrics over 8 iterations
            for _ in range(6):
                # Get ensemble validation batch by duplicating single batch
                # ensemble_val_batch = get_ensemble_batch(val_iter, ensemble_size)
                single_val_batch = next(val_iter)
                rng, val_rng = jax.random.split(rng)
                val_rngs = jax.random.split(val_rng, ensemble_size)
                # sanity check: compute disagreement with noise observations
                # noise_batch = jax.tree_map(lambda x: x, ensemble_val_batch)  # Deep copy
                # noise_batch["observations"]["image"] = jax.random.normal(rng, noise_batch["observations"]["image"].shape)
                
                # noise_metrics = pmapped_debug_metrics(ensemble_agents, noise_batch, val_rngs)
                val_metrics = pmapped_debug_metrics(ensemble_agents, single_val_batch, val_rngs)

                val_metrics_list.append(val_metrics)
                # noise_metrics_list.append(noise_metrics)
            
            # Compute validation statistics (this happens on CPU after device_get)
            # average over 8 iterations
            avg_val_metrics = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *val_metrics_list)
            # avg_noise_metrics = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *noise_metrics_list)

            # Move to CPU for aggregation
            avg_val_metrics_cpu = jax.device_get(avg_val_metrics)
            # avg_noise_metrics_cpu = jax.device_get(avg_noise_metrics)
            # Compute aggregated disagreement levels
            real_disagreement = jnp.mean(jnp.std(avg_val_metrics_cpu["online_q"], axis=0))
            # noise_disagreement = jnp.mean(jnp.std(avg_noise_metrics_cpu["online_q"], axis=0))
            logging.info(f"Aggregated real data disagreement (6 iterations): {real_disagreement:.3f}")
            # logging.info(f"Aggregated noise data disagreement (6 iterations): {noise_disagreement:.3f}")

            # CPU-based aggregation and member extraction
            aggregated_val_metrics = aggregate_ensemble_metrics_cpu(avg_val_metrics_cpu)
            # aggregated_val_metrics["ensemble_obs_noise_disagreement"] = noise_disagreement
            # aggregated_val_metrics["ensemble_disagreement_obs_noise_ratio"] = noise_disagreement / (real_disagreement + 1e-8)  # Avoid division by zero
            member_val_metrics = [] # prepare_member_metrics_cpu(avg_val_metrics_cpu, ensemble_size)
            
            batch_log_metrics(wandb_logger, aggregated_val_metrics, member_val_metrics, i, "validation/")

            if val_data_fractal_iter is not None:
                val_metrics_list = []
                for _ in range(6):
                    # ensemble_val_batch = get_ensemble_batch(val_data_fractal_iter, ensemble_size)
                    single_fractal_batch = next(val_data_fractal_iter)
                    rng, val_rng = jax.random.split(rng)
                    val_rngs = jax.random.split(val_rng, ensemble_size)
                    # val_metrics = pmapped_debug_metrics(ensemble_agents, ensemble_val_batch, val_rngs)
                    val_metrics = pmapped_debug_metrics(ensemble_agents, single_fractal_batch, val_rngs)
                    val_metrics_list.append(val_metrics)       
                # take the mean of the metrics across the 8 iterations
                avg_val_metrics = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *val_metrics_list)
                avg_val_metrics_cpu = jax.device_get(avg_val_metrics)

                aggregated_val_metrics = aggregate_ensemble_metrics_cpu(avg_val_metrics_cpu)
                
                member_val_metrics = [] # prepare_member_metrics_cpu(avg_val_metrics_cpu, ensemble_size)
                
                batch_log_metrics(wandb_logger, aggregated_val_metrics, member_val_metrics, i, "validation/fractal/")
            
            # Evaluate quantitatively how 1) Q values 2) How disagreement change with non-matching language prompts
            if FLAGS.config.create_negative_demos:
                original_metrics_list = []
                modified_metrics_list = []
                for _ in range(8):
                    # ensemble_negdemo_batch = get_ensemble_batch(val_data_negdemo_iter, ensemble_size)
                    single_negdemo_batch = next(val_data_negdemo_iter)
                    rng, val_rng = jax.random.split(rng)
                    val_rngs = jax.random.split(val_rng, ensemble_size)
                    # Evaluate with ORIGINAL language (should have higher Q-values)
                    original_batch = jax.tree_map(lambda x: x, single_negdemo_batch)  # Copy
                    original_batch["goals"]["language"] = single_negdemo_batch["goals"]["original_language"]
                    original_metrics = pmapped_debug_metrics(ensemble_agents, original_batch, val_rngs)
                    
                    # Evaluate with MODIFIED language (should have lower Q-values)
                    modified_metrics = pmapped_debug_metrics(ensemble_agents, single_negdemo_batch, val_rngs)

                    original_metrics_list.append(original_metrics)
                    modified_metrics_list.append(modified_metrics)
                # Average across 8 iterations
                avg_original_metrics = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *original_metrics_list)
                avg_modified_metrics = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *modified_metrics_list)
                
                # Move to CPU and compute differences
                avg_original_cpu = jax.device_get(avg_original_metrics)
                avg_modified_cpu = jax.device_get(avg_modified_metrics)
                
                # Key metrics: Q-value differences and disagreement changes
                q_difference = jnp.mean(avg_original_cpu["online_q"] - avg_modified_cpu["online_q"])
                original_disagreement = jnp.mean(jnp.std(avg_original_cpu["online_q"], axis=0))
                modified_disagreement = jnp.mean(jnp.std(avg_modified_cpu["online_q"], axis=0))
                
                logging.info(f"Q-value difference (original - modified): {q_difference:.3f}")
                logging.info(f"Disagreement - original: {original_disagreement:.3f}, modified: {modified_disagreement:.3f}")
                negdemo_metrics = {
                    "q_value_difference_lang": q_difference,  # both q values are negative, a positive difference means q values with modified lang prompts are larger?
                    "mean_q_original_lang": jnp.mean(avg_original_cpu["online_q"]),
                    "mean_q_random_lang": jnp.mean(avg_modified_cpu["online_q"]),
                    "q_disagreement_with_orig_lang": original_disagreement,
                    "q_disagreement_with_random_lang": modified_disagreement,
                    "disagreement_ratio_lang": modified_disagreement / (original_disagreement + 1e-8),
                }
                wandb_logger.log({f"validation/negative_demos/{k}": v for k, v in negdemo_metrics.items()}, step=i)
                
                # Do same for fractal data if present
                if val_data_fractal_negdemo_iter is not None:
                    fractal_original_metrics_list = []
                    fractal_modified_metrics_list = []
                    for iteration in range(8):
                        # ensemble_fractal_negdemo_batch = get_ensemble_batch(val_data_fractal_negdemo_iter, ensemble_size)
                        single_fractal_negdemo_batch = next(val_data_fractal_negdemo_iter)
                        rng, val_rng = jax.random.split(rng)
                        val_rngs = jax.random.split(val_rng, ensemble_size)
                        
                        # Evaluate with ORIGINAL language (should have higher Q-values)
                        fractal_original_batch = jax.tree_map(lambda x: x, single_fractal_negdemo_batch)  # Deep copy
                        fractal_original_batch["goals"]["language"] = single_fractal_negdemo_batch["goals"]["original_language"]
                        fractal_original_metrics = pmapped_debug_metrics(ensemble_agents, fractal_original_batch, val_rngs)
                        
                        # Evaluate with MODIFIED language (should have lower Q-values)
                        fractal_modified_metrics = pmapped_debug_metrics(ensemble_agents, single_fractal_negdemo_batch, val_rngs)
                        
                        fractal_original_metrics_list.append(fractal_original_metrics)
                        fractal_modified_metrics_list.append(fractal_modified_metrics)
                    
                    # Average across 8 iterations
                    avg_fractal_original_metrics = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *fractal_original_metrics_list)
                    avg_fractal_modified_metrics = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *fractal_modified_metrics_list)
                    
                    avg_fractal_original_cpu = jax.device_get(avg_fractal_original_metrics)
                    avg_fractal_modified_cpu = jax.device_get(avg_fractal_modified_metrics)
                    
                    fractal_q_difference = jnp.mean(avg_fractal_original_cpu["online_q"] - avg_fractal_modified_cpu["online_q"])
                    fractal_original_disagreement = jnp.mean(jnp.std(avg_fractal_original_cpu["online_q"], axis=0))
                    fractal_modified_disagreement = jnp.mean(jnp.std(avg_fractal_modified_cpu["online_q"], axis=0))
                    
                    logging.info(f"Fractal - Q-value difference (original - modified): {fractal_q_difference:.3f}")
                    logging.info(f"Fractal - Disagreement - original: {fractal_original_disagreement:.3f}, modified: {fractal_modified_disagreement:.3f}")
                    
                    fractal_negdemo_metrics = {
                        "q_value_difference_lang": fractal_q_difference,
                        "mean_q_original_lang": jnp.mean(avg_fractal_original_cpu["online_q"]),
                        "mean_q_random_lang": jnp.mean(avg_fractal_modified_cpu["online_q"]),
                        "q_disagreement_with_orig_lang": fractal_original_disagreement,
                        "q_disagreement_with_random_lang": fractal_modified_disagreement,
                        "disagreement_ratio_lang": fractal_modified_disagreement / (fractal_original_disagreement + 1e-8),
                    }
                    wandb_logger.log({f"validation/negative_demos/fractal/{k}": v for k, v in fractal_negdemo_metrics.items()}, step=i)
            # Simplified plotting (reduced frequency to avoid overhead)
            if "sarsa" in FLAGS.config.agent:
                # logging.info("Plotting single member value functions...")
                # for num in range(2):
                #     traj = next(val_traj_data_iter)
                #     for member_idx in range(min(2, ensemble_size)):
                #         rng, val_rng = jax.random.split(rng)
                        
                #         single_agent = jax.tree_map(lambda x: x[member_idx], ensemble_agents)
                #         # logging.info(f"traj should contain {traj['goals']['language_str']}")
                #         plot = single_agent.plot_values(traj, seed=val_rng)
                #         plot = wandb.Image(plot)
                #         wandb_logger.log({f"value_plots/member_{member_idx}/traj_{num}": plot}, step=i)

                #         plot = single_agent.plot_values(traj, seed=val_rng, goals=prev_val_traj["goals"])
                #         plot = wandb.Image(plot)
                #         wandb_logger.log({f"value_plots/member_{member_idx}/traj_random_lang_{num}": plot}, step=i)
                #     # all ensemble members should operate on same trajectory for comparability
                #     prev_val_traj = traj

                # Plot ensemble value functions
                logging.info("Plotting ensemble value functions...")
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
                    # logging.info("Plotting single member value functions for fractal..")
                    # for num in range(2):
                    #     traj = next(val_traj_data_fractal_iter)
                    #     for member_idx in range(min(2, ensemble_size)):
                    #         rng, val_rng = jax.random.split(rng)

                    #         single_agent = jax.tree_map(lambda x: x[member_idx], ensemble_agents)
                    #         plot = single_agent.plot_values(traj, seed=val_rng)
                    #         plot = wandb.Image(plot)
                    #         wandb_logger.log({f"value_plots/member_{member_idx}/fractal/traj_{num}": plot}, step=i)

                    #         plot = single_agent.plot_values(traj, seed=val_rng, goals=prev_val_traj_fractal["goals"])
                    #         plot = wandb.Image(plot)
                    #         wandb_logger.log({f"value_plots/member_{member_idx}/fractal/traj_random_lang_{num}": plot}, step=i)

                    #     prev_val_traj_fractal = traj
                    # Fractal ensemble plots  
                    logging.info("Plotting ensemble value functions for fractal...")
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
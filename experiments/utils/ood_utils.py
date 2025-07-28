"""Data Preparation Utilities for Offline OOD Experiments"""
import torch
import pickle
from PIL import Image
import numpy as np
from pathlib import Path

from ml_collections import ConfigDict
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.utils.train_utils import filter_eval_datasets
from octo.utils.train_callbacks import create_validation_dataset

def load_configs_for_jupyter(algorithm="ensemble_sarsa", 
                           data_dir="/V-GPS/datasets/open_x", 
                           batch_size=1024, 
                           ensemble_size=8, 
                           seed=44, 
                           resume_path="/V-GPS/results/VGPS_ensemble/VGPS_ensemble_sarsa_bridge_fractal_ens8_b1024s44_20250606_164313",
                           save_dir="/V-GPS/results"):
    """
    Load configs equivalent to train_ensemble.py command line execution.

    :param algorithm: Algorithm type (ensemble_sarsa, ensemble_lc_cql)
    :param data_dir: Path to dataset directory
    :param batch_size: Training batch size
    :param ensemble_size: Number of ensemble members
    :param seed: Random seed
    :param resume_path: Path to resume training from
    :param save_dir: Directory to save checkpoints
    :returns: Tuple of (train_config, data_config) ready for use
    """
    
    # Load base configs (import your actual config files)
    from configs.data_config import get_config as get_data_config
    from configs.train_ensemble_config import get_config as get_train_config
    
    # Get base configurations
    train_config = get_train_config(algorithm)
    data_config = get_data_config()
    
    # Apply command line overrides equivalent to your shell script
    train_config.update({
        "ensemble_size": ensemble_size,
        "num_steps": 510000,
        "seed": seed,
        "batch_size": batch_size,
        "resume_path": resume_path,
        "save_dir": save_dir,
    })
    
    # Update agent discount (from --config.agent_kwargs.discount 0.98)
    train_config.agent_kwargs.discount = 0.98
    
    # Apply data config overrides (from --oxedata_config.batch_size ${BATCH_SIZE})
    data_config.update({
        "batch_size": batch_size,
    })
    
    # Update oxe_kwargs with your specific values
    data_config.oxe_kwargs.update({
        "data_dir": data_dir,
        "data_mix": "bridge_fractal", 
        "discount": 0.98,
    })
        
    return train_config, data_config


def extract_trajectory_data(dataset_name: str, max_trajectories: int = None, keep_full_trajectory=False, keep_raw_instruction=False,
                                save_trajectories: bool = False, save_path: str = None, save_format: str = "pkl"):
    """Extract trajectory data efficiently from oxe dataset. 
    
    Uses Octo tools to create validation set of trajectories of Bridge or Fractal dataset.

    Args:
        dataset_name: "bridge" or "fractal"
        max_trajectories: Max number to extract, None for all; Bridge Validation set is ~7k
        keep_full_trajectory: If True, also keep full trajectory (memory intensive)
        save_trajectories: If True, save extracted data to disk for diffusion pipeline
        save_path: Path to save file (auto-generated if None)
        save_format: "pkl" 
    """
    
    # FLAGS = create_config()
    # oxe_config = create_oxe_config()
    # TODO: Load configs e.g. from files instead of creating them here
    FLAGS, oxe_config = load_configs_for_jupyter(
        algorithm="ensemble_sarsa",
        data_dir="/V-GPS/datasets/open_x",
        batch_size=1024,
        ensemble_size=8,
        seed=44,
        resume_path="/V-GPS/results/VGPS_ensemble/VGPS_ensemble_sarsa_bridge_fractal_ens8_b1024s44_20250606_164313"
    )
    
    # Create dataset kwargs
    (dataset_kwargs_list, sample_weights) = make_oxe_dataset_kwargs_and_weights(**oxe_config["oxe_kwargs"])
        
    # Filter for specific dataset
    if dataset_name == "bridge":
        filter_names = ["bridge_dataset"]
    elif dataset_name == "fractal":
        filter_names = ["fractal20220817_data"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    val_datasets_kwargs_list, _ = filter_eval_datasets(dataset_kwargs_list, sample_weights, filter_names)
    
    # Create validation dataset
    val_data = create_validation_dataset(
        val_datasets_kwargs_list[0],
        oxe_config["traj_transform_kwargs"],
        oxe_config["frame_transform_kwargs"],
        train=False
    )
    
    # Create trajectory iterator
    val_traj_iter = val_data.iterator()
    
    trajectories = []
    print(f"Extracting trajectories from {dataset_name}...")
    print(f"Keep full trajectory: {keep_full_trajectory}")
    
    for i, batch in enumerate(val_traj_iter):
        if max_trajectories is not None and i >= max_trajectories:
            break
            
        images = batch["observation"]["image_primary"].squeeze()  # Shape: (traj_len, 1, 256, 256, 3)
        language_raw = batch["task"]["language_instruction"]  # Shape: (traj_len,) of bytes
        actions = batch['action'].squeeze()
        next_actions = batch['next_action'].squeeze() 

        # Take first language instruction and decode from bytes
        language = language_raw[0].decode("utf-8")
        
        # Extract first and last images efficiently
        first_image = images[0]   # Shape: (1, 256, 256, 3)
        last_image = images[-1]   # Shape: (1, 256, 256, 3)
        first_action, last_action = actions[0], actions[-1]
        first_next_action, last_next_action = next_actions[0], next_actions[-1]

        trajectory_data = {
            'first_image': np.array(first_image),
            'last_image': np.array(last_image),
            'first_action': np.array(first_action),
            'last_action': np.array(last_action),
            'first_next_action': np.array(first_next_action),
            'last_next_action': np.array(last_next_action),
            'language': language,
            'dataset': dataset_name,
            'traj_length': int(images.shape[0]),
            'trajectory_id': i,
            'action': np.array(actions)
        }
        
        if keep_raw_instruction:
            trajectory_data["language_bytes"] = language_raw
        # Optionally keep full trajectory
        if keep_full_trajectory:
            trajectory_data['images'] = images
            trajectory_data['next_actions'] = np.array(next_actions)

        trajectories.append(trajectory_data)
        
        if (i + 1) % 1000 == 0:
            print(f"Extracted {i + 1} trajectories...")
    
    print(f"Extracted {len(trajectories)} trajectories from {dataset_name}")
    # Save for PyTorch diffusion pipeline if requested
    if save_trajectories:
        # Auto-generate save path if not provided
        if save_path is None:
            trajectory_count = len(trajectories)
            max_str = f"_max{max_trajectories}" if max_trajectories else "_all"
            save_path = f"{dataset_name}_trajectories{max_str}_n{trajectory_count}.pkl"
        elif not save_path.endswith('.pkl'):
            save_path += '.pkl'
        
        # Ensure directory exists
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving {len(trajectories)} trajectories to {save_path}...")
        
        # Prepare data with metadata
        save_data = {
            'trajectories': trajectories,
            'metadata': {
                'dataset_name': dataset_name,
                'num_trajectories': len(trajectories),
                'image_shape': trajectories[0]['first_image'].shape if trajectories else None,
                'action_shape': trajectories[0]['first_action'].shape if trajectories else None,
                'format_version': '1.0',
                'created_for': 'pytorch_diffusion_pipeline'
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Saved to: {save_path}")

    return trajectories

"""
Example Usage
from utils.ood_utils import extract_trajectory_data

dataset_name = "fractal"
num_traj = 100
bridge_trajectories_for_diffusion = extract_trajectory_data(
    dataset_name, 
    max_trajectories=num_traj,  # Limit for diffusion training
    keep_full_trajectory=False,
    save_trajectories=True,  # Enable saving
    save_path=f"data/{dataset_name}_diffusion_dataset_traj{num_traj}"  # Custom path
)
"""


def create_inpaintings_from_mask(data: dict, diffusion_pipeline, out_h: int= 1024, out_w: int = 1024, max_seq_len: int =512, save_path: str = None):
    """
    Takes a dict with key 'trajectories' dict with list of dicts with keys 'first_image', 'last_image', 'inpainting_prompt', 'first_mask', 'last_mask'.
    Creates inpainted versions of the first and last images using the provided diffusion pipeline and inpainting prompt.
    Updates trajectory IN-PLACE!

    :param traj: trajectory dict with images, masks and inpainting prompt
    :param diffusion_pipeline: diffusion pipeline for inpainting
    :param out_h: output height for inpainted images
    :param out_w: output width for inpainted images
    :param max_seq_len: maximum sequence length for the inpainting prompt
    :param save_path: path to save the dataset with inpainted images, if None does not save

    :return: None, updates traj in-place
    """
    for idx, traj in enumerate(data['trajectories']):
        if 'inpainted_first_image' in traj.keys():
            continue
        i1 = traj['first_image']
        m1 = traj['first_mask']
        i2 = traj['last_image']
        m2 = traj['last_mask']
        inpaint_prompt = traj['inpainting_prompt']
        
        i1_inpainted = pipe(
            prompt=inpaint_prompt,
            image=Image.fromarray(i1),
            mask_image=Image.fromarray(m1),
            height=out_h, # output res has to be divisible by 16
            width=out_w,
            max_sequence_length=max_seq_len,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

        W, H = i1.shape[:2]
        i1_inpainted = np.array(i1_inpainted.resize((W, H), Image.Resampling.BICUBIC))

        i2_inpainted = pipe(
            prompt=inpaint_prompt,
            image=Image.fromarray(i2),
            mask_image=Image.fromarray(m2),
            height=out_h, # output res has to be divisible by 16
            width=out_w,
            max_sequence_length=max_seq_len,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]
        
        W, H = i2.shape[:2]
        i2_inpainted = np.array(i2_inpainted.resize((W, H), Image.Resampling.BICUBIC))

        traj['inpainted_first_image'] = i1_inpainted
        traj['inpainted_last_image'] = i2_inpainted
        if idx % 10 == 0:
            print(f"Processing trajectory {idx+1}/{len(data['trajectories'])}")
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved inpainted dataset to {save_path}")
    # with open('data/bridge_v0_inpainted.pkl', 'wb') as f:
    else:
        return data
    
"""
# if __name__ == "__main__":
# Example Usage:
# Load Diffusion Pipeline
import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image

repo_id = "black-forest-labs/FLUX.1-Fill-dev"
pipe = FluxFillPipeline.from_pretrained(repo_id, torch_dtype=torch.bfloat16).to("cuda")

# Load Dataset
# with open('data/bridge_v0.pkl', 'rb') as f:
#     data = pickle.load(f)

# Create Inpaintings
# create_inpaintings_from_mask(data, pipe, save_path='data/bridge_v0_inpainted.pkl')

"""
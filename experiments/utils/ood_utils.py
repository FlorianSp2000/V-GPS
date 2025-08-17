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
import tensorflow as tf
import os
from utils.constants import ENSEMBLE_MODEL_PATH

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

def _get_full_dataset_name(short_name: str) -> str:
    """Convert short name to full dataset name for filtering."""
    name_mapping = {
        "bridge": "bridge_dataset",
        "fractal": "fractal20220817_data"
    }
    return name_mapping.get(short_name, short_name)

def _extract_from_single_dataset(val_data, dataset_name, max_count, 
                                keep_full_trajectory, keep_raw_instruction, skip_missing_instruction: bool = True):
    """Extract trajectories from a single dataset."""
    # Create trajectory iterator
    val_traj_iter = val_data.iterator()
    trajectories = []
    print(f"Extracting trajectories from {dataset_name}...")
    print(f"Keep full trajectory: {keep_full_trajectory}")
    
    valid_traj_count = 0  # Count only valid trajectories
    total_processed = 0   # Count all processed trajectories

    for batch in val_traj_iter:
        if max_count and valid_traj_count >= max_count:
            break            
        # if skip_missing_instruction
        total_processed += 1
        images = batch["observation"]["image_primary"].squeeze()  # Shape: (traj_len, 1, 256, 256, 3)
        language_raw = batch["task"]["language_instruction"]  # Shape: (traj_len,) of bytes
        actions = batch['action'].squeeze()
        next_actions = batch['next_action'].squeeze() 

        # Take first language instruction and decode from bytes
        language = language_raw[0].decode("utf-8")
        if skip_missing_instruction and len(language.strip()) == 0:
            if total_processed % 100 == 0:
                print(f"  Processed {total_processed}, found {valid_traj_count} valid trajectories (skipped empty instruction)")
            continue

        trajectory_data = {
            'first_image': np.array(images[0]),
            'last_image': np.array(images[-1]),
            'first_action': np.array(actions[0]),
            'last_action': np.array(actions[-1]),
            'first_next_action': np.array(next_actions[0]),
            'last_next_action': np.array(next_actions[-1]),
            'language': language,
            'dataset': dataset_name,  # Short name (bridge/fractal)
            'traj_length': int(images.shape[0]),
            'trajectory_id': valid_traj_count,  # Local ID per dataset
            'action': np.array(actions)
        }
        
        if keep_raw_instruction:
            trajectory_data["language_bytes"] = language_raw
        # Optionally keep full trajectory
        if keep_full_trajectory:
            trajectory_data.update({
                'images': images,
                'next_actions': np.array(next_actions)
            })
        trajectories.append(trajectory_data)

        valid_traj_count += 1
    
    print(f"  Final: {valid_traj_count} valid trajectories from {total_processed} total processed")   
    return trajectories

def extract_trajectory_data(dataset_name: str, max_trajectories: int = None, 
                           keep_full_trajectory=False, keep_raw_instruction=False,
                           save_trajectories: bool = False, save_path: str = None, 
                           save_format: str = "pkl", mix_ratio: dict = None,
                           seed: int = 44, skip_missing_instruction: bool = True):
    """
    Extract trajectory data from single or mixed OXE datasets with deterministic ordering.
        
    Uses Octo tools to create validation set of trajectories of Bridge or Fractal dataset.

    Args:
        :param dataset_name: "bridge", "fractal", or "bridge_fractal" for mixed
        :param max_trajectories: Max trajectories to extract. Set None for all; Bridge Validation set is ~7k
        keep_full_trajectory: If True, also keep full trajectory (memory intensive)
        save_trajectories: If True, save extracted data to disk for diffusion pipeline
        save_path: Path to save file (auto-generated if None)
        save_format: "pkl" 
        :param mix_ratio: Dict like {"bridge": 0.6, "fractal": 0.4} for custom mixing
        :param seed: Random seed for deterministic ordering
    """
    tf.random.set_seed(seed) # make sure TensorFlow is deterministic
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    # FLAGS = create_config()
    # oxe_config = create_oxe_config()
    # TODO: Load configs e.g. from files instead of creating them here
    FLAGS, oxe_config = load_configs_for_jupyter(
        algorithm="ensemble_sarsa",
        data_dir="/V-GPS/datasets/open_x",
        batch_size=1024,
        ensemble_size=8,
        seed=seed,
        resume_path=ENSEMBLE_MODEL_PATH
    )
    
    # Create dataset kwargs
    (dataset_kwargs_list, sample_weights) = make_oxe_dataset_kwargs_and_weights(**oxe_config["oxe_kwargs"])
        
    # Filter for specific dataset
    if dataset_name == "bridge_fractal":
        datasets_to_process = ["bridge", "fractal"]
        if mix_ratio is None:
            mix_ratio = {"bridge": 0.5, "fractal": 0.5}  # Default 50/50
    elif dataset_name in ["bridge", "fractal"]:
        datasets_to_process = [dataset_name]
        mix_ratio = {dataset_name: 1.0}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    all_trajectories = []

    for dataset_short_name in datasets_to_process:
        print(f"Processing {dataset_short_name} dataset...")
        
        # Map short name to full dataset name for filtering
        filter_name = _get_full_dataset_name(dataset_short_name)
        val_datasets_kwargs_list, _ = filter_eval_datasets(dataset_kwargs_list, sample_weights, [filter_name])
        
        if not val_datasets_kwargs_list:
            print(f"Warning: No data found for {dataset_short_name}")
            continue
            
        # Calculate how many trajectories to extract from this dataset
        expected_count = int(max_trajectories * mix_ratio.get(dataset_short_name, 0)) if max_trajectories else None
        print(f"Target trajectories for {dataset_short_name}: {expected_count}")
        
        val_data = create_validation_dataset(
            val_datasets_kwargs_list[0],
            oxe_config["traj_transform_kwargs"],
            oxe_config["frame_transform_kwargs"],
            train=False
        )
        
        # Extract trajectories from this dataset
        dataset_trajectories = _extract_from_single_dataset(
            val_data, dataset_short_name, expected_count, 
            keep_full_trajectory, keep_raw_instruction, skip_missing_instruction=skip_missing_instruction
        )
        
        all_trajectories.extend(dataset_trajectories)
        print(f"Extracted {len(dataset_trajectories)} trajectories from {dataset_short_name}")
    
    print(f"Total extracted: {len(all_trajectories)} trajectories from {dataset_name}")
        
    # Save for PyTorch diffusion pipeline if requested
    if save_trajectories:
        # Auto-generate save path if not provided
        if save_path is None:
            num_tra_str = f"_n{max_trajectories}" if max_trajectories else "_all"
            save_path = f"data/{dataset_name}_{num_tra_str}.pkl"
        elif not save_path.endswith('.pkl'):
            save_path += '.pkl'
        
        # Ensure directory exists
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving {len(all_trajectories)} trajectories to {save_path}...")
        
        # Prepare data with metadata
        save_data = {
            'trajectories': all_trajectories,
            'metadata': {
                'dataset_name': dataset_name,
                'num_trajectories': len(all_trajectories),
                'dataset_distribution': mix_ratio,
                'image_shape': all_trajectories[0]['first_image'].shape if all_trajectories else None,
                'action_shape': all_trajectories[0]['first_action'].shape if all_trajectories else None,
                'format_version': '1.1',
                'created_for': 'pytorch_diffusion_pipeline'
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Saved to: {save_path}")

    return all_trajectories

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
# or
traj = extract_trajectory_data("bridge_fractal", max_trajectories=300, save_trajectories=True)
"""

def _save_image_with_timestamp(img_array, filepath, timestamp):
    """Save image with specific timestamp in EXIF data."""
    # Convert numpy array to PIL Image
    img = Image.fromarray(img_array)
    
    try:
        import piexif
        
        # Create EXIF data with correct timestamp tags
        exif_dict = {
            "0th": {
                piexif.ImageIFD.DateTime: timestamp.strftime("%Y:%m:%d %H:%M:%S"),
            },
            "Exif": {
                piexif.ExifIFD.DateTimeOriginal: timestamp.strftime("%Y:%m:%d %H:%M:%S"),
                piexif.ExifIFD.DateTimeDigitized: timestamp.strftime("%Y:%m:%d %H:%M:%S"),
            },
            "GPS": {},
            "1st": {},
            "thumbnail": None
        }
        
        # Convert to piexif format and save
        exif_bytes = piexif.dump(exif_dict)
        img.save(filepath, exif=exif_bytes)
        
    except ImportError:
        print("Warning: piexif not installed, saving without EXIF timestamp")
        print("Install with: pip install piexif")
        img.save(filepath)
        
        # Fallback: set file modification time
        import os
        timestamp_epoch = timestamp.timestamp()
        os.utime(filepath, (timestamp_epoch, timestamp_epoch))
        
    except Exception as e:
        print(f"Warning: Could not set EXIF timestamp: {e}")
        img.save(filepath)
        
        # Fallback: set file modification time
        import os
        timestamp_epoch = timestamp.timestamp()
        os.utime(filepath, (timestamp_epoch, timestamp_epoch))

def extract_trajectory_images(trajectories, output_dir="data/trajectory_images", base_timestamp=None):
    """
    Extract first and last images from trajectories and save as PNGs with ordered timestamps.
    
    :param trajectories: List of trajectory dictionaries
    :param output_dir: Directory to save images
    :param base_timestamp: Starting timestamp (datetime), defaults to now
    :returns: Number of images saved
    """
    from PIL import Image
    from PIL.ExifTags import IFD, TAGS
    from datetime import datetime, timedelta

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if base_timestamp is None:
        base_timestamp = datetime(2025, 1, 1, 12, 0, 0)  # Fixed base date for consistency
    
    image_count = 0
    current_time = base_timestamp
    
    # Sort trajectories: bridge first, then fractal, by trajectory_id within each dataset
    sorted_trajs = sorted(trajectories, key=lambda x: (x['dataset'], x['trajectory_id']))
    
    for traj in sorted_trajs:
        dataset = traj['dataset']
        traj_id = traj['trajectory_id']
        
        # Save first image with timestamp
        first_img = traj['first_image']
        if first_img.dtype != np.uint8:
            first_img = (first_img * 255).astype(np.uint8)
        
        first_filename = f"first_image_{dataset}_{traj_id}.png"
        first_path = output_path / first_filename
        _save_image_with_timestamp(first_img, first_path, current_time)
        current_time += timedelta(minutes=1)  # Increment by 1 minute
        
        # Save last image with timestamp
        last_img = traj['last_image']
        if last_img.dtype != np.uint8:
            last_img = (last_img * 255).astype(np.uint8)
            
        last_filename = f"last_image_{dataset}_{traj_id}.png"
        last_path = output_path / last_filename
        _save_image_with_timestamp(last_img, last_path, current_time)
        current_time += timedelta(minutes=1)  # Increment by 1 minute
        
        image_count += 2
    
    print(f"Saved {image_count} images with ordered timestamps to {output_path}")
    print(f"Order: first_image_0, last_image_0, first_image_1, last_image_1, ...")
    print(f"Datasets ordered: bridge first, then fractal")


def load_edited_images_back(trajectories, images_dir="trajectory_images"):
    """
    Load edited images back into trajectory data structure.
    
    :param trajectories: Original trajectory list
    :param images_dir: Directory containing edited images
    :returns: Updated trajectory list with edited images
    """
    images_path = Path(images_dir)
    
    for traj in trajectories:
        dataset = traj['dataset'] 
        traj_id = traj['trajectory_id']
        
        # Load edited first image
        first_filename = f"first_image_{dataset}_{traj_id}.png"
        first_path = images_path / first_filename
        if first_path.exists():
            edited_first = np.array(Image.open(first_path))
            traj['first_image_inpainted'] = edited_first
        
        # Load edited last image
        last_filename = f"last_image_{dataset}_{traj_id}.png"
        last_path = images_path / last_filename
        if last_path.exists():
            edited_last = np.array(Image.open(last_path))
            traj['last_image_inpainted'] = edited_last
    
    print(f"Loaded edited images from {images_path}")
    return trajectories


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
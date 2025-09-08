import torch
import numpy as np
from PIL import Image
import pickle

def create_inpaintings_from_mask_flux(data: dict, diffusion_pipeline, out_h: int= 1024, out_w: int = 1024, max_seq_len: int =512, save_path: str = None):
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
        
        i1_inpainted = diffusion_pipeline(
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

        i2_inpainted = diffusion_pipeline(
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
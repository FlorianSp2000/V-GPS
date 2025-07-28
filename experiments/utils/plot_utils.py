"""Plotting Utilities for OOD evaluation"""

from collections import defaultdict
import matplotlib.pyplot as plt
import random

def plot_inpainting_results(data: dict, samples_per_category: int = 5, figsize: tuple = (24, 4), save_path: str = None):
    """
    Takes dictionary of robot trajectories with 'first_image', 'last_image' and corresponding inpainting results + OOD Category.
    This plot function visualizes the inpainting results for each trajectory.
    It shows the first and last images, their inpainted versions, the original language instruction,
    and the inpainting prompt, all organized by OOD category.
    
    :param data: data with inpainted results
    :param samples_per_category: number of random samples to show per category
    :param figsize: figure size per row
    :param save_path: path to save plots (without extension), if None shows plots
    """
    # Group trajectories by ood_category
    categories = defaultdict(list)
    for traj in data['trajectories']:
        if 'inpainted_first_image' in traj and 'inpainted_last_image' in traj:
            categories[traj['ood_category']].append(traj)
    
    # Sample random trajectories per category
    sampled_by_category = {}
    for category, trajs in categories.items():
        if len(trajs) >= samples_per_category:
            sampled = random.sample(trajs, samples_per_category)
        else:
            sampled = trajs  # Take all if less than requested
        sampled_by_category[category] = sampled
    
    # Create separate plot for each category
    for category, trajs in sampled_by_category.items():
        n_samples = len(trajs)
        
        print(f"\n{'='*60}")
        print(f"Category: {category.upper()} ({len(categories[category])} total samples)")
        print(f"{'='*60}")
        
        # Create plot with better proportions: 4 image columns get more space than 2 text columns
        fig, axes = plt.subplots(n_samples, 6, figsize=(figsize[0], figsize[1] * n_samples))
        
        # Adjust column width ratios: give more space to images
        gs = fig.add_gridspec(n_samples, 6, width_ratios=[3, 3, 3, 3, 2.5, 2.5])
        fig.clear()
        
        if n_samples == 1:
            axes = [[fig.add_subplot(gs[0, j]) for j in range(6)]]
        else:
            axes = [[fig.add_subplot(gs[i, j]) for j in range(6)] for i in range(n_samples)]
        
        for i, traj in enumerate(trajs):
            # Column 1: First image
            axes[i][0].imshow(traj['first_image'])
            if i == 0:
                axes[i][0].set_title('First Frame', fontsize=12, fontweight='bold')
            axes[i][0].axis('off')
            
            # Column 2: First inpainted
            axes[i][1].imshow(traj['inpainted_first_image'])
            if i == 0:
                axes[i][1].set_title('First Augmented', fontsize=12, fontweight='bold')
            axes[i][1].axis('off')
            
            # Column 3: Last image
            axes[i][2].imshow(traj['last_image'])
            if i == 0:
                axes[i][2].set_title('Last Frame', fontsize=12, fontweight='bold')
            axes[i][2].axis('off')
            
            # Column 4: Last inpainted
            axes[i][3].imshow(traj['inpainted_last_image'])
            if i == 0:
                axes[i][3].set_title('Last Augmented', fontsize=12, fontweight='bold')
            axes[i][3].axis('off')
            
            # Column 5: Original language instruction - more compact
            # Wrap text manually for better formatting
            instruction = traj['language']
            wrapped_instruction = '\n'.join([instruction[j:j+40] for j in range(0, len(instruction), 40)])
            
            axes[i][4].text(0.5, 0.5, wrapped_instruction, 
                           ha='center', va='center', fontsize=14, 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))
    
            if i == 0:
                axes[i][4].set_title('Original Instruction', fontsize=12, fontweight='bold')
            axes[i][4].axis('off')
            
            # Column 6: Inpainting prompt - more compact
            prompt = traj['inpainting_prompt']
            wrapped_prompt = '\n'.join([prompt[j:j+40] for j in range(0, len(prompt), 40)])
            
            axes[i][5].text(0.5, 0.5, wrapped_prompt, 
                           ha='center', va='center', fontsize=14,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.8))
            if i == 0:
                axes[i][5].set_title('Inpaint Prompt', fontsize=12, fontweight='bold')
            axes[i][5].axis('off')
        
        plt.tight_layout()
    
        if save_path:
           filename = f"{save_path}_{category}.pdf"
           plt.savefig(filename, dpi=150, bbox_inches='tight')
           print(f"Saved: {filename}")

        plt.show()
    
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"{'='*60}")
    for category, trajs in categories.items():
        shown = min(len(trajs), samples_per_category)
        print(f"  {category}: showing {shown}/{len(trajs)} trajectories")

import dlimp as dl
import tensorflow as tf

def create_negative_demonstrations(dataset: dl.DLataset, 
    negative_ratio: float = 0.05, 
    seed: int = 42, 
    prompt_pool_size: int = 10_000,
    use_bootstrap=True,
    ):
    """
    Takes a dlimp dataset of non-flattened, consistent trajectories.
    Add negative demonstrations by modifying language prompts and rewards for a percentage of trajectories.
    This adds a new key 'original_language_instruction' to each trajectory for reference.

    :param dataset: DLataset containing trajectories
    :param negative_ratio: Fraction of trajectories to convert to negative demos (e.g., 0.05 for 5%)
    :param seed: Random seed for reproducible sampling
    :param use_bootstrap: If True, set td_mask to all ones for negative demos
    :returns: Dataset with negative demonstrations added
    """
    
    # Step 1: Collect all unique language prompts for random sampling
    print("Collecting language prompts...")
    language_prompts = []

    for example in dataset.take(prompt_pool_size):
        lang_instruction = example['task']['language_instruction'].numpy()[0]
        # assume we have filtered out all empty language instructions for building the prompt pool
        language_prompts.append(lang_instruction)
    
    print(f"len(language_prompts) is {len(language_prompts)}")
    # Remove duplicates and convert to tensor
    unique_prompts = list(set(language_prompts))
    language_prompts_tensor = tf.constant(unique_prompts)
    print(f"Collected {len(unique_prompts)} unique language prompts")
    
    # Set up random selection for negative demos
    tf.random.set_seed(seed) # TODO: Probably better to not set global seed here

    def modify_trajectory(traj):
        """Randomly decide whether to convert trajectory to negative demo."""
        
        # Random decision based on negative_ratio
        should_modify = tf.random.uniform(()) < negative_ratio
        
        def create_negative_demo():
            """Convert to negative demo: random prompt + all rewards = -1"""
            # Random prompt selection
            random_idx = tf.random.uniform([], 0, len(unique_prompts), dtype=tf.int32)
            random_prompt = language_prompts_tensor[random_idx]
            
            # Create modified traj
            modified_example = dict(traj)
            modified_example['task'] = dict(traj['task'])
            modified_example['task']['original_language_instruction'] = traj['task']['language_instruction']

            # Replace language instruction with random prompt
            traj_length = tf.shape(traj['task']['language_instruction'])[0]
            modified_example['task']['language_instruction'] = tf.fill([traj_length], random_prompt)
            
            # Set ALL rewards to -1 (negative demonstration)
            modified_example['reward'] = tf.ones_like(traj['reward']) * -1
            if use_bootstrap:
                modified_example['td_mask'] = tf.ones_like(traj['reward']) # instead of 1,1,...,0,0,0
            # TODO: This lets 'mc_return' that is computed in make_dataset_from_rlds become invalid 
            return modified_example
        
        def keep_original():
            """Keep original trajectory unchanged. Just add original_language_instruction key for consistency """
            unchanged_traj = dict(traj)
            unchanged_traj['task'] = dict(traj['task'])
            unchanged_traj['task']['original_language_instruction'] = traj['task']['language_instruction']
            return unchanged_traj
        
        # Conditionally apply modification
        return tf.cond(should_modify, create_negative_demo, keep_original)
    
    # Step 3: Apply the transformation
    print(f"Creating negative demonstrations ({negative_ratio*100:.1f}% of trajectories)...")
    modified_dataset = dataset.traj_map(modify_trajectory)
    
    return modified_dataset, language_prompts_tensor
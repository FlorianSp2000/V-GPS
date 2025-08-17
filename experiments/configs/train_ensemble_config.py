from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_config(config_string):
    """
    Corrected ensemble configuration with proper agent naming.
    """
    base_real_config = dict(
        batch_size=256,  # This will be divided by ensemble_size
        num_steps=int(1e6),
        log_interval=1000,
        eval_interval=20000,
        save_interval=50000,
        save_dir=placeholder(str),
        resume_path="",
        resume_wandb_id=placeholder(str),
        seed=42,
        ensemble_size=8,  # Number of ensembles
        create_negative_demos=False, # Include Trajectories with non-matching language prompts
        negative_demo_ratio=0.05,
        exclude_empty_lang_instr=False,  # Exclude empty language instructions from training
    )

    possible_structures = {
        "ensemble_sarsa": ConfigDict(
            dict(
                agent="sarsa_ensemble",  # Corrected: points to SARSAEnsembleAgent
                agent_kwargs=dict(
                    # Goal conditioning
                    language_conditioned=True,
                    goal_conditioned=True,
                    early_goal_concat=None,
                    shared_goal_encoder=None,
                    shared_encoder=False,
                    
                    # Learning parameters
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    discount=0.98,
                    soft_target_update_rate=5e-3,
                    
                    # Critic configuration
                    critic_ensemble_size=2,
                    critic_subsample_size=None,
                    use_min_q=True,
                    
                    # Network architecture
                    network_kwargs=dict(
                        hidden_dims=[256, 256],
                        activate_final=True,
                        use_layer_norm=False,
                    ),
                    
                    # OOD evaluation
                    num_ood_actions=10,
                    ood_action_sample_method="uniform",
                ),
                
                # Text processing
                text_processor="muse_embedding",
                text_processor_kwargs=dict(),
                
                # Encoder
                encoder="resnetv1-34-bridge-film",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                
                # Base config
                **base_real_config,
            )
        ),

        "ensemble_lc_cql": ConfigDict(
            dict(
                agent="cql",  # Keep as standard CQL for now
                agent_kwargs=dict(
                    language_conditioned=True,
                    goal_conditioned=True,
                    early_goal_concat=None,
                    shared_goal_encoder=None,
                    shared_encoder=False,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    discount=0.98,
                    cql_alpha=5.0,
                    target_update_rate=5e-3,
                    gc_kwargs=dict(negative_proportion=0.0),
                    use_calql=False,
                    critic_network_kwargs=dict(
                        hidden_dims=[256, 256],
                        activate_final=True,
                        use_layer_norm=False,
                    ),
                    policy_network_kwargs=dict(
                        hidden_dims=[256, 256],
                        activate_final=True,
                        use_layer_norm=False,
                    ),
                    policy_kwargs=dict(
                        tanh_squash_distribution=True,
                        std_parameterization="exp",
                    ),
                    actor_optimizer_kwargs=dict(
                        learning_rate=1e-4,
                        warmup_steps=2000,
                    ),
                    critic_optimizer_kwargs=dict(
                        learning_rate=3e-4,
                        warmup_steps=2000,
                    ),
                ),
                text_processor="muse_embedding",
                text_processor_kwargs=dict(),
                encoder="resnetv1-34-bridge-film",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                **base_real_config,
            )
        ),
    }

    return possible_structures[config_string]
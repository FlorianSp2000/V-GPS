from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_config(config_string):
    base_real_config = dict(
        batch_size=256,
        num_steps=int(1e6),
        log_interval=1000,
        eval_interval=20000,
        save_interval=100000,
        save_dir=placeholder(str),
        resume_path="",
        seed=42,
    )

    possible_structures = {
         "lc_cql": ConfigDict(
            dict(
                agent="cql",
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
                    gc_kwargs=dict(
                        negative_proportion=0.0,
                    ),
                    use_calql=False,
                    critic_network_kwargs = dict(
                        hidden_dims = [256, 256],
                        activate_final = True,
                        use_layer_norm = False,
                    ),
                    policy_network_kwargs = dict(
                        hidden_dims = [256, 256],
                        activate_final = True,
                        use_layer_norm = False,
                    ),
                    policy_kwargs = dict(
                        tanh_squash_distribution=True,
                        std_parameterization="exp",
                    ),
                    actor_optimizer_kwargs = dict(
                        learning_rate=1e-4,
                        warmup_steps=2000,
                    ),
                    critic_optimizer_kwargs = dict(
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

        "lc_iql": ConfigDict(
            dict(
                agent="gc_iql",
                agent_kwargs=dict(
                    language_conditioned=True,
                    early_goal_concat=None,
                    shared_goal_encoder=None,
                    shared_encoder=False,
                    use_proprio=False,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    discount=0.98,
                    expectile=0.7,
                    temperature=1.0,
                    target_update_rate=0.002,
                    negative_proportion=0.1,
                    policy_kwargs=dict(
                        tanh_squash_distribution=False,
                        std_parameterization="exp",
                    ),
                    network_kwargs=dict(
                        hidden_dims=(256, 256),
                        dropout_rate=0.0,
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

         "lc_ddpm_bc": ConfigDict(
            dict(
                agent="gc_ddpm_bc",
                agent_kwargs=dict(
                    score_network_kwargs=dict(
                        time_dim=32,
                        num_blocks=3,
                        dropout_rate=0.1,
                        hidden_dim=256,
                        use_layer_norm=True,
                    ),
                    language_conditioned=True,
                    early_goal_concat=None,
                    shared_goal_encoder=None,
                    use_proprio=False,
                    beta_schedule="cosine",
                    diffusion_steps=20,
                    action_samples=1,
                    repeat_last_step=0,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    actor_decay_steps=int(2e6),
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
        "sarsa": ConfigDict(
            dict(
                agent="sarsa",
                agent_kwargs=dict(
                    # Goal conditioning (same as CQL pattern)
                    language_conditioned=True,
                    goal_conditioned=True,
                    early_goal_concat=None,
                    shared_goal_encoder=None,
                    shared_encoder=False,
                    
                    # Learning parameters
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    discount=0.98,
                    soft_target_update_rate=5e-3,  # TODO: was target_update_rate but sac.py also uses soft_target_update_rate
                    
                    # Critic configuration
                    critic_ensemble_size=2,
                    critic_subsample_size=None,
                    use_min_q=True,  # Use min over ensemble to reduce overestimation
                    
                    network_kwargs=dict(
                        hidden_dims=[256, 256],
                        activate_final=True,
                        use_layer_norm=False,
                    ),
                ),
                
                # Text processing (same as CQL)
                text_processor="muse_embedding",
                text_processor_kwargs=dict(),
                
                # Encoder (same as CQL)
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
    }

    return possible_structures[config_string]

from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_config(config_string):
    """
    Configuration for ensemble vs single agent testing.
    """
    base_config = dict(
        # Evaluation parameters
        ensemble_size=8,
        checkpoint_step=None,  # Use None for latest checkpoint
        batch_size=256,
        num_eval_batches=16,
        num_trajectory_plots=4,
        
        # No training parameters needed, just evaluation
        seed=42,
        # Agent configuration - must match the training config
        agent="sarsa",  # This will be used for single agent
        agent_kwargs=dict(
            # Goal conditioning
            language_conditioned=True,
            goal_conditioned=True,
            early_goal_concat=None,
            shared_goal_encoder=None,
            shared_encoder=False,
            
            # Learning parameters (not used in testing but needed for agent creation)
            learning_rate=3e-4,
            warmup_steps=2000,
            discount=0.98,
            soft_target_update_rate=5e-3,
            
            # Critic configuration
            critic_ensemble_size=2,
            critic_subsample_size=None,
            use_min_q=True,
            
            # Network architecture - must match training
            network_kwargs=dict(
                hidden_dims=[256, 256],
                activate_final=True,
                use_layer_norm=False,
            ),
            
            # OOD evaluation
            num_ood_actions=10,
            ood_action_sample_method="uniform",
        ),
        
        # Text processing - must match training
        text_processor="muse_embedding",
        text_processor_kwargs=dict(),
        
        # Encoder - must match training
        encoder="resnetv1-34-bridge-film",
        encoder_kwargs=dict(
            pooling_method="avg",
            add_spatial_coordinates=True,
            act="swish",
        ),
    )

    possible_structures = {
        "test_ensemble_sarsa": ConfigDict(base_config),        
        # Add other configurations if needed
        # "test_ensemble_cql": ConfigDict(
        #     dict(
        #         agent="cql",
        #         agent_kwargs=dict(
        #             language_conditioned=True,
        #             goal_conditioned=True,
        #             early_goal_concat=None,
        #             shared_goal_encoder=None,
        #             shared_encoder=False,
        #             learning_rate=3e-4,
        #             warmup_steps=2000,
        #             discount=0.98,
        #             cql_alpha=5.0,
        #             target_update_rate=5e-3,
        #             gc_kwargs=dict(negative_proportion=0.0),
        #             use_calql=False,
        #             critic_network_kwargs=dict(
        #                 hidden_dims=[256, 256],
        #                 activate_final=True,
        #                 use_layer_norm=False,
        #             ),
        #             policy_network_kwargs=dict(
        #                 hidden_dims=[256, 256],
        #                 activate_final=True,
        #                 use_layer_norm=False,
        #             ),
        #             policy_kwargs=dict(
        #                 tanh_squash_distribution=True,
        #                 std_parameterization="exp",
        #             ),
        #             actor_optimizer_kwargs=dict(
        #                 learning_rate=1e-4,
        #                 warmup_steps=2000,
        #             ),
        #             critic_optimizer_kwargs=dict(
        #                 learning_rate=3e-4,
        #                 warmup_steps=2000,
        #             ),
        #         ),
        #     **base_config,
        #     )
        # ),
    }

    return possible_structures[config_string]
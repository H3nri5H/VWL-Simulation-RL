"""Training script for VWL-Simulation-RL."""

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from env.economy_env import EconomyEnv


def main():
    """Main training function."""
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        # Configure PPO for Multi-Agent Training
        config = (
            PPOConfig()
            .environment(env=EconomyEnv, env_config={'n_firms': 2, 'n_households': 10})
            .framework('torch')
            .rollouts(num_rollout_workers=1)
            .training(
                train_batch_size=4000,
                sgd_minibatch_size=128,
                num_sgd_iter=10,
                lr=3e-4,
                gamma=0.99,
                lambda_=0.95,
                clip_param=0.2,
            )
            .multi_agent(
                policies={
                    'firm_policy': (
                        None,  # Use default policy class
                        EconomyEnv().observation_spaces['firm_0'],
                        EconomyEnv().action_spaces['firm_0'],
                        {}
                    )
                },
                policy_mapping_fn=lambda agent_id, *args, **kwargs: 'firm_policy',
            )
            .resources(num_gpus=0)
        )
        
        # Build algorithm
        algo = config.build()
        
        print("Starting training...")
        print(f"Environment: 2 Firms, 10 Households")
        print("-" * 50)
        
        # Training loop
        for i in range(10):  # 10 iterations for testing
            result = algo.train()
            print(f"Iteration {i+1}: reward_mean={result['episode_reward_mean']:.2f}")
        
        print("-" * 50)
        print("Training completed!")
        
        # Save checkpoint
        checkpoint_path = algo.save()
        print(f"Checkpoint saved to: {checkpoint_path}")
        
        algo.stop()
        
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()

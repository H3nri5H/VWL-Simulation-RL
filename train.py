"""Training script for VWL-Simulation Multi-Agent RL.

Trains 2 firms using PPO with shared policy (Parameter Sharing).
Households use rule-based behavior (not trained).
"""

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray import tune
from env.economy_env import EconomyEnv
import os
import argparse
from datetime import datetime


def env_creator(config):
    """Create wrapped PettingZoo environment for RLlib.
    
    Args:
        config: Environment configuration dict
        
    Returns:
        PettingZooEnv: Wrapped environment compatible with RLlib
    """
    return PettingZooEnv(EconomyEnv(config))


def get_training_config(args):
    """Create PPO training configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        PPOConfig: Configured PPO algorithm
    """
    # Register environment
    tune.register_env("economy_env", env_creator)
    
    # Create dummy env to get observation/action spaces
    dummy_env = env_creator({
        'n_firms': args.n_firms,
        'n_households': args.n_households,
    })
    
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space
    
    config = (
        PPOConfig()
        .api_stack(
            # Disable new API stack to use old stable API
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="economy_env",  # Use registered env name
            env_config={
                'n_firms': args.n_firms,
                'n_households': args.n_households,
            }
        )
        .framework('torch')  # PyTorch (better Ray 2.53+ support)
        .env_runners(
            num_env_runners=args.num_workers,
            rollout_fragment_length=200,
        )
        .training(
            train_batch_size=4000,
            minibatch_size=128,
            num_epochs=10,
            lr=args.learning_rate,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,  # Encourage exploration
        )
        .multi_agent(
            policies={
                'firm_policy': (
                    None,  # Use default PPO policy
                    obs_space,
                    act_space,
                    {}
                )
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: 'firm_policy',
        )
        .resources(num_gpus=0)  # CPU only (Windows compatibility)
        .debugging(log_level='INFO')
    )
    
    return config


def train(args):
    """Main training loop.
    
    Args:
        args: Command line arguments
    """
    # Initialize Ray
    ray.init(
        ignore_reinit_error=True,
        num_cpus=args.num_workers + 1,  # Workers + trainer
    )
    
    try:
        print("="*60)
        print("VWL-Simulation Multi-Agent RL Training")
        print("="*60)
        print(f"Environment Config:")
        print(f"  - Firms: {args.n_firms}")
        print(f"  - Households: {args.n_households}")
        print(f"\nTraining Config:")
        print(f"  - Algorithm: PPO (Proximal Policy Optimization)")
        print(f"  - Framework: PyTorch")
        print(f"  - Learning Rate: {args.learning_rate}")
        print(f"  - Workers: {args.num_workers}")
        print(f"  - Total Iterations: {args.iterations}")
        print(f"  - Checkpoint Every: {args.checkpoint_freq} iterations")
        print("="*60)
        print()
        
        # Build config
        config = get_training_config(args)
        
        # Build algorithm
        algo = config.build()
        
        # Training loop
        best_reward = float('-inf')
        
        for iteration in range(1, args.iterations + 1):
            print(f"\n[Iteration {iteration}/{args.iterations}]")
            
            # Train one iteration
            result = algo.train()
            
            # Extract metrics
            reward_mean = result.get('episode_reward_mean', 0)
            reward_min = result.get('episode_reward_min', 0)
            reward_max = result.get('episode_reward_max', 0)
            episode_len = result.get('episode_len_mean', 0)
            
            # Print progress
            print(f"  Reward: {reward_mean:.2f} (min: {reward_min:.2f}, max: {reward_max:.2f})")
            print(f"  Episode Length: {episode_len:.1f}")
            
            # Policy-specific metrics (if available)
            if 'policy_reward_mean' in result:
                for policy_id, policy_reward in result['policy_reward_mean'].items():
                    print(f"  {policy_id} reward: {policy_reward:.2f}")
            
            # Save checkpoint periodically
            if iteration % args.checkpoint_freq == 0:
                checkpoint_path = algo.save()
                print(f"  ✅ Checkpoint saved: {checkpoint_path}")
                
                # Track best model
                if reward_mean > best_reward:
                    best_reward = reward_mean
                    print(f"  🏆 New best reward: {best_reward:.2f}")
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        
        # Final checkpoint
        final_checkpoint = algo.save()
        print(f"Final checkpoint saved: {final_checkpoint}")
        print(f"Best reward achieved: {best_reward:.2f}")
        
        # Copy to models/ directory for easy access
        try:
            import shutil
            models_dir = "models"
            os.makedirs(models_dir, exist_ok=True)
            
            # Extract checkpoint folder name
            checkpoint_name = os.path.basename(final_checkpoint)
            dest_path = os.path.join(models_dir, checkpoint_name)
            
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            shutil.copytree(final_checkpoint, dest_path)
            
            print(f"\n✅ Checkpoint copied to: {dest_path}")
            print(f"   You can now use this in the dashboard!")
        except Exception as e:
            print(f"\n⚠️  Could not copy to models/: {e}")
            print(f"   Manual copy: cp -r {final_checkpoint} models/")
        
        algo.stop()
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        ray.shutdown()


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(
        description='Train Multi-Agent RL for VWL-Simulation'
    )
    
    # Environment parameters
    parser.add_argument(
        '--n-firms',
        type=int,
        default=2,
        help='Number of firms (default: 2)'
    )
    parser.add_argument(
        '--n-households',
        type=int,
        default=10,
        help='Number of households (default: 10)'
    )
    
    # Training parameters
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of training iterations (default: 100)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='Learning rate (default: 0.0003)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=2,
        help='Number of rollout workers (default: 2)'
    )
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=10,
        help='Save checkpoint every N iterations (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Start training
    train(args)


if __name__ == '__main__':
    main()

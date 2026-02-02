import os
import json
import yaml
import warnings
import argparse
import shutil
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig
from env.economy_env import SimpleEconomyEnv

warnings.filterwarnings('ignore', category=DeprecationWarning)


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def train(
    # Environment parameters
    n_firms=None,
    n_households=None,
    max_steps=None,
    # Training parameters
    iterations=None,
    checkpoint_freq=None,
    # PPO parameters
    learning_rate=None,
    gamma=None,
    lambda_=None,
    clip_param=None,
    # Resources
    num_env_runners=None,
    num_gpus=None,
):
    # Load config
    config = load_config()
    
    # Environment config (use args or config.yaml or defaults)
    env_cfg = config.get('environment', {})
    env_config = {
        'n_firms': n_firms or env_cfg.get('n_firms', 2),
        'n_households': n_households or env_cfg.get('n_households', 10),
        'max_steps': max_steps or env_cfg.get('max_steps', 100),
    }
    
    # Training config
    train_cfg = config.get('training', {})
    iterations = iterations or train_cfg.get('iterations', 50)
    checkpoint_freq = checkpoint_freq or train_cfg.get('checkpoint_frequency', 10)
    
    # PPO config
    ppo_cfg = train_cfg.get('ppo', {})
    lr = learning_rate or ppo_cfg.get('learning_rate', 3e-4)
    gamma_val = gamma or ppo_cfg.get('gamma', 0.99)
    lambda_val = lambda_ or ppo_cfg.get('lambda', 0.95)
    clip_val = clip_param or ppo_cfg.get('clip_param', 0.2)
    train_batch = ppo_cfg.get('train_batch_size', 400)
    minibatch = ppo_cfg.get('minibatch_size', 128)
    epochs = ppo_cfg.get('num_epochs', 10)
    
    # Resources
    res_cfg = train_cfg.get('resources', {})
    n_workers = num_env_runners or res_cfg.get('num_env_runners', 2)
    n_gpus = num_gpus if num_gpus is not None else res_cfg.get('num_gpus', 0)
    
    # Setup directories
    checkpoint_dir = os.path.abspath("./checkpoints")
    metrics_dir = os.path.abspath("./metrics")
    
    print("\n" + "="*70)
    print("  VWL SIMULATION - REINFORCEMENT LEARNING TRAINING")
    print("="*70)
    
    # Clear old data
    print("\n[1/4] Preparing environment...")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print("  ✓ Cleared old checkpoints")
    if os.path.exists(metrics_dir):
        shutil.rmtree(metrics_dir)
        print("  ✓ Cleared old metrics")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Display configuration
    print("\n[2/4] Configuration:")
    print(f"  Environment:")
    print(f"    - Firms: {env_config['n_firms']}")
    print(f"    - Households: {env_config['n_households']}")
    print(f"    - Steps per episode: {env_config['max_steps']}")
    print(f"  Training:")
    print(f"    - Iterations: {iterations}")
    print(f"    - Checkpoint every: {checkpoint_freq} iterations")
    print(f"    - Learning rate: {lr}")
    print(f"    - Workers: {n_workers}")
    print(f"    - GPUs: {n_gpus}")
    
    # Build PPO config
    print("\n[3/4] Building PPO algorithm...")
    rllib_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env=SimpleEconomyEnv,
            env_config=env_config,
        )
        .framework("torch")
        .env_runners(
            num_env_runners=n_workers,
            rollout_fragment_length=200,
        )
        .training(
            train_batch_size=train_batch,
            minibatch_size=minibatch,
            num_epochs=epochs,
            lr=lr,
            gamma=gamma_val,
            lambda_=lambda_val,
            clip_param=clip_val,
        )
        .multi_agent(
            policies={
                "shared_policy": (
                    None,
                    SimpleEconomyEnv({}).observation_space,
                    SimpleEconomyEnv({}).action_space,
                    {},
                )
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .resources(
            num_gpus=n_gpus,
        )
    )
    
    algo = rllib_config.build()
    print("  ✓ Algorithm ready")
    
    # Training loop
    print("\n[4/4] Training...")
    print("\n" + "-"*70)
    print(f"{'Iter':<6} {'Reward':<12} {'Min':<10} {'Max':<10} {'Ep Len':<10}")
    print("-"*70)
    
    for i in range(iterations):
        result = algo.train()
        
        env_runners = result.get('env_runners', {})
        reward_mean = env_runners.get('episode_reward_mean', 0.0)
        reward_min = env_runners.get('episode_reward_min', 0.0)
        reward_max = env_runners.get('episode_reward_max', 0.0)
        episode_len = env_runners.get('episode_len_mean', 0.0)
        
        print(f"{i+1:<6} {reward_mean:<12.2f} {reward_min:<10.2f} {reward_max:<10.2f} {episode_len:<10.0f}")
        
        # Checkpoint?
        should_checkpoint = (i + 1) % checkpoint_freq == 0 or (i + 1) == iterations
        
        if should_checkpoint:
            # Save metrics
            iteration_dir = os.path.join(metrics_dir, f"iteration_{i+1}")
            os.makedirs(iteration_dir, exist_ok=True)
            
            result_file = os.path.join(iteration_dir, "result.json")
            with open(result_file, 'w') as f:
                json.dump({
                    'training_iteration': i + 1,
                    'env_runners': {
                        'episode_reward_mean': reward_mean,
                        'episode_reward_min': reward_min,
                        'episode_reward_max': reward_max,
                        'episode_len_mean': episode_len,
                    }
                }, f)
            
            # Create specific checkpoint folder
            checkpoint_name = f"checkpoint_{i+1:06d}"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Save checkpoint to specific directory
            checkpoint_result = algo.save()
            # Copy from temporary location to our folder
            checkpoint_result.checkpoint.to_directory(checkpoint_path)
            
            # Save metadata
            metadata_file = os.path.join(checkpoint_path, "metadata.json")
            is_final = (i + 1) == iterations
            
            metadata = {
                'iteration': i + 1,
                'reward_mean': reward_mean,
                'episode_len_mean': episode_len,
                'timestamp': result.get('timestamp', 0),
                'is_favorite': is_final,
                'checkpoint_path': checkpoint_path
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            if is_final:
                print(f"\n  ⭐ Checkpoint {i+1} saved: {checkpoint_path}")
            else:
                print(f"\n  💾 Checkpoint {i+1} saved: {checkpoint_path}")
    
    print("-"*70)
    print("\n✅ Training complete!\n")
    
    algo.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train VWL simulation with configurable parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults from config.yaml
  python train.py
  
  # Override specific parameters
  python train.py --n-firms 5 --n-households 20 --iterations 100
  
  # Quick test run
  python train.py --iterations 10 --checkpoint-freq 5
        """
    )
    
    # Environment parameters
    env_group = parser.add_argument_group('Environment')
    env_group.add_argument("--n-firms", type=int, help="Number of firms")
    env_group.add_argument("--n-households", type=int, help="Number of households")
    env_group.add_argument("--max-steps", type=int, help="Steps per episode")
    
    # Training parameters
    train_group = parser.add_argument_group('Training')
    train_group.add_argument("--iterations", type=int, help="Total training iterations")
    train_group.add_argument("--checkpoint-freq", type=int, help="Checkpoint frequency")
    
    # PPO parameters
    ppo_group = parser.add_argument_group('PPO Hyperparameters')
    ppo_group.add_argument("--lr", type=float, help="Learning rate")
    ppo_group.add_argument("--gamma", type=float, help="Discount factor")
    ppo_group.add_argument("--lambda", type=float, dest='lambda_', help="GAE lambda")
    ppo_group.add_argument("--clip-param", type=float, help="PPO clip parameter")
    
    # Resources
    res_group = parser.add_argument_group('Resources')
    res_group.add_argument("--num-workers", type=int, help="Number of parallel workers")
    res_group.add_argument("--num-gpus", type=int, help="Number of GPUs")
    
    args = parser.parse_args()
    
    train(
        n_firms=args.n_firms,
        n_households=args.n_households,
        max_steps=args.max_steps,
        iterations=args.iterations,
        checkpoint_freq=args.checkpoint_freq,
        learning_rate=args.lr,
        gamma=args.gamma,
        lambda_=args.lambda_,
        clip_param=args.clip_param,
        num_env_runners=args.num_workers,
        num_gpus=args.num_gpus,
    )

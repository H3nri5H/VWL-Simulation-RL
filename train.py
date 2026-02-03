import os
import json
import yaml
import warnings
import argparse
import shutil
import time
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from env.economy_env import SimpleEconomyEnv

warnings.filterwarnings('ignore', category=DeprecationWarning)


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            "config.yaml not found! Please create a config.yaml file with your training parameters."
        )
    with open(config_path, 'r', encoding='utf-8') as f:  # FIX: Added UTF-8 encoding
        return yaml.safe_load(f)


def find_latest_checkpoint():
    """Find the most recent checkpoint to resume from"""
    checkpoint_dir = Path("./checkpoints").absolute()
    if not checkpoint_dir.exists():
        return None, 0
    
    checkpoints = []
    for cp_dir in checkpoint_dir.iterdir():
        if cp_dir.is_dir() and cp_dir.name.startswith('checkpoint_'):
            metadata_file = cp_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    # Use absolute path!
                    checkpoints.append((os.path.abspath(str(cp_dir)), metadata.get('iteration', 0)))
    
    if not checkpoints:
        return None, 0
    
    # Return checkpoint with highest iteration
    checkpoints.sort(key=lambda x: x[1])
    latest_path, latest_iter = checkpoints[-1]
    return latest_path, latest_iter


def safe_rmtree(path, max_retries=3):
    """Safely remove directory tree with retry on Windows permission errors"""
    for attempt in range(max_retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            return True
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(0.5)  # Wait a bit
                continue
            else:
                # Last attempt failed, but don't crash - just warn
                print(f"  Warning: Could not delete {path} (files in use)")
                print(f"    Training will continue, but old files may remain.")
                return False
    return False


def train(resume=False):
    """Train the VWL simulation using config.yaml parameters
    
    Args:
        resume: If True, continues training from the latest checkpoint
    """
    # Load all configuration from config.yaml
    config = load_config()
    
    # Extract environment configuration
    env_cfg = config.get('environment', {})
    env_config = {
        'n_firms': env_cfg.get('n_firms', 2),
        'n_households': env_cfg.get('n_households', 10),
        'max_steps': env_cfg.get('max_steps', 100),
    }
    
    # Extract training configuration
    train_cfg = config.get('training', {})
    config_iterations = train_cfg.get('iterations', 50)
    checkpoint_freq = train_cfg.get('checkpoint_frequency', 10)
    
    # Extract PPO configuration
    ppo_cfg = train_cfg.get('ppo', {})
    lr = ppo_cfg.get('learning_rate', 3e-4)
    gamma_val = ppo_cfg.get('gamma', 0.99)
    lambda_val = ppo_cfg.get('lambda', 0.95)
    clip_val = ppo_cfg.get('clip_param', 0.2)
    train_batch = ppo_cfg.get('train_batch_size', 400)
    minibatch = ppo_cfg.get('minibatch_size', 128)
    epochs = ppo_cfg.get('num_epochs', 10)
    
    # Extract resources configuration
    res_cfg = train_cfg.get('resources', {})
    n_workers = res_cfg.get('num_env_runners', 2)
    n_gpus = res_cfg.get('num_gpus', 0)
    
    # Setup directories
    checkpoint_dir = os.path.abspath("./checkpoints")
    metrics_dir = os.path.abspath("./metrics")
    
    print("\n" + "="*70)
    print("  VWL SIMULATION - REINFORCEMENT LEARNING TRAINING")
    print("="*70)
    
    # Check for resume and calculate iterations
    start_iteration = 0
    resume_checkpoint = None
    total_iterations = config_iterations
    
    if resume:
        resume_checkpoint, start_iteration = find_latest_checkpoint()
        if resume_checkpoint:
            # Calculate new target: current_iteration + config_iterations
            total_iterations = start_iteration + config_iterations
            print(f"\n[RESUME MODE]")
            print(f"  * Found checkpoint at iteration {start_iteration}")
            print(f"  * Path: {resume_checkpoint}")
            print(f"  * Config specifies {config_iterations} iterations")
            print(f"  -> Training from iteration {start_iteration + 1} to {total_iterations}")
        else:
            print("\n[RESUME MODE]")
            print(f"  Warning: No checkpoint found, starting fresh training")
            print(f"  -> Training for {config_iterations} iterations")
            resume = False
    else:
        print(f"\n[FRESH TRAINING]")
        print(f"  -> Training for {config_iterations} iterations")
    
    # Clear old data only if NOT resuming
    if not resume:
        print("\n[1/4] Preparing environment...")
        if safe_rmtree(checkpoint_dir):
            print("  * Cleared old checkpoints")
        if safe_rmtree(metrics_dir):
            print("  * Cleared old metrics")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
    else:
        print("\n[1/4] Resuming training...")
        print("  * Keeping existing checkpoints and metrics")
    
    # Display configuration
    print("\n[2/4] Configuration (from config.yaml):")
    print(f"  Environment:")
    print(f"    - Firms: {env_config['n_firms']}")
    print(f"    - Households: {env_config['n_households']}")
    print(f"    - Steps per episode: {env_config['max_steps']}")
    print(f"  Training:")
    if resume and resume_checkpoint:
        print(f"    - Resuming from iteration: {start_iteration}")
        print(f"    - Config iterations: {config_iterations}")
        print(f"    - New target iteration: {total_iterations}")
        print(f"    - Iterations to train: {total_iterations - start_iteration}")
    else:
        print(f"    - Total iterations: {total_iterations}")
    print(f"    - Checkpoint every: {checkpoint_freq} iterations")
    print(f"  PPO:")
    print(f"    - Learning rate: {lr}")
    print(f"    - Gamma: {gamma_val}")
    print(f"    - Lambda: {lambda_val}")
    print(f"    - Clip param: {clip_val}")
    print(f"  Resources:")
    print(f"    - Workers: {n_workers}")
    print(f"    - GPUs: {n_gpus}")
    
    # Build or load PPO algorithm
    print("\n[3/4] Building PPO algorithm...")
    
    if resume and resume_checkpoint:
        # Load from checkpoint - ensure absolute path
        abs_checkpoint = os.path.abspath(resume_checkpoint)
        print(f"  Loading checkpoint: {abs_checkpoint}")
        algo = PPO.from_checkpoint(abs_checkpoint)
        print("  * Algorithm loaded from checkpoint")
    else:
        # Build fresh
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
        print("  * Algorithm ready")
    
    # Training loop
    print("\n[4/4] Training...")
    print("\n" + "-"*70)
    print(f"{'Iter':<6} {'Reward':<12} {'Min':<10} {'Max':<10} {'Ep Len':<10}")
    print("-"*70)
    
    for i in range(start_iteration, total_iterations):
        result = algo.train()
        
        env_runners = result.get('env_runners', {})
        reward_mean = env_runners.get('episode_reward_mean', 0.0)
        reward_min = env_runners.get('episode_reward_min', 0.0)
        reward_max = env_runners.get('episode_reward_max', 0.0)
        episode_len = env_runners.get('episode_len_mean', 0.0)
        
        print(f"{i+1:<6} {reward_mean:<12.2f} {reward_min:<10.2f} {reward_max:<10.2f} {episode_len:<10.0f}")
        
        # Checkpoint?
        should_checkpoint = (i + 1) % checkpoint_freq == 0 or (i + 1) == total_iterations
        
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
            
            # Save metadata WITH env_config!
            metadata_file = os.path.join(checkpoint_path, "metadata.json")
            is_final = (i + 1) == total_iterations
            
            metadata = {
                'iteration': i + 1,
                'reward_mean': reward_mean,
                'episode_len_mean': episode_len,
                'timestamp': result.get('timestamp', 0),
                'is_favorite': is_final,
                'checkpoint_path': checkpoint_path,
                # ADD ENV CONFIG HERE!
                'env_config': {
                    'n_firms': env_config['n_firms'],
                    'n_households': env_config['n_households'],
                    'max_steps': env_config['max_steps'],
                }
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            if is_final:
                print(f"\n  [*] Checkpoint {i+1} saved: {checkpoint_path}")
            else:
                print(f"\n  [+] Checkpoint {i+1} saved: {checkpoint_path}")
    
    print("-"*70)
    print("\n[SUCCESS] Training complete!\n")
    
    algo.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train VWL simulation (all parameters from config.yaml)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
All training parameters are now configured in config.yaml!

Examples:
  # Start fresh training with config.yaml parameters
  python train.py
  
  # Resume from last checkpoint and train for additional iterations
  # (adds config.yaml iterations to current checkpoint iteration)
  python train.py --resume
  
Note:
  - If you have 100 iterations saved and config.yaml has iterations: 50
  - Running with --resume will train from iteration 101 to 150
  - Without --resume, it starts fresh from iteration 1 to 50
        """
    )
    
    parser.add_argument(
        "--resume", 
        action='store_true', 
        help="Resume from latest checkpoint and train for config.yaml iterations"
    )
    
    args = parser.parse_args()
    
    train(resume=args.resume)

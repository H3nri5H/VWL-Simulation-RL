import os
import sys
import json
import yaml
import warnings
import argparse
import shutil
import time
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from env.economy_env import SimpleEconomyEnv

# Suppress warnings and verbose output
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore')
os.environ['RAY_DEDUP_LOGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Redirect Ray's verbose output
import logging
logging.getLogger('ray').setLevel(logging.ERROR)
logging.getLogger('ray.tune').setLevel(logging.ERROR)
logging.getLogger('ray.rllib').setLevel(logging.ERROR)


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            "config.yaml not found! Please create a config.yaml file with your training parameters."
        )
    with open(config_path, 'r', encoding='utf-8') as f:
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
                    checkpoints.append((os.path.abspath(str(cp_dir)), metadata.get('iteration', 0)))
    
    if not checkpoints:
        return None, 0
    
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
                time.sleep(0.5)
                continue
            else:
                print(f"  Warning: Could not delete {path} (files in use)")
                return False
    return False


def train(resume=False):
    """Train the VWL simulation using config.yaml parameters"""
    config = load_config()
    
    env_cfg = config.get('environment', {})
    env_config = {
        'n_firms': env_cfg.get('n_firms', 2),
        'n_households': env_cfg.get('n_households', 10),
        'max_steps': env_cfg.get('max_steps', 100),
    }
    
    # Extract n_firms for policy creation
    n_firms = env_config['n_firms']
    
    train_cfg = config.get('training', {})
    config_iterations = train_cfg.get('iterations', 50)
    checkpoint_freq = train_cfg.get('checkpoint_frequency', 10)
    
    ppo_cfg = train_cfg.get('ppo', {})
    lr = ppo_cfg.get('learning_rate', 3e-4)
    gamma_val = ppo_cfg.get('gamma', 0.99)
    lambda_val = ppo_cfg.get('lambda', 0.95)
    clip_val = ppo_cfg.get('clip_param', 0.2)
    train_batch = ppo_cfg.get('train_batch_size', 400)
    minibatch = ppo_cfg.get('minibatch_size', 128)
    epochs = ppo_cfg.get('num_epochs', 10)
    
    res_cfg = train_cfg.get('resources', {})
    n_workers = res_cfg.get('num_env_runners', 2)
    n_gpus = res_cfg.get('num_gpus', 0)
    
    checkpoint_dir = os.path.abspath("./checkpoints")
    metrics_dir = os.path.abspath("./metrics")
    
    print("\n" + "="*70)
    print("  VWL SIMULATION - TRAINING")
    print("="*70)
    
    start_iteration = 0
    resume_checkpoint = None
    total_iterations = config_iterations
    
    if resume:
        resume_checkpoint, start_iteration = find_latest_checkpoint()
        if resume_checkpoint:
            total_iterations = start_iteration + config_iterations
            print(f"\nResuming from iteration {start_iteration} -> {total_iterations}")
        else:
            print(f"\nNo checkpoint found, starting fresh")
            resume = False
    else:
        print(f"\nFresh training: {config_iterations} iterations")
    
    if not resume:
        safe_rmtree(checkpoint_dir)
        safe_rmtree(metrics_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
    
    print(f"Environment: {env_config['n_firms']} firms, {env_config['n_households']} households")
    print(f"Resources: {n_workers} workers, {n_gpus} GPUs")
    print(f"Policy setup: {n_firms} independent firm policies (heterogeneous learning)")
    print("\nBuilding algorithm...")
    
    if resume and resume_checkpoint:
        # Suppress Ray's output during loading
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            algo = PPO.from_checkpoint(os.path.abspath(resume_checkpoint))
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
        print("Algorithm loaded from checkpoint")
    else:
        # Create separate policy for each firm
        env_temp = SimpleEconomyEnv(env_config)
        obs_space = env_temp.observation_space
        act_space = env_temp.action_space
        
        policies = {}
        for i in range(n_firms):
            policy_id = f"firm_{i}"
            policies[policy_id] = (None, obs_space, act_space, {})
        
        def policy_mapping_fn(agent_id, *args, **kwargs):
            """Map each firm agent to its own policy"""
            return agent_id  # agent_id is already "firm_0", "firm_1", etc.
        
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
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
            )
            .resources(
                num_gpus=n_gpus,
            )
        )
        
        # Suppress Ray's initialization output
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            algo = rllib_config.build()
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
        print("Algorithm built")
    
    print("\n" + "-"*70)
    print(f"{'Iter':<6} {'Reward':<12} {'Min':<10} {'Max':<10} {'EpLen':<8}")
    print("-"*70)
    
    for i in range(start_iteration, total_iterations):
        # Suppress Ray's training output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        try:
            result = algo.train()
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        env_runners = result.get('env_runners', {})
        reward_mean = env_runners.get('episode_reward_mean', 0.0)
        reward_min = env_runners.get('episode_reward_min', 0.0)
        reward_max = env_runners.get('episode_reward_max', 0.0)
        episode_len = env_runners.get('episode_len_mean', 0.0)
        
        print(f"{i+1:<6} {reward_mean:<12.2f} {reward_min:<10.2f} {reward_max:<10.2f} {episode_len:<8.0f}")
        
        should_checkpoint = (i + 1) % checkpoint_freq == 0 or (i + 1) == total_iterations
        
        if should_checkpoint:
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
            
            checkpoint_name = f"checkpoint_{i+1:06d}"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Suppress checkpoint saving output
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                checkpoint_result = algo.save()
                checkpoint_result.checkpoint.to_directory(checkpoint_path)
            finally:
                sys.stdout.close()
                sys.stdout = old_stdout
            
            metadata_file = os.path.join(checkpoint_path, "metadata.json")
            is_final = (i + 1) == total_iterations
            
            metadata = {
                'iteration': i + 1,
                'reward_mean': reward_mean,
                'episode_len_mean': episode_len,
                'timestamp': result.get('timestamp', 0),
                'is_favorite': is_final,
                'checkpoint_path': checkpoint_path,
                'env_config': env_config
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            marker = "[*]" if is_final else "[+]"
            print(f"  {marker} Checkpoint saved: iteration {i+1}")
    
    print("-"*70)
    print("\nTraining complete!\n")
    
    algo.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train VWL simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--resume", 
        action='store_true', 
        help="Resume from latest checkpoint"
    )
    
    args = parser.parse_args()
    train(resume=args.resume)

import os
import json
import warnings
import argparse
import shutil
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig
from env.economy_env import SimpleEconomyEnv

warnings.filterwarnings('ignore', category=DeprecationWarning)


def train(iterations=50, checkpoint_freq=10):
    env_config = {
        'n_firms': 2,
        'n_households': 10,
        'max_steps': 100,
    }
    
    # Use absolute paths
    checkpoint_dir = os.path.abspath("./checkpoints")
    metrics_dir = os.path.abspath("./metrics")
    
    # Clear old data from previous training runs
    print("Clearing old training data...")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print(f"  ✓ Removed old checkpoints")
    if os.path.exists(metrics_dir):
        shutil.rmtree(metrics_dir)
        print(f"  ✓ Removed old metrics")
    
    # Create fresh directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    config = (
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
            num_env_runners=2,
            rollout_fragment_length=200,
        )
        .training(
            train_batch_size=400,
            minibatch_size=128,
            num_epochs=10,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
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
            num_gpus=0,
        )
    )
    
    print(f"\nStarting new training: {iterations} iterations")
    print(f"Firms: {env_config['n_firms']}, Households: {env_config['n_households']}")
    print(f"Checkpoint frequency: every {checkpoint_freq} iterations\n")
    
    algo = config.build()
    
    for i in range(iterations):
        result = algo.train()
        
        env_runners = result.get('env_runners', {})
        reward_mean = env_runners.get('episode_reward_mean', 0.0)
        episode_len = env_runners.get('episode_len_mean', 0.0)
        
        print(f"[{i+1}/{iterations}] Reward: {reward_mean:.2f}, Length: {episode_len:.0f}")
        
        # Only save metrics at checkpoint intervals (not every iteration)
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
                        'episode_reward_min': env_runners.get('episode_reward_min', 0.0),
                        'episode_reward_max': env_runners.get('episode_reward_max', 0.0),
                        'episode_len_mean': episode_len,
                    }
                }, f)
            
            print(f"\n=== SAVING CHECKPOINT ===")
            print(f"Target directory: {checkpoint_dir}")
            
            # Save full checkpoint
            checkpoint_result = algo.save(checkpoint_dir)
            
            print(f"\nCheckpoint result type: {type(checkpoint_result)}")
            print(f"Checkpoint result value: {checkpoint_result}")
            print(f"Checkpoint result dir: {dir(checkpoint_result)}")
            
            # Get actual checkpoint path - try multiple methods
            checkpoint_path = None
            
            if hasattr(checkpoint_result, 'checkpoint'):
                print(f"Has .checkpoint attribute")
                checkpoint_path = checkpoint_result.checkpoint.path
            elif hasattr(checkpoint_result, 'path'):
                print(f"Has .path attribute")
                checkpoint_path = checkpoint_result.path
            elif isinstance(checkpoint_result, (str, os.PathLike)):
                print(f"Is string/PathLike")
                checkpoint_path = str(checkpoint_result)
            else:
                print(f"Unknown type, converting to string")
                checkpoint_path = str(checkpoint_result)
            
            print(f"\nExtracted checkpoint path: {checkpoint_path}")
            print(f"Path exists: {os.path.exists(checkpoint_path)}")
            print(f"Path is dir: {os.path.isdir(checkpoint_path) if os.path.exists(checkpoint_path) else 'N/A'}")
            
            # List all directories in checkpoint_dir
            print(f"\nAll items in {checkpoint_dir}:")
            if os.path.exists(checkpoint_dir):
                for item in os.listdir(checkpoint_dir):
                    item_path = os.path.join(checkpoint_dir, item)
                    print(f"  - {item} (dir: {os.path.isdir(item_path)})")
            
            # Try to find ANY checkpoint directory
            checkpoint_base = Path(checkpoint_dir)
            if checkpoint_base.exists():
                checkpoint_dirs = [d for d in checkpoint_base.iterdir() if d.is_dir()]
                print(f"\nFound {len(checkpoint_dirs)} directories in checkpoint folder")
                
                if checkpoint_dirs:
                    # Use most recent
                    actual_path = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)
                    checkpoint_path = str(actual_path)
                    print(f"Using most recent: {checkpoint_path}")
            
            # Save metadata
            if checkpoint_path and os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
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
                
                print(f"\n✓ Metadata saved: {metadata_file}")
                print(f"Metadata content: {metadata}")
                
                if is_final:
                    print(f"⭐ Marked as favorite\n")
            else:
                print(f"\n⚠️ Could not save metadata - invalid checkpoint path")
            
            print("=" * 50 + "\n")
    
    print(f"\n✅ Training complete!")
    print(f"\nFinal checkpoint directory listing:")
    if os.path.exists(checkpoint_dir):
        for item in Path(checkpoint_dir).iterdir():
            if item.is_dir():
                print(f"  - {item.name}")
                metadata_file = item / "metadata.json"
                if metadata_file.exists():
                    print(f"    ✓ Has metadata.json")
    
    algo.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    args = parser.parse_args()
    
    train(iterations=args.iterations, checkpoint_freq=args.checkpoint_freq)

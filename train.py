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
            
            # Save full checkpoint
            checkpoint_result = algo.save(checkpoint_dir)
            
            # Get actual checkpoint path
            if hasattr(checkpoint_result, 'checkpoint'):
                checkpoint_path = checkpoint_result.checkpoint.path
            else:
                checkpoint_path = str(checkpoint_result)
            
            print(f"\n✓ Checkpoint saved to: {checkpoint_path}")
            
            # Verify checkpoint exists and save metadata
            if os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
                metadata_file = os.path.join(checkpoint_path, "metadata.json")
                is_final = (i + 1) == iterations
                
                with open(metadata_file, 'w') as f:
                    json.dump({
                        'iteration': i + 1,
                        'reward_mean': reward_mean,
                        'episode_len_mean': episode_len,
                        'timestamp': result.get('timestamp', 0),
                        'is_favorite': is_final,
                        'checkpoint_path': checkpoint_path
                    }, f)
                
                print(f"  ✓ Metadata saved: {metadata_file}")
                
                if is_final:
                    print(f"  ⭐ Marked as favorite\n")
            else:
                print(f"  ⚠️ Warning: Checkpoint directory not found at {checkpoint_path}")
                # Try to find it in the checkpoint base directory
                checkpoint_base = Path(checkpoint_dir)
                checkpoint_dirs = sorted([d for d in checkpoint_base.iterdir() if d.is_dir()], 
                                       key=lambda p: p.stat().st_mtime, reverse=True)
                if checkpoint_dirs:
                    actual_path = checkpoint_dirs[0]
                    print(f"  Found checkpoint at: {actual_path}")
                    metadata_file = actual_path / "metadata.json"
                    is_final = (i + 1) == iterations
                    
                    with open(metadata_file, 'w') as f:
                        json.dump({
                            'iteration': i + 1,
                            'reward_mean': reward_mean,
                            'episode_len_mean': episode_len,
                            'timestamp': result.get('timestamp', 0),
                            'is_favorite': is_final,
                            'checkpoint_path': str(actual_path)
                        }, f)
                    print(f"  ✓ Metadata saved: {metadata_file}")
    
    print(f"\n✅ Training complete!")
    print(f"\nAll checkpoints saved in: {checkpoint_dir}")
    print(f"\nCheckpoint contents:")
    for item in Path(checkpoint_dir).iterdir():
        if item.is_dir():
            print(f"  - {item.name}")
    
    algo.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    args = parser.parse_args()
    
    train(iterations=args.iterations, checkpoint_freq=args.checkpoint_freq)

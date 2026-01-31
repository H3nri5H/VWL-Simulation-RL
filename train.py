import os
import json
import warnings
import argparse
from ray.rllib.algorithms.ppo import PPOConfig
from env.economy_env import SimpleEconomyEnv

warnings.filterwarnings('ignore', category=DeprecationWarning)


def train(iterations=50, checkpoint_freq=10):
    env_config = {
        'n_firms': 2,
        'n_households': 10,
        'max_steps': 100,
    }
    
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
    
    print(f"Training: {iterations} iterations")
    print(f"Firms: {env_config['n_firms']}, Households: {env_config['n_households']}")
    
    algo = config.build()
    
    checkpoint_dir = "./checkpoints"
    metrics_dir = "./metrics"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    for i in range(iterations):
        result = algo.train()
        
        env_runners = result.get('env_runners', {})
        reward_mean = env_runners.get('episode_reward_mean', 0.0)
        episode_len = env_runners.get('episode_len_mean', 0.0)
        
        print(f"[{i+1}/{iterations}] Reward: {reward_mean:.2f}, Length: {episode_len:.0f}")
        
        # Save metrics for dashboard every iteration (lightweight)
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
        
        # Only save full checkpoints at specified intervals
        if (i + 1) % checkpoint_freq == 0:
            checkpoint_result = algo.save(checkpoint_dir)
            checkpoint_path = checkpoint_result.checkpoint.path
            print(f"Checkpoint saved: {checkpoint_path}")
            
            # Save metadata
            metadata_file = os.path.join(checkpoint_path, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump({
                    'iteration': i + 1,
                    'reward_mean': reward_mean,
                    'timestamp': result.get('timestamp', 0),
                    'is_favorite': False
                }, f)
    
    # Save final checkpoint and mark as favorite
    final_result = algo.save(checkpoint_dir)
    final_checkpoint = final_result.checkpoint.path
    print(f"\nTraining complete: {final_checkpoint}")
    
    # Mark final checkpoint as favorite
    metadata_file = os.path.join(final_checkpoint, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump({
            'iteration': iterations,
            'reward_mean': reward_mean,
            'timestamp': result.get('timestamp', 0),
            'is_favorite': True
        }, f)
    
    algo.stop()
    return final_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    args = parser.parse_args()
    
    train(iterations=args.iterations, checkpoint_freq=args.checkpoint_freq)

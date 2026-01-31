"""Simplified Training Script for VWL-Simulation using direct Gymnasium env."""

import os
import argparse
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from env.simple_economy_env import SimpleEconomyEnv


class RLlibMultiAgentEnv(MultiAgentEnv):
    """Wrapper to make SimpleEconomyEnv compatible with RLlib's MultiAgentEnv."""
    
    def __init__(self, config=None):
        super().__init__()
        self.env = SimpleEconomyEnv(config)
        self._agent_ids = self.env._agent_ids
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
    
    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)
    
    def step(self, action_dict):
        return self.env.step(action_dict)


def train(iterations=100, checkpoint_freq=10):
    """Train multi-agent PPO on economy simulation.
    
    Args:
        iterations: Number of training iterations
        checkpoint_freq: Save checkpoint every N iterations
    """
    print("\n" + "="*60)
    print("VWL-Simulation Multi-Agent RL Training (Simplified)")
    print("="*60)
    
    # Environment configuration
    env_config = {
        'n_firms': 2,
        'n_households': 10,
        'max_steps': 100,
    }
    
    print(f"\nEnvironment Config:")
    print(f"  - Firms: {env_config['n_firms']}")
    print(f"  - Households: {env_config['n_households']}")
    print(f"  - Max Steps: {env_config['max_steps']}")
    
    # Create sample env to get agent IDs
    sample_env = SimpleEconomyEnv(env_config)
    agent_ids = list(sample_env._agent_ids)
    
    # RLlib configuration
    config = (
        PPOConfig()
        .environment(
            env=RLlibMultiAgentEnv,
            env_config=env_config,
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=2,
            rollout_fragment_length=200,
        )
        .training(
            train_batch_size=400,
            sgd_minibatch_size=128,
            num_sgd_iter=10,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
        )
        .multi_agent(
            policies={
                "firm_policy": (
                    None,  # Use default policy
                    sample_env.observation_space,
                    sample_env.action_space,
                    {},
                )
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "firm_policy",
        )
        .resources(
            num_gpus=0,
        )
    )
    
    print(f"\nTraining Config:")
    print(f"  - Algorithm: PPO (Proximal Policy Optimization)")
    print(f"  - Framework: PyTorch")
    print(f"  - Workers: 2")
    print(f"  - Learning Rate: 0.0003")
    print(f"  - Total Iterations: {iterations}")
    print(f"  - Checkpoint Every: {checkpoint_freq} iterations\n")
    
    # Build algorithm
    algo = config.build()
    
    # Create checkpoint directory
    checkpoint_dir = "./checkpoints_simple"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    print("Starting training...\n")
    
    for i in range(iterations):
        result = algo.train()
        
        # Print progress
        print(f"Iteration {i+1}/{iterations}:")
        print(f"  Reward: {result['episode_reward_mean']:.2f} "
              f"(min: {result['episode_reward_min']:.2f}, max: {result['episode_reward_max']:.2f})")
        print(f"  Episode Length: {result['episode_len_mean']:.1f}")
        
        # Save checkpoint
        if (i + 1) % checkpoint_freq == 0:
            checkpoint_path = algo.save(checkpoint_dir)
            print(f"  \u2713 Checkpoint saved: {checkpoint_path}")
        
        print()
    
    # Final checkpoint
    final_checkpoint = algo.save(checkpoint_dir)
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Final checkpoint: {final_checkpoint}")
    print(f"{'='*60}\n")
    
    algo.stop()
    
    return final_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VWL-Simulation Multi-Agent RL")
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations (default: 100)"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10,
        help="Save checkpoint every N iterations (default: 10)"
    )
    
    args = parser.parse_args()
    
    train(iterations=args.iterations, checkpoint_freq=args.checkpoint_freq)

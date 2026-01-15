"""Multi-Agent Training mit RLlib"""

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import ParallelPettingZooEnv
from envs.economy_env import EconomyEnv
from ray.tune.registry import register_env

print("\n" + "="*60)
print("MULTI-AGENT TRAINING - 3 Competing Firms")
print("="*60 + "\n")

# Environment registrieren
def env_creator(config):
    return ParallelPettingZooEnv(EconomyEnv(**config))

register_env("economy_marl", env_creator)

# Config
config = (
    PPOConfig()
    .environment(
        env="economy_marl",
        env_config={
            "n_firms": 3,
            "n_households": 30,
            "max_steps": 100
        }
    )
    .framework("tf2")
    .training(
        lr=0.0003,
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=10,
        entropy_coeff=0.01  # Exploration
    )
    .rollouts(num_rollout_workers=4)
    .resources(num_gpus=0)
    .multi_agent(
        policies={f"firm_{i}" for i in range(3)},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id
    )
)

print("Starte Multi-Agent Training...")
print("3 Firmen lernen simultan zu konkurrieren\n")

# Training
algo = config.build()

for i in range(100):
    result = algo.train()
    
    if i % 10 == 0:
        print(f"\nIteration {i}:")
        print(f"  Reward Mean: {result['episode_reward_mean']:.2f}")
        print(f"  Episode Length: {result['episode_len_mean']:.0f}")
        
        # Policy-spezifische Rewards
        for policy_id in ["firm_0", "firm_1", "firm_2"]:
            if policy_id in result['policy_reward_mean']:
                print(f"    {policy_id}: {result['policy_reward_mean'][policy_id]:.2f}")

algo.save("models/marl_3firms")
print("\n✓ Training abgeschlossen!")
print("→ Teste mit: python test.py\n")

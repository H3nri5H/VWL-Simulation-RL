"""Train a single firm to understand the environment"""

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from envs.economy_env import EconomyEnv
from ray.tune.registry import register_env

print("\n" + "="*60)
print("SINGLE FIRM TRAINING - Proof of Concept")
print("="*60 + "\n")

# Environment registrieren
register_env("economy", lambda config: EconomyEnv(**config))

# Config
config = (
    PPOConfig()
    .environment(
        env="economy",
        env_config={
            "n_firms": 1,  # Nur 1 Firma zum Testen
            "n_households": 20,
            "max_steps": 50
        }
    )
    .framework("tf2")
    .training(
        lr=0.0003,
        train_batch_size=2000,
        sgd_minibatch_size=128,
        num_sgd_iter=10
    )
    .rollouts(num_rollout_workers=2)
    .resources(num_gpus=0)
)

print("Starte Training...\n")
print("⚠️ Dies ist ein SINGLE-AGENT Test")
print("Wenn erfolgreich: Wechsel zu Multi-Agent\n")

# Training
algo = config.build()

for i in range(30):
    result = algo.train()
    
    if i % 10 == 0:
        print(f"\nIteration {i}:")
        print(f"  Reward: {result['episode_reward_mean']:.2f}")
        print(f"  Length: {result['episode_len_mean']:.0f}")

algo.save("models/single_firm")
print("\n✓ Training erfolgreich!")
print("→ Nächster Schritt: train_marl.py\n")

from stable_baselines3 import PPO
from economy_env import MultiAgentEconomyEnv

# Environment mit 50 Haushalten und 10 Unternehmen
env = MultiAgentEconomyEnv(n_households=50, n_firms=10)

# PPO Modell
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95
)

print("\n" + "="*60)
print("Training Multi-Agent Volkswirtschafts-Simulation")
print(f"Haushalte: {env.n_households}, Unternehmen: {env.n_firms}")
print("="*60 + "\n")

# Training
model.learn(total_timesteps=200000)

# Speichern
model.save("economy_multiagent_model")
print("\n✓ Modell gespeichert als 'economy_multiagent_model'")

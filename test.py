from stable_baselines3 import PPO
from environment import EconomyEnv
import numpy as np

# Environment und trainiertes Modell laden
env = EconomyEnv()
model = PPO.load("economy_model")

# Testen
obs, _ = env.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i}: BIP={obs[0]:.2f}, Kapital={obs[1]:.2f}, Reward={reward:.2f}")
    
    if terminated or truncated:
        break

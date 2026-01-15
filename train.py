from stable_baselines3 import PPO
from environment import EconomyEnv

# Environment erstellen
env = EconomyEnv()

# Modell erstellen (PPO = Proximal Policy Optimization)
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

# Training
print("Starte Training...")
model.learn(total_timesteps=50000)

# Modell speichern
model.save("economy_model")
print("Modell gespeichert!")

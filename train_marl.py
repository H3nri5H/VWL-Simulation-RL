"""Training script für Multi-Agent RL Volkswirtschaft"""

import gymnasium as gym
from stable_baselines3 import PPO
from marl_economy_env import MARLEconomyEnv
import numpy as np

print("\n" + "="*60)
print("MULTI-AGENT RL VOLKSWIRTSCHAFT")
print("Training: 10 Firmen-Agents + 1 Regierungs-Agent")
print("="*60 + "\n")

# === ENVIRONMENT ERSTELLEN ===
env = MARLEconomyEnv(n_households=50, n_firms=10)

print("Environment erstellt:")
print(f"  - {env.n_firms} Unternehmen (RL-Agents)")
print(f"  - {env.n_households} Haushalte (regelbasiert)")
print(f"  - 1 Regierung (RL-Agent)")
print()

# === WICHTIG: VEREINFACHTES TRAINING ===
# Für echtes MARL brauchen wir komplexere Frameworks (RLlib, PettingZoo)
# Hier: Sequentielles Training als Proof-of-Concept

print("⚠️ HINWEIS: Dies ist eine vereinfachte Implementierung!")
print("Für echtes MARL empfehlen wir: RLlib oder PettingZoo\n")

# === WRAPPER FÜR STABLE-BASELINES3 ===
# SB3 unterstützt kein natives Multi-Agent Learning
# Wir trainieren: Government Agent (Firmen machen erstmal Random Actions)

class GovOnlyWrapper(gym.Env):
    """Wrapper der nur Government agent trainiert, Firmen machen random"""
    
    def __init__(self, base_env):
        super().__init__()
        self.env = base_env
        self.action_space = base_env.gov_action_space
        self.observation_space = base_env.gov_observation_space
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs['government'], info
    
    def step(self, gov_action):
        # Firmen machen random actions
        firm_actions = np.array([
            self.env.firm_action_space.sample() 
            for _ in range(self.env.n_firms)
        ])
        
        actions = {
            'firms': firm_actions,
            'government': gov_action
        }
        
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        
        return obs['government'], rewards['government'], terminated, truncated, info

wrapped_env = GovOnlyWrapper(env)

print("Starte Training (Government Agent mit Random Firms)...")
print("Timesteps: 100.000 (ca. 10-15 Minuten)\n")

# === MODELL TRAINIEREN ===
model = PPO(
    "MlpPolicy",
    wrapped_env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10
)

model.learn(total_timesteps=100000)

# === SPEICHERN ===
model.save("economy_gov_agent")
print("\n✓ Modell gespeichert: economy_gov_agent")

print("\n" + "="*60)
print("NÄCHSTE SCHRITTE:")
print("="*60)
print("1. Für echtes MARL: Integration mit RLlib")
print("2. Testen: python test_marl.py")
print("3. Analyse der emergenten Verhaltensweisen")
print()

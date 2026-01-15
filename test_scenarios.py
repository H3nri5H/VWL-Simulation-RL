from stable_baselines3 import PPO
from environment import EconomyEnv
from scenarios import ScenarioWrapper
import numpy as np

# Modell laden
model = PPO.load("economy_model")

# Verschiedene Szenarien testen
scenarios = [
    ("normal", "🟢 Normal"),
    ("tax_increase", "💰 Steuererhöhung"),
    ("natural_disaster", "🌪️ Naturkatastrophe"),
    ("recession", "📉 Rezession"),
    ("boom", "📈 Boom"),
    ("labor_shortage", "👷 Arbeitskräftemangel")
]

print("="*60)
print("SZENARIO-TESTS: Wie reagiert die KI auf Schocks?")
print("="*60)

for scenario_type, name in scenarios:
    print(f"\n{'='*60}")
    print(f"Szenario: {name}")
    print(f"{'='*60}")
    
    env = EconomyEnv()
    wrapped_env = ScenarioWrapper(env, scenario_type=scenario_type)
    
    obs, _ = wrapped_env.reset()
    total_reward = 0
    bip_start = obs[0]
    
    for i in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        total_reward += reward
        
        # Nur wichtige Steps zeigen
        if i % 20 == 0 or i == 99:
            print(f"Step {i:3d}: BIP={obs[0]:6.2f}, Kapital={obs[1]:7.2f}, Reward={reward:6.2f}")
        
        if terminated or truncated:
            break
    
    bip_end = obs[0]
    bip_change = ((bip_end - bip_start) / bip_start) * 100
    
    print(f"\n📊 Ergebnis:")
    print(f"   Total Reward: {total_reward:.2f}")
    print(f"   BIP-Änderung: {bip_change:+.1f}%")
    print(f"   End-Kapital: {obs[1]:.2f}")

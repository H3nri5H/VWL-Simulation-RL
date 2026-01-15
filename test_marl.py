"""Test-Script für Multi-Agent Economy"""

from marl_economy_env import MARLEconomyEnv
import numpy as np
import time

print("\n" + "="*60)
print("MULTI-AGENT ECONOMY - SIMULATION")
print("="*60 + "\n")

# Environment erstellen
env = MARLEconomyEnv(n_households=50, n_firms=10)
obs, _ = env.reset()

print("Simulation gestartet...\n")
print("⚠️ Firmen treffen ZUFALLS-Entscheidungen (kein Training)\n")

# === SIMULATION MIT RANDOM ACTIONS ===
for step in range(50):
    # Random actions für alle Agents
    firm_actions = np.array([
        env.firm_action_space.sample() 
        for _ in range(env.n_firms)
    ])
    gov_action = env.gov_action_space.sample()
    
    actions = {
        'firms': firm_actions,
        'government': gov_action
    }
    
    obs, rewards, terminated, truncated, info = env.step(actions)
    
    # Alle 10 Steps ausgeben
    if step % 10 == 0:
        env.render()
        print(f"\nRegierungs-Reward: {rewards['government']:.2f}")
        print(f"Durchschnittlicher Firmen-Reward: {np.mean(rewards['firms']):.2f}")
        print(f"Beste Firma Reward: {np.max(rewards['firms']):.2f}")
        print(f"Schlechteste Firma Reward: {np.min(rewards['firms']):.2f}")
        time.sleep(0.5)
    
    if terminated:
        print("\nSimulation beendet!")
        break

print("\n" + "="*60)
print("ERGEBNISSE")
print("="*60)

# Firmen-Statistiken
print("\nFirmen-Ranking nach Gewinn:")
firmen_sorted = sorted(env.firms, key=lambda f: f.gewinn, reverse=True)
for i, firm in enumerate(firmen_sorted[:5]):
    state = firm.get_state()
    print(f"{i+1}. Firma {state['id']}: "
          f"Gewinn={state['gewinn']:.0f}, "
          f"Preis={state['preis']:.0f}, "
          f"Marktanteil={state['marktanteil']*100:.1f}%")

print("\n💡 HINWEIS:")
print("Dies sind RANDOM actions. Für intelligentes Verhalten:")
print("1. Implementiere RLlib Multi-Agent Training")
print("2. Trainiere alle Firmen simultan")
print("3. Beobachte emergente Strategien (Kartelle, Preiskampf, etc.)")

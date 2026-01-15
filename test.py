"""Test trained agents"""

from envs.economy_env import EconomyEnv
import numpy as np
from ray.rllib.algorithms.ppo import PPO

print("\n" + "="*60)
print("TESTING TRAINED AGENTS")
print("="*60 + "\n")

# Load trained model
try:
    algo = PPO.from_checkpoint("models/marl_3firms")
    print("✓ Modell geladen: models/marl_3firms\n")
except:
    print("⚠️ Kein trainiertes Modell gefunden!")
    print("Verwende RANDOM actions als Demo\n")
    algo = None

# Environment erstellen
env = EconomyEnv(n_firms=3, n_households=30, max_steps=50)
observations, infos = env.reset()

print("Simulation startet...\n")

for step in range(50):
    # Actions von trainierten Agents (oder random)
    if algo:
        actions = {}
        for agent in env.agents:
            actions[agent] = algo.compute_single_action(
                observations[agent],
                policy_id=agent
            )
    else:
        # Random fallback
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    
    # Step
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # Render every 10 steps
    if step % 10 == 0:
        env.render()
        print(f"\nRewards: {[f'{r:.1f}' for r in rewards.values()]}")
    
    # Check if done
    if all(terminations.values()):
        break

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

for agent in env.agents:
    state = env.firms[agent].get_state()
    print(f"\n{agent}:")
    print(f"  Gewinn: {state['gewinn']}€")
    print(f"  Marktanteil: {state['marktanteil']}%")
    print(f"  Preis: {state['preis']}€")
    print(f"  Mitarbeiter: {state['mitarbeiter']}")

print("\n💡 Nächste Schritte:")
print("- Analysiere Preisstrategien")
print("- Teste mit Szenarien (Wirtschaftsschocks)")
print("- Trainiere mit mehr Firmen\n")

"""Test script for EconomyEnv - runs without Ray/TensorFlow.

Dieses Script testet das Environment manuell, ohne RL-Training.
Funktioniert auf Windows ohne DLL-Probleme!
"""

import numpy as np
from env.economy_env import EconomyEnv


def test_environment():
    """Testet das Economy Environment mit zufälligen Actions."""
    
    print("=" * 60)
    print("VWL-Simulation Environment Test")
    print("=" * 60)
    
    # Environment erstellen
    env = EconomyEnv(n_firms=2, n_households=10, max_steps=20)
    
    # Reset
    obs, info = env.reset(seed=42)
    print(f"\nEnvironment initialisiert!")
    print(f"Agents: {env.agents}")
    print(f"Initial Observation für {env.agent_selection}: {obs}")
    
    # 20 Steps durchlaufen
    step_count = 0
    
    while not all(env.terminations.values()):
        # Aktueller Agent
        agent = env.agent_selection
        
        # Zufällige Action wählen (später lernt die KI optimale Actions)
        action = env.action_spaces[agent].sample()
        
        # Step ausführen
        env.step(action)
        
        # Wenn alle Agents agiert haben, ist ein Market-Step vollendet
        if env.agent_selection == env.agents[0]:
            step_count += 1
            
            # Render: Zeige Wirtschafts-State
            env.render()
            
            # Zeige Rewards
            print("  Rewards:")
            for agent_name, reward in env.rewards.items():
                print(f"    {agent_name}: {reward:.3f}")
    
    print("\n" + "=" * 60)
    print("Simulation beendet!")
    print(f"Total Steps: {step_count}")
    print("=" * 60)
    
    # Finale Statistiken
    print("\nFinale Firmen-States:")
    for name, firm in env.firms.items():
        print(f"  {name}:")
        print(f"    Preis: {firm['price']:.2f}€")
        print(f"    Lohn: {firm['wage']:.2f}€")
        print(f"    Kapital: {firm['capital']:.2f}€")
        print(f"    Inventory: {firm['inventory']:.0f}")
        print(f"    Profit (letzter Step): {firm['profit']:.2f}€")
    
    print("\nFinale Haushalts-States (erste 3):")
    for i, household in enumerate(env.households[:3]):
        print(f"  Haushalt {i}:")
        print(f"    Geld: {household['money']:.2f}€")
        print(f"    Arbeitgeber: {household['employer']}")


if __name__ == '__main__':
    test_environment()

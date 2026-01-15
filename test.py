from stable_baselines3 import PPO
from economy_env import MultiAgentEconomyEnv

env = MultiAgentEconomyEnv(n_households=50, n_firms=10)
model = PPO.load("economy_multiagent_model")

obs, _ = env.reset()
print("\n" + "="*60)
print("Testing trained policy")
print("="*60)

for i in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if i % 20 == 0:
        env.render()
        print(f"  Reward: {reward:.2f}")
        print(f"  Policy: Steuern={action[0]:.2%}, Ausgaben={action[1]:.2%}, Zins={action[2]:.2%}")
    
    if terminated or truncated:
        break

print("\n" + "="*60)
print("Test abgeschlossen")
print("="*60)

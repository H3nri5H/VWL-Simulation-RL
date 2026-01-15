import gymnasium as gym
from gymnasium import spaces
import numpy as np

class EconomyEnv(gym.Env):
    """Einfache Volkswirtschafts-Simulation mit Reinforcement Learning"""
    
    def __init__(self):
        super(EconomyEnv, self).__init__()
        
        # Action space: [Produktion, Konsum, Investition]
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        # Observation space: [BIP, Kapital, Arbeitskräfte, Inflation]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
        
        # Initiale Werte
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bip = 100.0
        self.kapital = 50.0
        self.arbeitskraefte = 100.0
        self.inflation = 0.02
        self.timestep = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        return np.array([self.bip, self.kapital, self.arbeitskraefte, self.inflation], dtype=np.float32)
    
    def step(self, action):
        produktion, konsum, investition = action
        
        # Einfache Wirtschaftslogik
        self.kapital += investition * 10 - 2  # Kapitalakkumulation
        self.bip = produktion * self.kapital * 0.5 + self.arbeitskraefte * 0.3
        self.inflation = 0.02 + (konsum - 0.5) * 0.01
        
        # Reward: Balance zwischen BIP-Wachstum und Stabilität
        reward = self.bip * 0.01 - abs(self.inflation - 0.02) * 50
        
        self.timestep += 1
        terminated = self.timestep >= 100
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}

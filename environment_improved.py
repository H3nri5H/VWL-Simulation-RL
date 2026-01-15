import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ImprovedEconomyEnv(gym.Env):
    """Verbesserte Volkswirtschafts-Simulation mit realistischeren Mechanismen"""
    
    def __init__(self):
        super(ImprovedEconomyEnv, self).__init__()
        
        # Action space: [Investitionsrate, Konsumquote, Steuersatz]
        self.action_space = spaces.Box(low=np.array([0.0, 0.3, 0.1]), 
                                       high=np.array([0.5, 0.9, 0.5]), 
                                       dtype=np.float32)
        
        # Observation: [BIP, Kapital, Arbeitskräfte, Inflation, Verschuldung]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bip = 100.0
        self.kapital = 50.0
        self.arbeitskraefte = 100.0
        self.inflation = 0.02
        self.verschuldung = 20.0
        self.timestep = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        return np.array([self.bip, self.kapital, self.arbeitskraefte, 
                        self.inflation, self.verschuldung], dtype=np.float32)
    
    def step(self, action):
        investitionsrate, konsumquote, steuersatz = action
        
        # 1. Produktion (Cobb-Douglas)
        alpha = 0.3  # Kapitalanteil
        produktion = (self.kapital ** alpha) * (self.arbeitskraefte ** (1 - alpha))
        
        # 2. BIP-Berechnung
        konsum = konsumquote * produktion
        investition = investitionsrate * produktion
        staatsausgaben = steuersatz * produktion
        self.bip = konsum + investition + staatsausgaben
        
        # 3. Kapitalakkumulation mit Abschreibung
        abschreibung = 0.05 * self.kapital
        self.kapital += investition - abschreibung
        self.kapital = max(0.1, self.kapital)  # Verhindere negative Werte
        
        # 4. Inflation (Phillips-Kurve)
        output_gap = (self.bip - 100.0) / 100.0
        self.inflation = 0.02 + 0.5 * output_gap + np.random.normal(0, 0.005)
        self.inflation = np.clip(self.inflation, -0.05, 0.15)
        
        # 5. Verschuldung
        self.verschuldung += staatsausgaben - steuersatz * self.bip
        self.verschuldung = max(0, self.verschuldung)
        
        # 6. Arbeitskräfte (leicht wachsend)
        self.arbeitskraefte *= 1.001
        
        # Reward: Balance zwischen Wachstum, Stabilität und Schulden
        bip_reward = (self.bip - 100.0) * 0.1  # Wachstum belohnen
        inflation_penalty = -abs(self.inflation - 0.02) * 100  # Ziel-Inflation 2%
        debt_penalty = -self.verschuldung * 0.01  # Schulden bestrafen
        capital_reward = self.kapital * 0.05  # Kapitalaufbau belohnen
        
        reward = bip_reward + inflation_penalty + debt_penalty + capital_reward
        
        self.timestep += 1
        terminated = self.timestep >= 200 or self.kapital < 1.0
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}

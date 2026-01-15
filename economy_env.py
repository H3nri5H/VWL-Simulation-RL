import gymnasium as gym
from gymnasium import spaces
import numpy as np
from agents import Household, Firm

class MultiAgentEconomyEnv(gym.Env):
    """Volkswirtschaft mit mehreren Haushalten und Unternehmen"""
    
    def __init__(self, n_households=50, n_firms=10):
        super().__init__()
        
        self.n_households = n_households
        self.n_firms = n_firms
        
        # Action: [Steuersatz, Staatsausgaben-Rate, Leitzins]
        self.action_space = spaces.Box(
            low=np.array([0.1, 0.05, 0.0]),
            high=np.array([0.5, 0.3, 0.1]),
            dtype=np.float32
        )
        
        # Observation: [BIP, Arbeitslosenquote, Inflation, Durchschnittslohn, 
        #               Durchschnittspreis, Staatsschulden, Gesamtkapital]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(7,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Agenten erstellen
        self.households = [Household(i) for i in range(self.n_households)]
        self.firms = [Firm(i) for i in range(self.n_firms)]
        
        # Makro-Variablen
        self.bip = 0.0
        self.inflation = 0.02
        self.staatsschulden = 5000.0
        self.timestep = 0
        self.preis_level_alt = 100.0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        arbeitende = sum(1 for h in self.households if h.arbeitet)
        arbeitslosenquote = 1.0 - (arbeitende / self.n_households)
        
        avg_lohn = np.mean([f.lohn for f in self.firms])
        avg_preis = np.mean([f.preis for f in self.firms])
        gesamtkapital = sum(f.kapital for f in self.firms)
        
        return np.array([
            self.bip,
            arbeitslosenquote,
            self.inflation,
            avg_lohn,
            avg_preis,
            self.staatsschulden,
            gesamtkapital
        ], dtype=np.float32)
    
    def step(self, action):
        steuersatz, staatsausgaben_rate, leitzins = action
        
        # 1. ARBEITSMARKT
        gesamte_arbeitskraefte = sum(h.arbeitsproduktivitaet for h in self.households)
        for firm in self.firms:
            firm.einstellen(gesamte_arbeitskraefte / self.n_firms)
        
        # Haushalte arbeiten
        avg_lohn = np.mean([f.lohn for f in self.firms])
        for household in self.households:
            household.arbeiten(avg_lohn)
        
        # 2. PRODUKTION
        for firm in self.firms:
            firm.produzieren()
        
        gesamtproduktion = sum(f.produktion for f in self.firms)
        
        # 3. KONSUMMARKT
        gesamtkonsum = 0.0
        avg_preis = np.mean([f.preis for f in self.firms])
        
        for household in self.households:
            konsum = household.konsumieren(avg_preis)
            gesamtkonsum += konsum
        
        nachfrage = gesamtkonsum / avg_preis  # In Einheiten
        
        # Firmen verkaufen
        for firm in self.firms:
            firm.verkaufen(nachfrage, gesamtproduktion)
            firm.preis_anpassen(nachfrage, gesamtproduktion)
        
        # 4. STAATSSEKTOR
        steuereinnahmen = steuersatz * gesamtproduktion * avg_preis
        staatsausgaben = staatsausgaben_rate * (gesamtproduktion * avg_preis)
        self.staatsschulden += (staatsausgaben - steuereinnahmen)
        
        # 5. BIP BERECHNEN
        investitionen = sum(f.gewinn * 0.3 for f in self.firms if f.gewinn > 0)
        self.bip = gesamtkonsum + investitionen + staatsausgaben
        
        # 6. INFLATION
        preis_level_neu = np.mean([f.preis for f in self.firms])
        self.inflation = (preis_level_neu - self.preis_level_alt) / self.preis_level_alt
        self.preis_level_alt = preis_level_neu
        
        # 7. REWARD
        bip_wachstum = (self.bip - 50000) / 50000  # Ziel: 50k BIP
        arbeitslose = sum(1 for h in self.households if not h.arbeitet)
        arbeitslosenquote = arbeitslose / self.n_households
        
        reward = (
            bip_wachstum * 100 +  # BIP-Wachstum belohnen
            -abs(self.inflation - 0.02) * 500 +  # Ziel-Inflation 2%
            -arbeitslosenquote * 100 +  # Arbeitslosigkeit bestrafen
            -self.staatsschulden * 0.001  # Schulden bestrafen
        )
        
        self.timestep += 1
        terminated = self.timestep >= 200 or self.staatsschulden > 50000
        
        return self._get_obs(), reward, terminated, False, {}
    
    def render(self):
        obs = self._get_obs()
        print(f"\nStep {self.timestep}:")
        print(f"  BIP: {obs[0]:.0f}")
        print(f"  Arbeitslosigkeit: {obs[1]*100:.1f}%")
        print(f"  Inflation: {obs[2]*100:.1f}%")
        print(f"  Avg Lohn: {obs[3]:.0f}")
        print(f"  Avg Preis: {obs[4]:.0f}")
        print(f"  Staatsschulden: {obs[5]:.0f}")

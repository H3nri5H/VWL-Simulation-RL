import numpy as np
from environment import EconomyEnv

class ScenarioWrapper:
    """Wrapper für verschiedene Wirtschafts-Szenarien"""
    
    def __init__(self, env, scenario_type="normal"):
        self.env = env
        self.scenario_type = scenario_type
        self.shock_step = None
        
    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.shock_step = np.random.randint(20, 50)  # Zufälliger Schock-Zeitpunkt
        return obs, info
    
    def step(self, action):
        # Schock anwenden wenn Zeitpunkt erreicht
        if self.env.timestep == self.shock_step:
            self._apply_shock()
        
        return self.env.step(action)
    
    def _apply_shock(self):
        """Verschiedene Wirtschaftsschocks"""
        if self.scenario_type == "tax_increase":
            # Steuererhöhung: Kapital sinkt um 20%
            self.env.kapital *= 0.8
            print(f"\n💰 SCHOCK (Step {self.env.timestep}): Steuererhöhung! Kapital -20%\n")
            
        elif self.scenario_type == "natural_disaster":
            # Naturkatastrophe: BIP -30%, Kapital -40%
            self.env.bip *= 0.7
            self.env.kapital *= 0.6
            self.env.arbeitskraefte *= 0.9
            print(f"\n🌪️ SCHOCK (Step {self.env.timestep}): Naturkatastrophe! BIP -30%, Kapital -40%, Arbeitskräfte -10%\n")
            
        elif self.scenario_type == "recession":
            # Rezession: Allmählicher BIP-Rückgang
            self.env.bip *= 0.85
            self.env.inflation += 0.01
            print(f"\n📉 SCHOCK (Step {self.env.timestep}): Rezession beginnt! BIP -15%, Inflation +1%\n")
            
        elif self.scenario_type == "boom":
            # Wirtschaftsboom
            self.env.bip *= 1.3
            self.env.kapital *= 1.2
            print(f"\n📈 SCHOCK (Step {self.env.timestep}): Wirtschaftsboom! BIP +30%, Kapital +20%\n")
            
        elif self.scenario_type == "labor_shortage":
            # Arbeitskräftemangel
            self.env.arbeitskraefte *= 0.7
            print(f"\n👷 SCHOCK (Step {self.env.timestep}): Arbeitskräftemangel! -30% Arbeitskräfte\n")

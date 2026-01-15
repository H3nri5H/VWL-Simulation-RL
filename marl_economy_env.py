import gymnasium as gym
from gymnasium import spaces
import numpy as np
from agents import Household, Firm

class MARLEconomyEnv(gym.Env):
    """
    Multi-Agent Reinforcement Learning Volkswirtschaft
    
    Agents:
    - n_firms RL-Agents (Unternehmen): Lernen Preise, Produktion, Löhne
    - 1 RL-Agent (Staat): Lernt Steuern, Ausgaben, Zinsen
    - n_households Regelbasierte Agents (Haushalte)
    
    Jedes Unternehmen ist ein eigener RL-Agent mit eigenem Observation/Action Space.
    """
    
    def __init__(self, n_households=50, n_firms=10):
        super().__init__()
        
        self.n_households = n_households
        self.n_firms = n_firms
        
        # === ACTION SPACES ===
        # Pro Firma: [Preisänderung, Mitarbeiteränderung, Lohnänderung]
        self.firm_action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Staat: [Steuersatz, Staatsausgaben-Rate, Leitzins]
        self.gov_action_space = spaces.Box(
            low=np.array([0.1, 0.05, 0.0]),
            high=np.array([0.5, 0.3, 0.1]),
            dtype=np.float32
        )
        
        # === OBSERVATION SPACES ===
        # Pro Firma: 10 Features (siehe Firm.get_observation)
        self.firm_observation_space = spaces.Box(
            low=0, high=np.inf, shape=(10,), dtype=np.float32
        )
        
        # Staat: Makroökonomische Indikatoren
        self.gov_observation_space = spaces.Box(
            low=0, high=np.inf, shape=(7,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # === AGENTEN ERSTELLEN ===
        self.households = [Household(i) for i in range(self.n_households)]
        self.firms = [Firm(i) for i in range(self.n_firms)]
        
        # === MAKRO-VARIABLEN ===
        self.bip = 0.0
        self.inflation = 0.02
        self.staatsschulden = 5000.0
        self.timestep = 0
        self.preis_level_alt = 100.0
        
        # Observations für alle Agents
        firm_obs = self._get_firm_observations()
        gov_obs = self._get_gov_observation()
        
        return {'firms': firm_obs, 'government': gov_obs}, {}
    
    def _get_markt_info(self):
        """Marktinformationen für alle Firmen"""
        avg_preis = np.mean([f.preis for f in self.firms])
        nachfrage = sum(h.vermoegen * h.konsumneigung for h in self.households) / avg_preis
        angebot = sum(f.produktion for f in self.firms)
        arbeitende = sum(1 for h in self.households if h.arbeitet)
        arbeitslosenquote = 1.0 - (arbeitende / self.n_households)
        
        return {
            'avg_preis': avg_preis,
            'nachfrage': nachfrage,
            'angebot': angebot,
            'arbeitslosenquote': arbeitslosenquote
        }
    
    def _get_firm_observations(self):
        """Observations für alle Firmen"""
        markt_info = self._get_markt_info()
        return np.array([firm.get_observation(markt_info) for firm in self.firms])
    
    def _get_gov_observation(self):
        """Observation für Regierung"""
        arbeitende = sum(1 for h in self.households if h.arbeitet)
        arbeitslosenquote = 1.0 - (arbeitande / self.n_households)
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
    
    def step(self, actions):
        """
        actions = {
            'firms': np.array shape (n_firms, 3),  # Aktionen aller Firmen
            'government': np.array shape (3,)      # Aktion der Regierung
        }
        """
        firm_actions = actions['firms']
        gov_action = actions['government']
        
        steuersatz, staatsausgaben_rate, leitzins = gov_action
        
        # === 1. FIRMEN: AKTIONEN ANWENDEN ===
        for i, firm in enumerate(self.firms):
            firm.apply_action(firm_actions[i])
        
        # === 2. ARBEITSMARKT ===
        # Durchschnittslohn am Markt
        avg_lohn = np.mean([f.lohn for f in self.firms])
        
        # Haushalte arbeiten
        for household in self.households:
            household.arbeiten(avg_lohn)
        
        # === 3. PRODUKTION ===
        for firm in self.firms:
            firm.produzieren()
        
        gesamtproduktion = sum(f.produktion for f in self.firms)
        
        # === 4. KONSUMMARKT ===
        avg_preis = np.mean([f.preis for f in self.firms])
        gesamtkonsum = 0.0
        
        for household in self.households:
            konsum = household.konsumieren(avg_preis)
            gesamtkonsum += konsum
        
        # Nachfrage in Einheiten
        nachfrage = gesamtkonsum / avg_preis if avg_preis > 0 else 0
        
        # === 5. FIRMEN VERKAUFEN (MIT PREISWETTBEWERB) ===
        alle_preise = [f.preis for f in self.firms]
        
        # Schritt 1: Berechne gewichtete Marktanteile
        gewichtete_anteile = []
        for firm in self.firms:
            if gesamtproduktion > 0:
                preis_faktor = (avg_preis / firm.preis) ** 2
                anteil = (firm.produktion / gesamtproduktion) * preis_faktor
                gewichtete_anteile.append(anteil)
            else:
                gewichtete_anteile.append(0)
        
        # Normalisieren
        summe_anteile = sum(gewichtete_anteile)
        if summe_anteile > 0:
            marktanteile = [a / summe_anteile for a in gewichtete_anteile]
        else:
            marktanteile = [1/self.n_firms] * self.n_firms
        
        # Schritt 2: Firmen verkaufen entsprechend Marktanteil
        for i, firm in enumerate(self.firms):
            firm.marktanteil = marktanteile[i]
            verkaufte_menge = min(firm.produktion, nachfrage * firm.marktanteil)
            firm.verkaufen(nachfrage, gesamtproduktion, alle_preise)
        
        # === 6. STAATSSEKTOR ===
        steuereinnahmen = steuersatz * gesamtproduktion * avg_preis
        staatsausgaben = staatsausgaben_rate * (gesamtproduktion * avg_preis)
        self.staatsschulden += (staatsausgaben - steuereinnahmen)
        
        # === 7. BIP BERECHNEN ===
        investitionen = sum(f.gewinn * 0.3 for f in self.firms if f.gewinn > 0)
        self.bip = gesamtkonsum + investitionen + staatsausgaben
        
        # === 8. INFLATION ===
        preis_level_neu = avg_preis
        if self.preis_level_alt > 0:
            self.inflation = (preis_level_neu - self.preis_level_alt) / self.preis_level_alt
        self.preis_level_alt = preis_level_neu
        
        # === 9. REWARDS BERECHNEN ===
        markt_info = self._get_markt_info()
        
        # Reward für jede Firma
        firm_rewards = np.array([firm.calculate_reward(markt_info) for firm in self.firms])
        
        # Reward für Regierung
        arbeitslose = sum(1 for h in self.households if not h.arbeitet)
        arbeitslosenquote = arbeitslose / self.n_households
        
        gov_reward = (
            (self.bip - 50000) / 500 +  # BIP-Wachstum
            -abs(self.inflation - 0.02) * 500 +  # Ziel-Inflation 2%
            -arbeitslosenquote * 100 +  # Arbeitslosigkeit
            -self.staatsschulden * 0.001  # Schulden
        )
        
        rewards = {
            'firms': firm_rewards,
            'government': gov_reward
        }
        
        # === 10. TERMINATION ===
        self.timestep += 1
        terminated = self.timestep >= 200 or self.staatsschulden > 100000
        
        # Neue Observations
        firm_obs = self._get_firm_observations()
        gov_obs = self._get_gov_observation()
        
        observations = {'firms': firm_obs, 'government': gov_obs}
        
        return observations, rewards, terminated, False, {}
    
    def render(self):
        """Ausgabe für Monitoring"""
        print(f"\n{'='*60}")
        print(f"Step {self.timestep}")
        print(f"{'='*60}")
        print(f"BIP: {self.bip:.0f}")
        print(f"Inflation: {self.inflation*100:.1f}%")
        print(f"Staatsschulden: {self.staatsschulden:.0f}")
        print(f"\nFirmen:")
        for firm in self.firms:
            state = firm.get_state()
            print(f"  Firma {state['id']}: "
                  f"Preis={state['preis']:.0f}, "
                  f"Gewinn={state['gewinn']:.0f}, "
                  f"Markt={state['marktanteil']*100:.1f}%")

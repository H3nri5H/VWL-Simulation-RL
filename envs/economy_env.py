"""PettingZoo Multi-Agent Environment für Volkswirtschaft"""

from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
import functools
from agents import Firm, Household

class EconomyEnv(ParallelEnv):
    """
    Multi-Agent Volkswirtschafts-Environment.
    Kompatibel mit RLlib und PettingZoo.
    
    Agents: firm_0, firm_1, ..., firm_N
    """
    
    metadata = {'render_modes': ['human'], 'name': 'economy_v1'}
    
    def __init__(self, n_firms=3, n_households=30, max_steps=100):
        super().__init__()
        
        self.n_firms = n_firms
        self.n_households = n_households
        self.max_steps = max_steps
        
        # Agent names
        self.possible_agents = [f"firm_{i}" for i in range(n_firms)]
        
        # Makro-Variablen
        self.bip = 0.0
        self.inflation = 0.02
        self.timestep = 0
        self.preis_level_alt = 100.0
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """Observation Space: 8 Features pro Firma"""
        return spaces.Box(low=0, high=10, shape=(8,), dtype=np.float32)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        Action Space: [preis_faktor, lohn_faktor, mitarbeiter_change]
        - preis_faktor: 0.8 - 1.2
        - lohn_faktor: 0.9 - 1.1
        - mitarbeiter_change: -5 bis +5
        """
        return spaces.Box(
            low=np.array([0.8, 0.9, -5]),
            high=np.array([1.2, 1.1, 5]),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Environment zurücksetzen"""
        if seed is not None:
            np.random.seed(seed)
        
        # Agents erstellen
        self.agents = self.possible_agents[:]
        self.firms = {agent: Firm(i) for i, agent in enumerate(self.agents)}
        self.households = [Household(i) for i in range(self.n_households)]
        
        # Zustand zurücksetzen
        self.timestep = 0
        self.bip = 0.0
        self.preis_level_alt = 100.0
        
        # Initial observations
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions):
        """
        Simulationsschritt:
        1. Firmen setzen Aktionen um
        2. Produktion
        3. Haushalte konsumieren
        4. Markt räumt sich
        5. Rewards berechnen
        """
        # 1. AKTIONEN ANWENDEN
        for agent in self.agents:
            self.firms[agent].apply_action(actions[agent])
        
        # 2. PRODUKTION
        for agent in self.agents:
            self.firms[agent].produzieren()
        
        gesamtproduktion = sum(f.produktion for f in self.firms.values())
        
        # 3. HAUSHALTE EINKOMMEN & KONSUM
        avg_lohn = np.mean([f.lohn for f in self.firms.values()])
        for household in self.households:
            household.receive_income(avg_lohn)
        
        gesamtkonsum = sum(h.konsumieren() for h in self.households)
        
        # 4. MARKT (Preiswettbewerb)
        avg_preis = np.mean([f.preis for f in self.firms.values()])
        nachfrage_einheiten = gesamtkonsum / avg_preis if avg_preis > 0 else 0
        
        # Marktanteile basierend auf Preiswettbewerb
        marktanteile = self._calculate_market_shares(nachfrage_einheiten, gesamtproduktion)
        
        # Firmen verkaufen
        for i, agent in enumerate(self.agents):
            firm = self.firms[agent]
            firm.marktanteil = marktanteile[i]
            verkauft = min(firm.produktion, nachfrage_einheiten * firm.marktanteil)
            firm.update_finanzen(verkauft)
        
        # 5. MAKROÖKONOMIE
        self.bip = gesamtkonsum
        self.inflation = (avg_preis - self.preis_level_alt) / self.preis_level_alt if self.preis_level_alt > 0 else 0
        self.preis_level_alt = avg_preis
        
        # 6. REWARDS
        rewards = {agent: self.firms[agent].calculate_reward() for agent in self.agents}
        
        # 7. TERMINATION
        self.timestep += 1
        terminations = {agent: self.timestep >= self.max_steps for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        
        # 8. OBSERVATIONS
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def _calculate_market_shares(self, nachfrage, angebot):
        """
        Marktanteile basierend auf Preiswettbewerb:
        Günstigere Firmen bekommen höheren Marktanteil
        """
        if angebot == 0:
            return [1/self.n_firms] * self.n_firms
        
        avg_preis = np.mean([f.preis for f in self.firms.values()])
        
        # Wettbewerbsfaktoren berechnen
        wettbewerb_faktoren = []
        for agent in self.agents:
            firm = self.firms[agent]
            # Je günstiger, desto höher der Faktor
            preis_faktor = (avg_preis / firm.preis) ** 1.5
            # Gewichtet mit Produktionskapazität
            kapazitaet = firm.produktion / angebot if angebot > 0 else 0
            wettbewerb_faktoren.append(preis_faktor * kapazitaet)
        
        # Normalisieren
        summe = sum(wettbewerb_faktoren)
        if summe > 0:
            return [f / summe for f in wettbewerb_faktoren]
        else:
            return [1/self.n_firms] * self.n_firms
    
    def _get_observations(self):
        """Observations für alle Agents"""
        markt_info = self._get_market_info()
        observations = {}
        for agent in self.agents:
            observations[agent] = self.firms[agent].get_observation(markt_info)
        return observations
    
    def _get_market_info(self):
        """Marktinformationen"""
        firms = list(self.firms.values())
        return {
            'avg_preis': np.mean([f.preis for f in firms]),
            'nachfrage': sum(h.vermoegen * 0.8 for h in self.households),
            'angebot': sum(f.produktion for f in firms)
        }
    
    def render(self):
        """Visualisierung"""
        print(f"\n{'='*60}")
        print(f"Timestep {self.timestep} | BIP: {self.bip:.0f} | Inflation: {self.inflation*100:.1f}%")
        print(f"{'='*60}")
        for agent in self.agents:
            state = self.firms[agent].get_state()
            print(f"{agent}: Preis={state['preis']}, Gewinn={state['gewinn']}, Markt={state['marktanteil']}%")
    
    def close(self):
        pass

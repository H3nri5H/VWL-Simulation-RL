"""Economy Environment for Multi-Agent RL Volkswirtschaftssimulation."""

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector


class EconomyEnv(AECEnv):
    """Multi-Agent Volkswirtschaftssimulation.
    
    Agents:
    - Firmen: Entscheiden über Preise und Löhne
    - Haushalte: Regelbasiert (kaufen/arbeiten basierend auf Einkommen/Preisen)
    
    Observation Space:
    - Eigener Preis, Lohn, Nachfrage, Inventory, Profit
    - Marktdurchschnitte
    
    Action Space:
    - Preis-Anpassung (-10% bis +10%)
    - Lohn-Anpassung (-10% bis +10%)
    """
    
    metadata = {'render_modes': ['human'], 'name': 'economy_v0'}
    
    def __init__(self, n_firms=2, n_households=10):
        """Initialize Economy Environment.
        
        Args:
            n_firms: Anzahl lernender Firmen-Agents
            n_households: Anzahl regelbasierter Haushalte
        """
        super().__init__()
        
        self.n_firms = n_firms
        self.n_households = n_households
        
        # Agent-Namen
        self.possible_agents = [f"firm_{i}" for i in range(n_firms)]
        self.agents = self.possible_agents[:]
        
        # Observation Space: [eigener_preis, eigener_lohn, nachfrage, inventory, profit, markt_avg_preis, markt_avg_lohn]
        self.observation_spaces = {
            agent: Box(low=0.0, high=100.0, shape=(7,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        # Action Space: Diskret mit 9 Aktionen (Preis/Lohn: -10%, -5%, 0%, +5%, +10%)
        # Action Index: 0-4 für Preis, 5-8 für Lohn kombiniert -> Vereinfacht zu 9 Kombinationen
        self.action_spaces = {
            agent: Discrete(9)
            for agent in self.possible_agents
        }
        
        self._agent_selector = agent_selector(self.agents)
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
        # Initialize state (placeholder)
        self.state = {
            'firms': {},
            'households': {},
            'market': {},
            'timestep': 0
        }
        
        # TODO: Initialize firm and household states
        
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        return self.observe(self.agent_selection), self.infos[self.agent_selection]
    
    def step(self, action):
        """Execute one step of the environment."""
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)
        
        agent = self.agent_selection
        
        # TODO: Implement action logic (price/wage adjustment)
        # TODO: Implement market clearing
        # TODO: Calculate rewards
        
        self.rewards[agent] = 0.0  # Placeholder
        
        # Check termination conditions
        self.state['timestep'] += 1
        if self.state['timestep'] >= 100:  # Max 100 steps
            self.terminations = {a: True for a in self.agents}
        
        # Select next agent
        self.agent_selection = self._agent_selector.next()
        
    def observe(self, agent):
        """Return observation for given agent."""
        # TODO: Construct observation from state
        return np.zeros(7, dtype=np.float32)  # Placeholder
    
    def render(self):
        """Render environment (optional)."""
        pass

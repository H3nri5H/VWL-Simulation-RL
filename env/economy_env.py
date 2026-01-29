"""Economy Environment for Multi-Agent RL Volkswirtschaftssimulation."""

import numpy as np
from gymnasium.spaces import Box, MultiDiscrete
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
    - MultiDiscrete([5, 5]):
      - Dimension 0 (Preis): 0=-10%, 1=-5%, 2=0%, 3=+5%, 4=+10%
      - Dimension 1 (Lohn): 0=-10%, 1=-5%, 2=0%, 3=+5%, 4=+10%
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
        
        # Action Space: MultiDiscrete für separate Preis- und Lohn-Entscheidungen
        # [Preis-Aktion (0-4), Lohn-Aktion (0-4)]
        self.action_spaces = {
            agent: MultiDiscrete([5, 5])
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
        
        # Decode action to price and wage adjustments
        price_delta, wage_delta = self._decode_action(action)
        
        # TODO: Apply price_delta and wage_delta to firm state
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
    
    def _decode_action(self, action):
        """Konvertiert MultiDiscrete Action zu Preis- und Lohn-Änderungen.
        
        Args:
            action: np.array([price_action, wage_action]) mit Werten 0-4
            
        Returns:
            tuple: (price_delta, wage_delta) als Prozentsätze (z.B. -0.10 für -10%)
        """
        # Mapping: 0 → -10%, 1 → -5%, 2 → 0%, 3 → +5%, 4 → +10%
        changes = [-0.10, -0.05, 0.0, 0.05, 0.10]
        
        price_delta = changes[action[0]]
        wage_delta = changes[action[1]]
        
        return price_delta, wage_delta
    
    def render(self):
        """Render environment (optional)."""
        pass

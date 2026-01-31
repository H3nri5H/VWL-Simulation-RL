"""Economy Environment for Multi-Agent RL Volkswirtschaftssimulation."""

import numpy as np
from gymnasium.spaces import Box, MultiDiscrete
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector


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
    
    Supports both AEC API and Parallel API for RLlib compatibility.
    """
    
    metadata = {'render_modes': ['human'], 'name': 'economy_v0'}
    
    def __init__(self, config=None):
        """Initialize Economy Environment.
        
        Args:
            config: Dictionary or EnvContext with:
                - n_firms: Anzahl lernender Firmen-Agents (default: 2)
                - n_households: Anzahl regelbasierter Haushalte (default: 10)
                - max_steps: Maximale Anzahl Steps pro Episode (default: 100)
        """
        super().__init__()
        
        # Handle both dict and EnvContext (Ray passes EnvContext)
        if config is None:
            config = {}
        
        # Extract parameters from config
        self.n_firms = config.get('n_firms', 2) if hasattr(config, 'get') else getattr(config, 'n_firms', 2)
        self.n_households = config.get('n_households', 10) if hasattr(config, 'get') else getattr(config, 'n_households', 10)
        self.max_steps = config.get('max_steps', 100) if hasattr(config, 'get') else getattr(config, 'max_steps', 100)
        
        # Agent-Namen
        self.possible_agents = [f"firm_{i}" for i in range(self.n_firms)]
        self.agents = self.possible_agents[:]
        
        # Observation Space: [eigener_preis, eigener_lohn, nachfrage, inventory, profit, markt_avg_preis, markt_avg_lohn]
        self.observation_spaces = {
            agent: Box(low=0.0, high=1000.0, shape=(7,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        # Action Space: MultiDiscrete für separate Preis- und Lohn-Entscheidungen
        self.action_spaces = {
            agent: MultiDiscrete([5, 5])
            for agent in self.possible_agents
        }
        
        self._agent_selector = agent_selector(self.agents)
        self._last_rewards = {agent: 0.0 for agent in self.agents}
        
    def reset(self, seed=None, options=None):
        """Reset environment (supports both AEC and Parallel API).
        
        Returns:
            - AEC mode: (observation, info) for first agent
            - Parallel mode: (observations_dict, infos_dict) for all agents
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
        # Initialize firm states
        self.firms = {}
        for i in range(self.n_firms):
            self.firms[f"firm_{i}"] = {
                'price': 10.0 + np.random.uniform(-2, 2),  # Startpreis um 10€
                'wage': 8.0 + np.random.uniform(-1, 1),     # Startlohn um 8€
                'inventory': 100.0,                         # Anfangsbestand
                'capital': 1000.0,                          # Startkapital
                'profit': 0.0,                              # Kein Profit zu Beginn
                'demand': 0.0,                              # Nachfrage im letzten Step
                'employees': 0,                             # Anzahl Mitarbeiter
            }
        
        # Initialize household states
        self.households = []
        for i in range(self.n_households):
            self.households.append({
                'money': 100.0 + np.random.uniform(-20, 20),  # Startgeld
                'income': 0.0,                                 # Einkommen pro Step
                'employer': None,                              # Bei welcher Firma angestellt
            })
        
        # Market state
        self.state = {
            'timestep': 0,
            'total_gdp': 0.0,
            'avg_price': np.mean([f['price'] for f in self.firms.values()]),
            'avg_wage': np.mean([f['wage'] for f in self.firms.values()]),
        }
        
        # Buffer für Actions (alle Agents müssen agieren bevor Step ausgeführt wird)
        self._action_buffer = {}
        
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._last_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Check if parallel mode (options contains 'parallel' flag or called from wrapper)
        is_parallel = options and options.get('parallel', False) if options else False
        
        if is_parallel:
            # Parallel API: return observations and infos for ALL agents
            observations = {agent: self.observe(agent) for agent in self.agents}
            # CRITICAL FIX: Ensure infos is explicitly a dict of dicts
            infos = {agent: {} for agent in self.agents}
            return observations, infos
        else:
            # AEC API: return observation and info for first agent
            return self.observe(self.agent_selection), self.infos[self.agent_selection]
    
    def step(self, action):
        """Execute step (supports both AEC and Parallel API).
        
        Args:
            action: Either single action (AEC) or dict of actions (Parallel)
            
        Returns:
            - AEC mode: None (updates internal state)
            - Parallel mode: (observations, rewards, terminations, truncations, infos)
        """
        # Check if parallel mode (action is dict)
        if isinstance(action, dict):
            return self._parallel_step(action)
        else:
            return self._aec_step(action)
    
    def _aec_step(self, action):
        """AEC API step: one agent acts at a time."""
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)
        
        agent = self.agent_selection
        
        # Action speichern
        self._action_buffer[agent] = action
        
        # Wenn alle Agents ihre Action abgegeben haben: Market Step ausführen
        if len(self._action_buffer) == self.n_firms:
            self._execute_market_step()
            self._action_buffer = {}  # Reset buffer
        
        # Select next agent
        self.agent_selection = self._agent_selector.next()
    
    def _parallel_step(self, actions):
        """Parallel API step: all agents act simultaneously.
        
        Args:
            actions: Dict[agent_id, action] - Actions for all agents
            
        Returns:
            observations: Dict[agent_id, observation]
            rewards: Dict[agent_id, float]
            terminations: Dict[agent_id, bool]
            truncations: Dict[agent_id, bool]
            infos: Dict[agent_id, dict]
        """
        # Store all actions
        for agent, act in actions.items():
            self._action_buffer[agent] = act
        
        # Execute market step with all actions
        self._execute_market_step()
        self._action_buffer = {}  # Reset buffer
        
        # Build observations for all agents
        observations = {
            agent: self.observe(agent) 
            for agent in self.agents
        }
        
        # Build rewards (from last market step)
        rewards = {agent: self._last_rewards.get(agent, 0.0) for agent in self.agents}
        
        # Terminations and truncations
        terminations = {agent: self.terminations.get(agent, False) for agent in self.agents}
        truncations = {agent: self.truncations.get(agent, False) for agent in self.agents}
        
        # CRITICAL FIX: Infos MUST be dict of dicts! 
        # Explicitly create new dict to ensure proper structure
        infos = {}
        for agent in self.agents:
            # Get info dict for this agent, default to empty dict if not exists
            agent_info = self.infos.get(agent, {})
            # Ensure it's a dict (not tuple or other type)
            if not isinstance(agent_info, dict):
                agent_info = {}
            infos[agent] = agent_info
        
        return observations, rewards, terminations, truncations, infos
    
    def _execute_market_step(self):
        """Führt einen kompletten Markt-Step aus: Preise/Löhne anpassen, Haushalte agieren, Rewards berechnen."""
        
        # 1. Firmen passen Preise und Löhne an
        for agent_name, action in self._action_buffer.items():
            price_delta, wage_delta = self._decode_action(action)
            firm = self.firms[agent_name]
            
            # Preisanpassung (min 1€, max 100€)
            firm['price'] = np.clip(firm['price'] * (1 + price_delta), 1.0, 100.0)
            
            # Lohnanpassung (min 1€, max 50€)
            firm['wage'] = np.clip(firm['wage'] * (1 + wage_delta), 1.0, 50.0)
        
        # 2. Haushalte wählen Arbeitgeber (höchster Lohn)
        firm_wages = {name: firm['wage'] for name, firm in self.firms.items()}
        
        for household in self.households:
            # Wähle Firma mit höchstem Lohn
            best_firm = max(firm_wages, key=firm_wages.get)
            household['employer'] = best_firm
            household['income'] = firm_wages[best_firm]
            household['money'] += household['income']  # Lohn erhalten
            
            # Mitarbeiterzähler erhöhen
            self.firms[best_firm]['employees'] += 1
        
        # 3. Haushalte kaufen Güter (billigste Firma zuerst)
        firm_prices = {name: firm['price'] for name, firm in self.firms.items()}
        sorted_firms = sorted(firm_prices, key=firm_prices.get)  # Sortiert nach Preis
        
        for household in self.households:
            # Haushalt gibt 80% seines Geldes aus
            budget = household['money'] * 0.8
            
            for firm_name in sorted_firms:
                firm = self.firms[firm_name]
                price = firm['price']
                
                # Wie viel kann der Haushalt kaufen?
                quantity = min(budget / price, firm['inventory'])
                
                if quantity > 0:
                    # Kauf durchführen
                    household['money'] -= quantity * price
                    firm['inventory'] -= quantity
                    firm['demand'] += quantity
                    budget -= quantity * price
                
                if budget < price:  # Kein Geld mehr
                    break
        
        # 4. Firmen berechnen Profit und produzieren neue Güter
        total_gdp = 0.0
        
        for firm_name, firm in self.firms.items():
            # Umsatz = verkaufte Menge * Preis
            revenue = firm['demand'] * firm['price']
            
            # Kosten = Lohn * Anzahl Mitarbeiter
            labor_cost = firm['wage'] * firm['employees']
            
            # Profit
            profit = revenue - labor_cost
            firm['profit'] = profit
            firm['capital'] += profit
            
            # Produktion: Jeder Mitarbeiter produziert 10 Einheiten
            production = firm['employees'] * 10.0
            firm['inventory'] += production
            
            # Inventory nicht über 500 (Lagerbeschränkung)
            firm['inventory'] = min(firm['inventory'], 500.0)
            
            # GDP beitragen
            total_gdp += revenue
            
            # Mitarbeiter zurücksetzen für nächsten Step
            firm['employees'] = 0
            firm['demand'] = 0.0
        
        # 5. Rewards berechnen (normalisierter Profit)
        profits = [firm['profit'] for firm in self.firms.values()]
        mean_profit = np.mean(profits)
        std_profit = np.std(profits) + 1e-8  # Avoid division by zero
        
        for agent_name, firm in self.firms.items():
            # Normalisierter Reward
            normalized_reward = (firm['profit'] - mean_profit) / std_profit
            self.rewards[agent_name] = normalized_reward
            self._last_rewards[agent_name] = normalized_reward  # Store for parallel API
        
        # 6. State aktualisieren
        self.state['timestep'] += 1
        self.state['total_gdp'] = total_gdp
        self.state['avg_price'] = np.mean([f['price'] for f in self.firms.values()])
        self.state['avg_wage'] = np.mean([f['wage'] for f in self.firms.values()])
        
        # 7. Termination prüfen
        if self.state['timestep'] >= self.max_steps:
            self.terminations = {a: True for a in self.agents}
            self.truncations = {a: True for a in self.agents}
        
        # 8. Infos - CRITICAL: Must be dict of dicts, not tuple!
        # Explicitly create new dict structure
        self.infos = {}
        for agent_name in self.agents:
            self.infos[agent_name] = {
                'profit': float(self.firms[agent_name]['profit']),
                'price': float(self.firms[agent_name]['price']),
                'wage': float(self.firms[agent_name]['wage']),
                'inventory': float(self.firms[agent_name]['inventory']),
                'timestep': int(self.state['timestep']),
            }
    
    def observe(self, agent):
        """Return observation for given agent."""
        firm = self.firms[agent]
        
        obs = np.array([
            firm['price'],
            firm['wage'],
            firm['demand'],
            firm['inventory'],
            firm['profit'],
            self.state['avg_price'],
            self.state['avg_wage'],
        ], dtype=np.float32)
        
        return obs
    
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
        print(f"\n=== Step {self.state['timestep']} ===")
        print(f"GDP: {self.state['total_gdp']:.2f}€")
        print(f"Avg Price: {self.state['avg_price']:.2f}€")
        print(f"Avg Wage: {self.state['avg_wage']:.2f}€")
        
        for name, firm in self.firms.items():
            print(f"  {name}: Price={firm['price']:.2f}€, Wage={firm['wage']:.2f}€, "
                  f"Profit={firm['profit']:.2f}€, Inventory={firm['inventory']:.0f}")
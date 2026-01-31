"""Simple Gymnasium-based Economy Environment for RLlib MultiAgent (Ray 2.40+)."""

import numpy as np
from gymnasium.spaces import Box, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class SimpleEconomyEnv(MultiAgentEnv):
    """Simple Multi-Agent Economy Environment (Ray 2.40 compatible).
    
    This environment is fully compatible with Ray 2.40's new API stack.
    """
    
    def __init__(self, config=None):
        """Initialize environment.
        
        Args:
            config: Dict with n_firms, n_households, max_steps
        """
        super().__init__()
        
        if config is None:
            config = {}
        
        self.n_firms = config.get('n_firms', 2)
        self.n_households = config.get('n_households', 10)
        self.max_steps = config.get('max_steps', 100)
        
        # Agent IDs (required by new API)
        self._agent_ids = [f"firm_{i}" for i in range(self.n_firms)]
        self._possible_agents = self._agent_ids.copy()
        
        # Spaces (same for all agents)
        self._obs_space = Box(low=0.0, high=1000.0, shape=(7,), dtype=np.float32)
        self._action_space = MultiDiscrete([5, 5])
        
        self.reset()
    
    @property
    def agents(self):
        """List of active agent IDs (required by Ray 2.40)."""
        return self._agent_ids
    
    @property
    def possible_agents(self):
        """List of all possible agent IDs (required by Ray 2.40)."""
        return self._possible_agents
    
    def observation_space(self, agent_id):
        """Return observation space for a specific agent."""
        return self._obs_space
    
    def action_space(self, agent_id):
        """Return action space for a specific agent."""
        return self._action_space
    
    def reset(self, *, seed=None, options=None):
        """Reset environment.
        
        Returns:
            observations: Dict[agent_id, obs]
            infos: Dict[agent_id, info]
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize firms
        self.firms = {}
        for i in range(self.n_firms):
            self.firms[f"firm_{i}"] = {
                'price': 10.0 + np.random.uniform(-2, 2),
                'wage': 8.0 + np.random.uniform(-1, 1),
                'inventory': 100.0,
                'capital': 1000.0,
                'profit': 0.0,
                'demand': 0.0,
                'employees': 0,
            }
        
        # Initialize households
        self.households = []
        for _ in range(self.n_households):
            self.households.append({
                'money': 100.0 + np.random.uniform(-20, 20),
                'income': 0.0,
                'employer': None,
            })
        
        # Market state
        self.timestep = 0
        self.total_gdp = 0.0
        
        # Build observations
        observations = {
            agent_id: self._get_observation(agent_id)
            for agent_id in self._agent_ids
        }
        
        infos = {agent_id: {} for agent_id in self._agent_ids}
        
        return observations, infos
    
    def step(self, action_dict):
        """Execute one step.
        
        Args:
            action_dict: Dict[agent_id, action]
            
        Returns:
            observations: Dict[agent_id, obs]
            rewards: Dict[agent_id, float]
            terminateds: Dict[agent_id, bool]
            truncateds: Dict[agent_id, bool]
            infos: Dict[agent_id, dict]
        """
        # 1. Apply price/wage changes
        for agent_id, action in action_dict.items():
            price_delta, wage_delta = self._decode_action(action)
            firm = self.firms[agent_id]
            
            firm['price'] = np.clip(firm['price'] * (1 + price_delta), 1.0, 100.0)
            firm['wage'] = np.clip(firm['wage'] * (1 + wage_delta), 1.0, 50.0)
        
        # 2. Households choose employers (highest wage)
        firm_wages = {name: firm['wage'] for name, firm in self.firms.items()}
        
        for household in self.households:
            best_firm = max(firm_wages, key=firm_wages.get)
            household['employer'] = best_firm
            household['income'] = firm_wages[best_firm]
            household['money'] += household['income']
            self.firms[best_firm]['employees'] += 1
        
        # 3. Households buy goods (cheapest first)
        firm_prices = {name: firm['price'] for name, firm in self.firms.items()}
        sorted_firms = sorted(firm_prices, key=firm_prices.get)
        
        for household in self.households:
            budget = household['money'] * 0.8
            
            for firm_name in sorted_firms:
                firm = self.firms[firm_name]
                price = firm['price']
                
                quantity = min(budget / price, firm['inventory'])
                
                if quantity > 0:
                    household['money'] -= quantity * price
                    firm['inventory'] -= quantity
                    firm['demand'] += quantity
                    budget -= quantity * price
                
                if budget < price:
                    break
        
        # 4. Calculate profits and production
        total_gdp = 0.0
        
        for firm_name, firm in self.firms.items():
            revenue = firm['demand'] * firm['price']
            labor_cost = firm['wage'] * firm['employees']
            
            profit = revenue - labor_cost
            firm['profit'] = profit
            firm['capital'] += profit
            
            production = firm['employees'] * 10.0
            firm['inventory'] = min(firm['inventory'] + production, 500.0)
            
            total_gdp += revenue
            
            firm['employees'] = 0
            firm['demand'] = 0.0
        
        # 5. Calculate rewards (normalized profit)
        profits = [firm['profit'] for firm in self.firms.values()]
        mean_profit = np.mean(profits)
        std_profit = np.std(profits) + 1e-8
        
        rewards = {}
        for agent_id, firm in self.firms.items():
            rewards[agent_id] = (firm['profit'] - mean_profit) / std_profit
        
        # 6. Update state
        self.timestep += 1
        self.total_gdp = total_gdp
        
        # 7. Check termination
        done = self.timestep >= self.max_steps
        terminateds = {agent_id: done for agent_id in self._agent_ids}
        truncateds = {agent_id: done for agent_id in self._agent_ids}
        terminateds['__all__'] = done
        truncateds['__all__'] = done
        
        # 8. Build observations and infos
        observations = {
            agent_id: self._get_observation(agent_id)
            for agent_id in self._agent_ids
        }
        
        infos = {
            agent_id: {
                'profit': float(self.firms[agent_id]['profit']),
                'price': float(self.firms[agent_id]['price']),
                'wage': float(self.firms[agent_id]['wage']),
                'timestep': self.timestep,
            }
            for agent_id in self._agent_ids
        }
        
        return observations, rewards, terminateds, truncateds, infos
    
    def _get_observation(self, agent_id):
        """Get observation for agent."""
        firm = self.firms[agent_id]
        
        avg_price = np.mean([f['price'] for f in self.firms.values()])
        avg_wage = np.mean([f['wage'] for f in self.firms.values()])
        
        obs = np.array([
            firm['price'],
            firm['wage'],
            firm['demand'],
            firm['inventory'],
            firm['profit'],
            avg_price,
            avg_wage,
        ], dtype=np.float32)
        
        return obs
    
    def _decode_action(self, action):
        """Convert action to price/wage deltas."""
        changes = [-0.10, -0.05, 0.0, 0.05, 0.10]
        price_delta = changes[action[0]]
        wage_delta = changes[action[1]]
        return price_delta, wage_delta

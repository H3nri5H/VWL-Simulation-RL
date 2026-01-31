import numpy as np
from gymnasium.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class SimpleEconomyEnv(MultiAgentEnv):
    
    def __init__(self, config=None):
        super().__init__()
        
        config = config or {}
        self.n_firms = config.get('n_firms', 2)
        self.n_households = config.get('n_households', 10)
        self.max_steps = config.get('max_steps', 100)
        
        self._agent_ids = set(f"firm_{i}" for i in range(self.n_firms))
        
        self._obs_space = Box(low=0.0, high=200.0, shape=(6,), dtype=np.float32)
        self._action_space = Discrete(5)
        
        self.reset()
    
    @property
    def observation_space(self):
        return self._obs_space
    
    @property
    def action_space(self):
        return self._action_space
    
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.firms = {}
        for i in range(self.n_firms):
            self.firms[f"firm_{i}"] = {
                'price': 10.0,
                'wage': 8.0,
                'profit': 0.0,
            }
        
        self.households = [{'money': 100.0} for _ in range(self.n_households)]
        self.timestep = 0
        
        obs = {agent_id: self._get_obs(agent_id) for agent_id in self._agent_ids}
        return obs, {agent_id: {} for agent_id in self._agent_ids}
    
    def step(self, action_dict):
        for agent_id, action in action_dict.items():
            delta = (action - 2) * 0.05
            self.firms[agent_id]['price'] = np.clip(
                self.firms[agent_id]['price'] * (1 + delta), 1.0, 50.0
            )
        
        total_demand = {agent_id: 0.0 for agent_id in self._agent_ids}
        
        for household in self.households:
            budget = household['money'] * 0.8
            prices = {aid: self.firms[aid]['price'] for aid in self._agent_ids}
            cheapest = min(prices, key=prices.get)
            
            quantity = budget / prices[cheapest]
            total_demand[cheapest] += quantity
            household['money'] -= quantity * prices[cheapest]
        
        for household in self.households:
            household['money'] += 10.0
        
        rewards = {}
        for agent_id in self._agent_ids:
            revenue = total_demand[agent_id] * self.firms[agent_id]['price']
            self.firms[agent_id]['profit'] = revenue
            rewards[agent_id] = revenue / 100.0
        
        self.timestep += 1
        done = self.timestep >= self.max_steps
        
        obs = {agent_id: self._get_obs(agent_id) for agent_id in self._agent_ids}
        dones = {agent_id: done for agent_id in self._agent_ids}
        dones['__all__'] = done
        
        infos = {agent_id: {'profit': self.firms[agent_id]['profit']} for agent_id in self._agent_ids}
        
        return obs, rewards, dones, dones, infos
    
    def _get_obs(self, agent_id):
        firm = self.firms[agent_id]
        other_prices = [self.firms[aid]['price'] for aid in self._agent_ids if aid != agent_id]
        avg_other_price = np.mean(other_prices) if other_prices else firm['price']
        
        return np.array([
            firm['price'],
            firm['profit'],
            avg_other_price,
            self.timestep / self.max_steps,
            len(self.households),
            0.0,
        ], dtype=np.float32)

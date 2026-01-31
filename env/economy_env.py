import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class SimpleEconomyEnv(MultiAgentEnv):
    
    def __init__(self, config=None):
        super().__init__()
        
        config = config or {}
        self.n_firms = config.get('n_firms', 2)
        self.n_households = config.get('n_households', 10)
        self.max_steps = config.get('max_steps', 100)
        
        self._agent_ids = set(f"firm_{i}" for i in range(self.n_firms))
        
        # Observation: [own_price, own_wage, own_employees, own_profit, avg_other_price, avg_other_wage, timestep]
        self._obs_space = Box(low=-100.0, high=100.0, shape=(7,), dtype=np.float32)
        
        # Action: [price_change, wage_change]
        # Each can be: 0=decrease, 1=keep, 2=increase
        self._action_space = MultiDiscrete([3, 3])
        
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
            # Firms start with random values
            self.firms[f"firm_{i}"] = {
                'price': np.random.uniform(8.0, 15.0),
                'wage': np.random.uniform(5.0, 12.0),
                'max_employees': np.random.randint(3, 8),  # Limited capacity
                'employees': 0,
                'profit': 0.0,
                'revenue': 0.0,
                'costs': 0.0,
            }
        
        self.households = []
        for _ in range(self.n_households):
            self.households.append({
                'money': 50.0,
                'employer': None,
                'wage': 0.0,
            })
        
        self.timestep = 0
        
        obs = {agent_id: self._get_obs(agent_id) for agent_id in self._agent_ids}
        return obs, {agent_id: {} for agent_id in self._agent_ids}
    
    def step(self, action_dict):
        # Phase 1: Firms adjust price and wage based on actions
        for agent_id, action in action_dict.items():
            price_action = action[0]  # 0=decrease, 1=keep, 2=increase
            wage_action = action[1]
            
            # Price adjustment
            if price_action == 0:
                self.firms[agent_id]['price'] *= 0.95  # -5%
            elif price_action == 2:
                self.firms[agent_id]['price'] *= 1.05  # +5%
            
            # Wage adjustment
            if wage_action == 0:
                self.firms[agent_id]['wage'] *= 0.95  # -5%
            elif wage_action == 2:
                self.firms[agent_id]['wage'] *= 1.05  # +5%
            
            # Clip to reasonable bounds
            self.firms[agent_id]['price'] = np.clip(self.firms[agent_id]['price'], 1.0, 50.0)
            self.firms[agent_id]['wage'] = np.clip(self.firms[agent_id]['wage'], 1.0, 20.0)
            
            # Reset employee count for this step
            self.firms[agent_id]['employees'] = 0
        
        # Phase 2: Households choose employer (highest wage with capacity)
        # Reset all employment
        for household in self.households:
            household['employer'] = None
            household['wage'] = 0.0
        
        # Sort firms by wage (highest first)
        firms_by_wage = sorted(
            self._agent_ids,
            key=lambda aid: self.firms[aid]['wage'],
            reverse=True
        )
        
        # Households apply to highest wage firms
        for household in self.households:
            for firm_id in firms_by_wage:
                firm = self.firms[firm_id]
                if firm['employees'] < firm['max_employees']:
                    household['employer'] = firm_id
                    household['wage'] = firm['wage']
                    firm['employees'] += 1
                    household['money'] += firm['wage']  # Get paid
                    break
        
        # Phase 3: Households shop (buy from cheapest firm)
        total_demand = {agent_id: 0.0 for agent_id in self._agent_ids}
        
        for household in self.households:
            if household['money'] <= 0:
                continue
            
            budget = household['money'] * 0.6  # Spend 60% of money
            prices = {aid: self.firms[aid]['price'] for aid in self._agent_ids}
            cheapest = min(prices, key=prices.get)
            
            quantity = budget / prices[cheapest]
            total_demand[cheapest] += quantity
            household['money'] -= quantity * prices[cheapest]
        
        # Phase 4: Calculate firm profits
        rewards = {}
        for agent_id in self._agent_ids:
            firm = self.firms[agent_id]
            
            # Revenue from sales
            revenue = total_demand[agent_id] * firm['price']
            
            # Costs from wages
            wage_costs = firm['employees'] * firm['wage']
            
            # Profit = Revenue - Costs
            profit = revenue - wage_costs
            
            firm['revenue'] = revenue
            firm['costs'] = wage_costs
            firm['profit'] = profit
            
            # Reward is profit (can be negative!)
            rewards[agent_id] = profit / 10.0  # Scale down for training
        
        self.timestep += 1
        done = self.timestep >= self.max_steps
        
        obs = {agent_id: self._get_obs(agent_id) for agent_id in self._agent_ids}
        dones = {agent_id: done for agent_id in self._agent_ids}
        dones['__all__'] = done
        
        infos = {
            agent_id: {
                'profit': self.firms[agent_id]['profit'],
                'revenue': self.firms[agent_id]['revenue'],
                'costs': self.firms[agent_id]['costs'],
                'employees': self.firms[agent_id]['employees'],
            } 
            for agent_id in self._agent_ids
        }
        
        return obs, rewards, dones, dones, infos
    
    def _get_obs(self, agent_id):
        firm = self.firms[agent_id]
        
        # Calculate averages of other firms
        other_firms = [self.firms[aid] for aid in self._agent_ids if aid != agent_id]
        
        if other_firms:
            avg_other_price = np.mean([f['price'] for f in other_firms])
            avg_other_wage = np.mean([f['wage'] for f in other_firms])
        else:
            avg_other_price = firm['price']
            avg_other_wage = firm['wage']
        
        obs = np.array([
            firm['price'],
            firm['wage'],
            firm['employees'],
            firm['profit'] / 10.0,  # Normalize profit
            avg_other_price,
            avg_other_wage,
            self.timestep / self.max_steps,
        ], dtype=np.float32)
        
        # Safety clip to observation space bounds
        return np.clip(obs, -100.0, 100.0)

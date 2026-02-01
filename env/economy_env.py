import numpy as np
import yaml
from pathlib import Path
from gymnasium.spaces import Box, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class SimpleEconomyEnv(MultiAgentEnv):
    
    def __init__(self, config=None):
        super().__init__()
        
        # Load default config from YAML
        config_path = Path(__file__).parent.parent / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                default_config = yaml.safe_load(f)
        else:
            default_config = {}
        
        # Merge with provided config (provided config takes precedence)
        config = config or {}
        
        # Environment parameters
        env_cfg = default_config.get('environment', {})
        self.n_firms = config.get('n_firms', env_cfg.get('n_firms', 2))
        self.n_households = config.get('n_households', env_cfg.get('n_households', 10))
        self.max_steps = config.get('max_steps', env_cfg.get('max_steps', 100))
        
        # Initial ranges
        init_ranges = default_config.get('initial_ranges', {})
        self.init_ranges = {
            'firms': init_ranges.get('firms', {}),
            'households': init_ranges.get('households', {})
        }
        
        # Economic parameters
        econ_cfg = default_config.get('economy', {})
        self.price_adjustment = econ_cfg.get('price_adjustment', 0.05)
        self.wage_adjustment = econ_cfg.get('wage_adjustment', 0.05)
        self.consumption_rate = econ_cfg.get('consumption_rate', 0.6)
        self.reward_scale = econ_cfg.get('reward_scale', 10.0)
        
        price_bounds = econ_cfg.get('price_bounds', {'min': 1.0, 'max': 50.0})
        wage_bounds = econ_cfg.get('wage_bounds', {'min': 1.0, 'max': 20.0})
        self.price_min = price_bounds['min']
        self.price_max = price_bounds['max']
        self.wage_min = wage_bounds['min']
        self.wage_max = wage_bounds['max']
        
        self._agent_ids = set(f"firm_{i}" for i in range(self.n_firms))
        
        # Improved Observation Space:
        # [own_price, own_wage, own_employees, own_profit,
        #  market_avg_price, market_min_price, market_max_price,
        #  market_avg_wage, market_min_wage, market_max_wage,
        #  total_employed, unemployment_rate, timestep]
        self._obs_space = Box(low=-100.0, high=100.0, shape=(13,), dtype=np.float32)
        
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
        
        # Get initial ranges
        firm_ranges = self.init_ranges.get('firms', {})
        hh_ranges = self.init_ranges.get('households', {})
        
        price_range = firm_ranges.get('price', {'min': 8.0, 'max': 15.0})
        wage_range = firm_ranges.get('wage', {'min': 5.0, 'max': 12.0})
        emp_range = firm_ranges.get('max_employees', {'min': 3, 'max': 8})
        money_range = hh_ranges.get('money', {'min': 40.0, 'max': 60.0})
        
        self.firms = {}
        for i in range(self.n_firms):
            self.firms[f"firm_{i}"] = {
                'price': np.random.uniform(price_range['min'], price_range['max']),
                'wage': np.random.uniform(wage_range['min'], wage_range['max']),
                'max_employees': np.random.randint(emp_range['min'], emp_range['max'] + 1),
                'employees': 0,
                'profit': 0.0,
                'revenue': 0.0,
                'costs': 0.0,
            }
        
        self.households = []
        for _ in range(self.n_households):
            self.households.append({
                'money': np.random.uniform(money_range['min'], money_range['max']),
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
                self.firms[agent_id]['price'] *= (1 - self.price_adjustment)
            elif price_action == 2:
                self.firms[agent_id]['price'] *= (1 + self.price_adjustment)
            
            # Wage adjustment
            if wage_action == 0:
                self.firms[agent_id]['wage'] *= (1 - self.wage_adjustment)
            elif wage_action == 2:
                self.firms[agent_id]['wage'] *= (1 + self.wage_adjustment)
            
            # Clip to bounds
            self.firms[agent_id]['price'] = np.clip(
                self.firms[agent_id]['price'], 
                self.price_min, 
                self.price_max
            )
            self.firms[agent_id]['wage'] = np.clip(
                self.firms[agent_id]['wage'], 
                self.wage_min, 
                self.wage_max
            )
            
            # Reset employee count for this step
            self.firms[agent_id]['employees'] = 0
        
        # Phase 2: Labor market - Households choose employer
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
        
        # Phase 3: Goods market - Households shop
        total_demand = {agent_id: 0.0 for agent_id in self._agent_ids}
        
        for household in self.households:
            if household['money'] <= 0:
                continue
            
            budget = household['money'] * self.consumption_rate
            prices = {aid: self.firms[aid]['price'] for aid in self._agent_ids}
            cheapest = min(prices, key=prices.get)
            
            quantity = budget / prices[cheapest]
            total_demand[cheapest] += quantity
            household['money'] -= quantity * prices[cheapest]
        
        # Phase 4: Calculate firm profits and rewards
        rewards = {}
        for agent_id in self._agent_ids:
            firm = self.firms[agent_id]
            
            revenue = total_demand[agent_id] * firm['price']
            wage_costs = firm['employees'] * firm['wage']
            profit = revenue - wage_costs
            
            firm['revenue'] = revenue
            firm['costs'] = wage_costs
            firm['profit'] = profit
            
            # Reward is scaled profit
            rewards[agent_id] = profit / self.reward_scale
        
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
        """Create observation with aggregated market data"""
        firm = self.firms[agent_id]
        
        # Collect all firm data
        all_prices = [f['price'] for f in self.firms.values()]
        all_wages = [f['wage'] for f in self.firms.values()]
        
        # Market aggregates
        market_avg_price = np.mean(all_prices)
        market_min_price = np.min(all_prices)
        market_max_price = np.max(all_prices)
        
        market_avg_wage = np.mean(all_wages)
        market_min_wage = np.min(all_wages)
        market_max_wage = np.max(all_wages)
        
        # Employment statistics
        total_employed = sum(f['employees'] for f in self.firms.values())
        unemployment_rate = 1.0 - (total_employed / self.n_households)
        
        obs = np.array([
            firm['price'],
            firm['wage'],
            firm['employees'],
            firm['profit'] / self.reward_scale,
            market_avg_price,
            market_min_price,
            market_max_price,
            market_avg_wage,
            market_min_wage,
            market_max_wage,
            total_employed,
            unemployment_rate,
            self.timestep / self.max_steps,
        ], dtype=np.float32)
        
        return np.clip(obs, -100.0, 100.0)

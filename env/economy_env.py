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
        
        # Adjustment rates for each action level
        self.adjustment_rates = {
            0: -0.10,  # -10%
            1: -0.05,  # -5%
            2: 0.00,   # 0% (no change)
            3: 0.05,   # +5%
            4: 0.10,   # +10%
        }
        
        # Production parameters
        prod_cfg = econ_cfg.get('production', {})
        self.productivity_per_employee = prod_cfg.get('productivity_per_employee', 10.0)
        self.production_cost_per_unit = prod_cfg.get('cost_per_unit', 2.0)
        self.fixed_costs = prod_cfg.get('fixed_costs', 15.0)
        self.storage_cost_per_unit = prod_cfg.get('storage_cost_per_unit', 0.5)
        
        # Household parameters
        hh_cfg = econ_cfg.get('households', {})
        self.consumption_rate = hh_cfg.get('consumption_rate', 0.6)
        self.utility_price_weight = hh_cfg.get('utility_price_weight', 1.0)
        self.utility_quality_weight = hh_cfg.get('utility_quality_weight', 0.5)
        
        # Reward parameters
        self.reward_scale = econ_cfg.get('reward_scale', 10.0)
        
        # Bounds
        price_bounds = econ_cfg.get('price_bounds', {'min': 1.0, 'max': 50.0})
        wage_bounds = econ_cfg.get('wage_bounds', {'min': 1.0, 'max': 20.0})
        self.price_min = price_bounds['min']
        self.price_max = price_bounds['max']
        self.wage_min = wage_bounds['min']
        self.wage_max = wage_bounds['max']
        
        self._agent_ids = set(f"firm_{i}" for i in range(self.n_firms))
        
        # Expanded Observation Space:
        # [own_price, own_wage, own_employees, own_inventory, own_profit,
        #  market_avg_price, market_min_price, market_max_price,
        #  market_avg_wage, market_min_wage, market_max_wage,
        #  total_employed, unemployment_rate, 
        #  avg_household_money, timestep]
        self._obs_space = Box(low=-100.0, high=100.0, shape=(15,), dtype=np.float32)
        
        # Action: [price_change, wage_change]
        # Each can be: 0=-10%, 1=-5%, 2=0%, 3=+5%, 4=+10%
        self._action_space = MultiDiscrete([5, 5])
        
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
                'inventory': 0.0,  # NEW: Stock of unsold goods
                'production': 0.0,  # NEW: Production this step
                'profit': 0.0,
                'revenue': 0.0,
                'costs': 0.0,
                'quality': np.random.uniform(0.5, 1.0),  # NEW: Product quality (perceived)
            }
        
        self.households = []
        for _ in range(self.n_households):
            self.households.append({
                'money': np.random.uniform(money_range['min'], money_range['max']),
                'employer': None,
                'wage': 0.0,
                'wealth_type': np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2]),  # NEW: Household type
            })
        
        self.timestep = 0
        
        obs = {agent_id: self._get_obs(agent_id) for agent_id in self._agent_ids}
        return obs, {agent_id: {} for agent_id in self._agent_ids}
    
    def step(self, action_dict):
        # Phase 1: Firms adjust price and wage based on actions
        for agent_id, action in action_dict.items():
            price_action = action[0]  # 0=-10%, 1=-5%, 2=0%, 3=+5%, 4=+10%
            wage_action = action[1]
            
            # Get adjustment rates
            price_adjustment = self.adjustment_rates[price_action]
            wage_adjustment = self.adjustment_rates[wage_action]
            
            # Apply adjustments
            if price_adjustment != 0:
                self.firms[agent_id]['price'] *= (1 + price_adjustment)
            
            if wage_adjustment != 0:
                self.firms[agent_id]['wage'] *= (1 + wage_adjustment)
            
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
        
        # Phase 3: Production
        for agent_id in self._agent_ids:
            firm = self.firms[agent_id]
            
            # Production based on employees
            production = firm['employees'] * self.productivity_per_employee
            firm['production'] = production
            
            # Add to inventory
            firm['inventory'] += production
        
        # Phase 4: Goods market - Households shop with utility maximization
        total_demand = {agent_id: 0.0 for agent_id in self._agent_ids}
        
        for household in self.households:
            if household['money'] <= 0:
                continue
            
            budget = household['money'] * self.consumption_rate
            
            # Adjust budget based on wealth type
            if household['wealth_type'] == 'low':
                budget *= 0.8  # Spend less
            elif household['wealth_type'] == 'high':
                budget *= 1.2  # Spend more
            
            # Calculate utility for each firm's product
            utilities = {}
            for firm_id in self._agent_ids:
                firm = self.firms[firm_id]
                price = firm['price']
                quality = firm['quality']
                
                # Utility = Quality / Price (with weights)
                # Higher quality OR lower price = higher utility
                if price > 0:
                    utility = (quality * self.utility_quality_weight) / (price * self.utility_price_weight)
                    utilities[firm_id] = utility
                else:
                    utilities[firm_id] = 0.0
            
            # Choose firm with highest utility
            if utilities:
                best_firm = max(utilities, key=utilities.get)
                firm = self.firms[best_firm]
                
                # How much can household afford?
                quantity = budget / firm['price'] if firm['price'] > 0 else 0
                
                # But firm can only sell what's in inventory
                quantity = min(quantity, firm['inventory'])
                
                if quantity > 0:
                    total_demand[best_firm] += quantity
                    actual_cost = quantity * firm['price']
                    household['money'] -= actual_cost
                    firm['inventory'] -= quantity
        
        # Phase 5: Calculate firm profits and rewards
        rewards = {}
        for agent_id in self._agent_ids:
            firm = self.firms[agent_id]
            
            # Revenue from sales
            revenue = total_demand[agent_id] * firm['price']
            
            # Costs:
            wage_costs = firm['employees'] * firm['wage']
            production_costs = firm['production'] * self.production_cost_per_unit
            storage_costs = firm['inventory'] * self.storage_cost_per_unit
            fixed_costs = self.fixed_costs
            
            total_costs = wage_costs + production_costs + storage_costs + fixed_costs
            
            profit = revenue - total_costs
            
            firm['revenue'] = revenue
            firm['costs'] = total_costs
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
                'inventory': self.firms[agent_id]['inventory'],
                'production': self.firms[agent_id]['production'],
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
        
        # Household statistics
        avg_household_money = np.mean([hh['money'] for hh in self.households])
        
        obs = np.array([
            firm['price'],
            firm['wage'],
            firm['employees'],
            firm['inventory'],
            firm['profit'] / self.reward_scale,
            market_avg_price,
            market_min_price,
            market_max_price,
            market_avg_wage,
            market_min_wage,
            market_max_wage,
            total_employed,
            unemployment_rate,
            avg_household_money,
            self.timestep / self.max_steps,
        ], dtype=np.float32)
        
        return np.clip(obs, -100.0, 100.0)

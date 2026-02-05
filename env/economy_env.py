import numpy as np
import yaml
from pathlib import Path
from gymnasium.spaces import Box, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class SimpleEconomyEnv(MultiAgentEnv):
    """
    Enhanced economy simulation with:
    - Household skill levels and job matching
    - Firm bankruptcy mechanism
    - Extended action space (price, wage, marketing, quality, capacity)
    - Unemployment benefits (state employer for jobless households)
    - Balanced economic parameters for challenging but fair gameplay
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        # Load default config from YAML
        config_path = Path(__file__).parent.parent / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
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
        
        # Adjustment rates for each action level (5 discrete options)
        self.adjustment_rates = {
            0: -0.10,  # -10%
            1: -0.05,  # -5%
            2: 0.00,   # 0% (no change)
            3: 0.05,   # +5%
            4: 0.10,   # +10%
        }
        
        # Production parameters (BALANCED)
        prod_cfg = econ_cfg.get('production', {})
        self.productivity_base = prod_cfg.get('productivity_per_employee', 6.0)
        self.production_cost_per_unit = prod_cfg.get('cost_per_unit', 2.0)
        self.fixed_costs = prod_cfg.get('fixed_costs', 30.0)
        self.storage_cost_per_unit = prod_cfg.get('storage_cost_per_unit', 0.2)
        
        # Investment costs
        self.marketing_cost_per_level = 20.0
        self.quality_improvement_cost = 30.0
        self.capacity_expansion_cost = 50.0
        
        # Household parameters
        hh_cfg = econ_cfg.get('households', {})
        self.consumption_rate = hh_cfg.get('consumption_rate', 0.7)
        self.utility_price_weight = hh_cfg.get('utility_price_weight', 1.0)
        self.utility_quality_weight = hh_cfg.get('utility_quality_weight', 0.5)
        self.utility_marketing_weight = 0.3
        
        # Reward parameters
        self.reward_scale = econ_cfg.get('reward_scale', 100.0)
        
        # Bounds
        price_bounds = econ_cfg.get('price_bounds', {'min': 5.0, 'max': 100.0})
        wage_bounds = econ_cfg.get('wage_bounds', {'min': 5.0, 'max': 50.0})
        self.price_min = price_bounds['min']
        self.price_max = price_bounds['max']
        self.wage_min = wage_bounds['min']
        self.wage_max = wage_bounds['max']
        
        # Bankruptcy threshold
        self.bankruptcy_threshold = -400.0
        
        self._agent_ids = set(f"firm_{i}" for i in range(self.n_firms))
        
        # Extended Observation Space (20 features)
        self._obs_space = Box(low=-1000.0, high=1000.0, shape=(20,), dtype=np.float32)
        
        # Extended Action Space: [price_change, wage_change, marketing_level, quality_improve, capacity_change]
        self._action_space = MultiDiscrete([5, 5, 3, 2, 3])
        
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
        
        price_range = firm_ranges.get('price', {'min': 20.0, 'max': 40.0})
        wage_range = firm_ranges.get('wage', {'min': 15.0, 'max': 30.0})
        emp_range = firm_ranges.get('max_employees', {'min': 3, 'max': 10})
        money_range = hh_ranges.get('money', {'min': 100.0, 'max': 200.0})
        
        # Initialize firms
        self.firms = {}
        for i in range(self.n_firms):
            self.firms[f"firm_{i}"] = {
                'price': np.random.uniform(price_range['min'], price_range['max']),
                'wage': np.random.uniform(wage_range['min'], wage_range['max']),
                'max_employees': np.random.randint(emp_range['min'], emp_range['max'] + 1),
                'employees': 0,
                'inventory': 0.0,
                'production': 0.0,
                'capital': np.random.uniform(1000.0, 1500.0),
                'profit': 0.0,
                'profit_last_step': 0.0,
                'revenue': 0.0,
                'costs': 0.0,
                'quality': np.random.uniform(0.5, 0.8),
                'marketing': np.random.uniform(0.3, 0.6),
                'bankrupt': False,
            }
        
        # Initialize households with skill levels
        self.households = []
        for _ in range(self.n_households):
            self.households.append({
                'money': np.random.uniform(money_range['min'], money_range['max']),
                'skill_level': np.random.uniform(0.3, 1.0),
                'employer': None,
                'wage': 0.0,
                'wealth_type': np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2]),
            })
        
        self.timestep = 0
        self.bankruptcies_this_episode = 0
        
        obs = {agent_id: self._get_obs(agent_id) for agent_id in self._agent_ids}
        return obs, {agent_id: {} for agent_id in self._agent_ids}
    
    def step(self, action_dict):
        # Phase 0: Check and mark bankruptcies
        active_firms = [aid for aid in self._agent_ids if not self.firms[aid]['bankrupt']]
        
        # Phase 1: Firms take strategic actions
        for agent_id in active_firms:
            action = action_dict.get(agent_id, [2, 2, 1, 0, 1])
            firm = self.firms[agent_id]
            
            price_action = action[0]
            wage_action = action[1]
            marketing_action = action[2]
            quality_action = action[3]
            capacity_action = action[4]
            
            # Apply price adjustment
            price_adjustment = self.adjustment_rates[price_action]
            if price_adjustment != 0:
                firm['price'] *= (1 + price_adjustment)
            firm['price'] = np.clip(firm['price'], self.price_min, self.price_max)
            
            # Apply wage adjustment
            wage_adjustment = self.adjustment_rates[wage_action]
            if wage_adjustment != 0:
                firm['wage'] *= (1 + wage_adjustment)
            firm['wage'] = np.clip(firm['wage'], self.wage_min, self.wage_max)
            
            # Marketing investment
            if marketing_action == 0:
                firm['marketing'] = max(0.1, firm['marketing'] - 0.1)
            elif marketing_action == 2:
                cost = self.marketing_cost_per_level
                if firm['capital'] >= cost:
                    firm['capital'] -= cost
                    firm['marketing'] = min(1.0, firm['marketing'] + 0.1)
            
            # Quality improvement
            if quality_action == 1:
                cost = self.quality_improvement_cost
                if firm['capital'] >= cost:
                    firm['capital'] -= cost
                    firm['quality'] = min(1.0, firm['quality'] + 0.05)
            
            # Capacity adjustment
            if capacity_action == 0:
                firm['max_employees'] = max(1, firm['max_employees'] - 1)
            elif capacity_action == 2:
                cost = self.capacity_expansion_cost
                if firm['capital'] >= cost:
                    firm['capital'] -= cost
                    firm['max_employees'] += 1
            
            # Reset employees for labor market
            firm['employees'] = 0
            firm['employee_skills'] = []
        
        # Phase 2: Labor market with skill-based matching
        for household in self.households:
            household['employer'] = None
            household['wage'] = 0.0
        
        # Sort households by skill (highest first)
        households_sorted = sorted(
            self.households, 
            key=lambda h: h['skill_level'], 
            reverse=True
        )
        
        # Sort firms by wage (highest first) - only active firms
        firms_by_wage = sorted(
            active_firms,
            key=lambda aid: self.firms[aid]['wage'],
            reverse=True
        )
        
        # Skill-based matching: Best skills go to best wages
        for household in households_sorted:
            for firm_id in firms_by_wage:
                firm = self.firms[firm_id]
                if firm['employees'] < firm['max_employees']:
                    household['employer'] = firm_id
                    household['wage'] = firm['wage']
                    firm['employees'] += 1
                    firm['employee_skills'].append(household['skill_level'])
                    
                    # Pay wage immediately
                    household['money'] += firm['wage']
                    firm['capital'] -= firm['wage']
                    break
        
        # NEW: Unemployment benefits - state employs jobless households
        if active_firms:
            # Calculate state wage: average of min and max wage from active firms
            wages = [self.firms[aid]['wage'] for aid in active_firms]
            state_wage = (min(wages) + max(wages)) / 2.0
            
            for household in self.households:
                if household['employer'] is None:
                    household['employer'] = 'state'
                    household['wage'] = state_wage
                    household['money'] += state_wage
        
        # Phase 3: Production (skill affects productivity)
        for agent_id in active_firms:
            firm = self.firms[agent_id]
            
            if firm['employees'] > 0:
                # Average skill of employees affects productivity
                avg_skill = np.mean(firm['employee_skills'])
                skill_multiplier = 0.5 + (avg_skill * 0.5)
                
                # Production = employees × base_productivity × skill_multiplier
                production = firm['employees'] * self.productivity_base * skill_multiplier
                firm['production'] = production
                firm['inventory'] += production
            else:
                firm['production'] = 0.0
        
        # Phase 4: Goods market with utility-based purchasing
        total_sales = {agent_id: 0.0 for agent_id in active_firms}
        
        for household in self.households:
            if household['money'] <= 0:
                continue
            
            budget = household['money'] * self.consumption_rate
            
            # Adjust budget by wealth type
            if household['wealth_type'] == 'low':
                budget *= 0.8
            elif household['wealth_type'] == 'high':
                budget *= 1.2
            
            # Calculate utility for each firm (only active firms)
            utilities = {}
            for firm_id in active_firms:
                firm = self.firms[firm_id]
                price = firm['price']
                quality = firm['quality']
                marketing = firm['marketing']
                
                # Utility = (Quality × Q_weight + Marketing × M_weight) / (Price × P_weight)
                if price > 0:
                    numerator = (quality * self.utility_quality_weight + 
                               marketing * self.utility_marketing_weight)
                    denominator = price * self.utility_price_weight
                    utility = numerator / denominator
                    utilities[firm_id] = utility
                else:
                    utilities[firm_id] = 0.0
            
            if not utilities:
                continue
            
            # Choose firm with highest utility
            best_firm = max(utilities, key=utilities.get)
            firm = self.firms[best_firm]
            
            # Calculate quantity to buy
            quantity = budget / firm['price'] if firm['price'] > 0 else 0
            quantity = min(quantity, firm['inventory'])
            
            if quantity > 0:
                total_sales[best_firm] += quantity
                actual_cost = quantity * firm['price']
                household['money'] -= actual_cost
                firm['inventory'] -= quantity
        
        # Phase 5: Calculate profits, update capital, check bankruptcies
        rewards = {}
        
        for agent_id in self._agent_ids:
            firm = self.firms[agent_id]
            
            if firm['bankrupt']:
                rewards[agent_id] = -10.0
                continue
            
            # Revenue from sales
            revenue = total_sales.get(agent_id, 0.0) * firm['price']
            
            # Costs
            production_costs = firm['production'] * self.production_cost_per_unit
            storage_costs = firm['inventory'] * self.storage_cost_per_unit
            fixed_costs = self.fixed_costs
            
            total_costs = production_costs + storage_costs + fixed_costs
            # Note: wage costs already deducted from capital in Phase 2
            
            profit = revenue - total_costs
            
            firm['revenue'] = revenue
            firm['costs'] = total_costs
            firm['profit_last_step'] = firm['profit']
            firm['profit'] = profit
            
            # Update capital
            firm['capital'] += profit
            
            # Check bankruptcy
            if firm['capital'] < self.bankruptcy_threshold:
                firm['bankrupt'] = True
                self.bankruptcies_this_episode += 1
                rewards[agent_id] = -20.0
            else:
                # Reward is profit scaled, with bonus for positive capital
                capital_bonus = 0.0
                if firm['capital'] > 0:
                    capital_bonus = min(firm['capital'] / 1000.0, 5.0)
                
                rewards[agent_id] = (profit / self.reward_scale) + capital_bonus
        
        self.timestep += 1
        done = self.timestep >= self.max_steps
        
        # Also end episode if all firms bankrupt
        if all(self.firms[aid]['bankrupt'] for aid in self._agent_ids):
            done = True
        
        obs = {agent_id: self._get_obs(agent_id) for agent_id in self._agent_ids}
        dones = {agent_id: done for agent_id in self._agent_ids}
        dones['__all__'] = done
        
        infos = {
            agent_id: {
                'profit': self.firms[agent_id]['profit'],
                'revenue': self.firms[agent_id]['revenue'],
                'costs': self.firms[agent_id]['costs'],
                'capital': self.firms[agent_id]['capital'],
                'employees': self.firms[agent_id]['employees'],
                'inventory': self.firms[agent_id]['inventory'],
                'production': self.firms[agent_id]['production'],
                'quality': self.firms[agent_id]['quality'],
                'marketing': self.firms[agent_id]['marketing'],
                'bankrupt': self.firms[agent_id]['bankrupt'],
            } 
            for agent_id in self._agent_ids
        }
        
        return obs, rewards, dones, dones, infos
    
    def _get_obs(self, agent_id):
        """Create observation with comprehensive market data"""
        firm = self.firms[agent_id]
        
        # Only consider active (non-bankrupt) firms for market stats
        active_firms = [f for aid, f in self.firms.items() if not f['bankrupt']]
        
        if not active_firms:
            return np.zeros(20, dtype=np.float32)
        
        # Market statistics
        all_prices = [f['price'] for f in active_firms]
        all_wages = [f['wage'] for f in active_firms]
        
        market_avg_price = np.mean(all_prices)
        market_min_price = np.min(all_prices)
        market_max_price = np.max(all_prices)
        
        market_avg_wage = np.mean(all_wages)
        market_min_wage = np.min(all_wages)
        market_max_wage = np.max(all_wages)
        
        # Employment statistics (only count firm employment, not state)
        total_employed = sum(f['employees'] for f in active_firms)
        unemployment_rate = 1.0 - (total_employed / self.n_households)
        
        # Household statistics
        avg_household_money = np.mean([hh['money'] for hh in self.households])
        avg_household_skill = np.mean([hh['skill_level'] for hh in self.households])
        
        # Competition info
        competitors_alive = len(active_firms)
        
        obs = np.array([
            firm['price'],
            firm['wage'],
            firm['employees'],
            firm['inventory'],
            firm['capital'] / 100.0,
            firm['quality'],
            firm['marketing'],
            market_avg_price,
            market_min_price,
            market_max_price,
            market_avg_wage,
            market_min_wage,
            market_max_wage,
            total_employed,
            unemployment_rate,
            avg_household_money / 10.0,
            avg_household_skill,
            firm['profit_last_step'] / self.reward_scale,
            competitors_alive,
            self.timestep / self.max_steps,
        ], dtype=np.float32)
        
        return np.clip(obs, -1000.0, 1000.0)

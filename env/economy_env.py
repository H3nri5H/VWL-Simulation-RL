import numpy as np
import yaml
from pathlib import Path
from gymnasium.spaces import Box, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class SimpleEconomyEnv(MultiAgentEnv):
    """
    Enhanced economy simulation with:
    - SEQUENTIAL PURCHASING - realistic market-clearing mechanism
    - PRICE-SENSITIVE HOUSEHOLDS - each household has max acceptable price
    - WEALTH-BASED UTILITY PREFERENCES - rich prefer quality, poor prefer price
    - Random household order each step (fair competition)
    - Household skill levels and job matching
    - Firm bankruptcy mechanism with severe penalties
    - Hard cap on employee expansion (prevent monopolies)
    - Survivor diversity incentive (keep competitors alive)
    - NO artificial quality/marketing caps (can improve indefinitely)
    - Extended action space (price, wage, marketing, quality, capacity)
    - Supplier economy (unemployed work for supply chain companies)
    - Enhanced observation space (25 features including market share, trends, competitiveness)
    - Balanced economic parameters for challenging but fair gameplay
    - ALL parameters loaded from config.yaml
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
        
        # Production parameters
        prod_cfg = econ_cfg.get('production', {})
        self.productivity_base = prod_cfg.get('productivity_per_employee', 6.0)
        self.production_cost_per_unit = prod_cfg.get('cost_per_unit', 2.0)
        self.fixed_costs = prod_cfg.get('fixed_costs', 30.0)
        self.storage_cost_per_unit = prod_cfg.get('storage_cost_per_unit', 0.2)
        
        # Investment costs
        invest_cfg = econ_cfg.get('investment_costs', {})
        self.marketing_cost_per_level = invest_cfg.get('marketing_per_level', 20.0)
        self.quality_improvement_cost = invest_cfg.get('quality_improvement', 30.0)
        self.capacity_expansion_cost = invest_cfg.get('capacity_expansion', 50.0)
        
        # Capacity hard cap (NEW)
        self.max_employees_hard_cap = econ_cfg.get('max_employees_hard_cap', 150)
        
        # Quality & Marketing bounds (NEW - load from config!)
        quality_bounds = econ_cfg.get('quality_bounds', {'min': 0.1, 'max': 1.0})
        marketing_bounds = econ_cfg.get('marketing_bounds', {'min': 0.1, 'max': 1.0})
        self.quality_min = quality_bounds['min']
        self.quality_max = quality_bounds['max']
        self.marketing_min = marketing_bounds['min']
        self.marketing_max = marketing_bounds['max']
        
        # Bankruptcy parameters
        bankr_cfg = econ_cfg.get('bankruptcy', {})
        self.bankruptcy_threshold = bankr_cfg.get('threshold', -400.0)
        self.bankruptcy_penalty = bankr_cfg.get('penalty_reward', -20.0)
        
        # Household parameters
        hh_cfg = econ_cfg.get('households', {})
        self.consumption_rate = hh_cfg.get('consumption_rate', 0.7)
        
        # BASE utility weights (modified by wealth type)
        self.utility_price_weight = hh_cfg.get('utility_price_weight', 1.0)
        self.utility_quality_weight = hh_cfg.get('utility_quality_weight', 0.5)
        self.utility_marketing_weight = hh_cfg.get('utility_marketing_weight', 0.3)
        
        # NEW: Wealth-based utility modifiers
        wealth_util_mods = hh_cfg.get('wealth_utility_modifiers', {})
        self.wealth_utility_modifiers = {
            'low': wealth_util_mods.get('low', {'price_weight': 1.0, 'quality_weight': 1.0, 'marketing_weight': 1.0}),
            'medium': wealth_util_mods.get('medium', {'price_weight': 1.0, 'quality_weight': 1.0, 'marketing_weight': 1.0}),
            'high': wealth_util_mods.get('high', {'price_weight': 1.0, 'quality_weight': 1.0, 'marketing_weight': 1.0})
        }
        
        # Wealth multipliers
        wealth_mult = hh_cfg.get('wealth_multipliers', {})
        self.wealth_multipliers = {
            'low': wealth_mult.get('low', 0.8),
            'medium': wealth_mult.get('medium', 1.0),
            'high': wealth_mult.get('high', 1.2)
        }
        
        # Skill system
        skill_cfg = econ_cfg.get('skill_system', {})
        self.skill_base_multiplier = skill_cfg.get('base_multiplier', 0.5)
        self.skill_factor = skill_cfg.get('skill_factor', 0.5)
        
        # Reward parameters
        reward_cfg = econ_cfg.get('reward', {})
        self.reward_scale = reward_cfg.get('scale', 100.0)
        self.capital_bonus_divisor = reward_cfg.get('capital_bonus_divisor', 1000.0)
        self.capital_bonus_max = reward_cfg.get('capital_bonus_max', 5.0)
        
        # Reward shaping parameters
        self.market_share_bonus_weight = reward_cfg.get('market_share_bonus', 10.0)
        self.inventory_penalty_weight = reward_cfg.get('inventory_penalty', 2.0)
        self.exploration_penalty = reward_cfg.get('exploration_penalty', 5.0)
        
        # Survivor diversity parameters (NEW)
        self.survivor_diversity_threshold = reward_cfg.get('survivor_diversity_threshold', 5)
        self.survivor_diversity_penalty = reward_cfg.get('survivor_diversity_penalty', 10000)
        
        # Bounds
        price_bounds = econ_cfg.get('price_bounds', {'min': 5.0, 'max': 100.0})
        wage_bounds = econ_cfg.get('wage_bounds', {'min': 5.0, 'max': 50.0})
        self.price_min = price_bounds['min']
        self.price_max = price_bounds['max']
        self.wage_min = wage_bounds['min']
        self.wage_max = wage_bounds['max']
        
        self._agent_ids = set(f"firm_{i}" for i in range(self.n_firms))
        
        # ENHANCED Observation Space (25 features)
        self._obs_space = Box(low=-1000.0, high=1000.0, shape=(25,), dtype=np.float32)
        
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
        
        # Get initial ranges from config
        firm_ranges = self.init_ranges.get('firms', {})
        hh_ranges = self.init_ranges.get('households', {})
        
        price_range = firm_ranges.get('price', {'min': 20.0, 'max': 40.0})
        wage_range = firm_ranges.get('wage', {'min': 15.0, 'max': 30.0})
        emp_range = firm_ranges.get('max_employees', {'min': 3, 'max': 10})
        capital_range = firm_ranges.get('capital', {'min': 1000.0, 'max': 1500.0})
        quality_range = firm_ranges.get('quality', {'min': 0.5, 'max': 0.8})
        marketing_range = firm_ranges.get('marketing', {'min': 0.3, 'max': 0.6})
        
        money_range = hh_ranges.get('money', {'min': 100.0, 'max': 200.0})
        skill_range = hh_ranges.get('skill_level', {'min': 0.3, 'max': 1.0})
        max_price_range = hh_ranges.get('max_acceptable_price', {'min': 10.0, 'max': 100.0})
        wealth_dist = hh_ranges.get('wealth_distribution', {'low': 0.3, 'medium': 0.5, 'high': 0.2})
        
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
                'capital': np.random.uniform(capital_range['min'], capital_range['max']),
                'profit': 0.0,
                'profit_last_step': 0.0,
                'revenue': 0.0,
                'costs': 0.0,
                'sales': 0.0,
                'sales_last_step': 0.0,
                'quality': np.random.uniform(quality_range['min'], quality_range['max']),
                'marketing': np.random.uniform(marketing_range['min'], marketing_range['max']),
                'bankrupt': False,
            }
        
        # Initialize households with skill levels, max acceptable price, AND wealth type
        wealth_probs = [wealth_dist['low'], wealth_dist['medium'], wealth_dist['high']]
        self.households = []
        for _ in range(self.n_households):
            self.households.append({
                'money': np.random.uniform(money_range['min'], money_range['max']),
                'skill_level': np.random.uniform(skill_range['min'], skill_range['max']),
                'max_acceptable_price': np.random.uniform(max_price_range['min'], max_price_range['max']),
                'employer': None,
                'wage': 0.0,
                'wealth_type': np.random.choice(['low', 'medium', 'high'], p=wealth_probs),
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
                firm['marketing'] = max(self.marketing_min, firm['marketing'] - 0.1)
            elif marketing_action == 2:
                cost = self.marketing_cost_per_level
                if firm['capital'] >= cost:
                    firm['capital'] -= cost
                    # NO hard cap at 1.0! Can go up to marketing_max (2.0)
                    firm['marketing'] = min(self.marketing_max, firm['marketing'] + 0.1)
            
            # Quality improvement
            if quality_action == 1:
                cost = self.quality_improvement_cost
                if firm['capital'] >= cost:
                    firm['capital'] -= cost
                    # NO hard cap at 1.0! Can go up to quality_max (2.0)
                    firm['quality'] = min(self.quality_max, firm['quality'] + 0.05)
            
            # Capacity adjustment with HARD CAP (NEW)
            if capacity_action == 0:
                firm['max_employees'] = max(1, firm['max_employees'] - 1)
            elif capacity_action == 2:
                cost = self.capacity_expansion_cost
                # Check both capital AND hard cap
                if firm['capital'] >= cost and firm['max_employees'] < self.max_employees_hard_cap:
                    firm['capital'] -= cost
                    firm['max_employees'] += 1
            
            # Reset employees for labor market
            firm['employees'] = 0
            firm['employee_skills'] = []
            
            # Store last step sales for trend calculation
            firm['sales_last_step'] = firm['sales']
            firm['sales'] = 0.0
        
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
        
        # Unemployed work for "suppliers" (supply chain companies)
        if active_firms:
            wages = [self.firms[aid]['wage'] for aid in active_firms]
            supplier_wage = (min(wages) + max(wages)) / 2.0
            
            for household in self.households:
                if household['employer'] is None:
                    household['employer'] = 'suppliers'
                    household['wage'] = supplier_wage
                    household['money'] += supplier_wage
        
        # Phase 3: Production (skill affects productivity)
        for agent_id in active_firms:
            firm = self.firms[agent_id]
            
            if firm['employees'] > 0:
                # Average skill of employees affects productivity
                avg_skill = np.mean(firm['employee_skills'])
                skill_multiplier = self.skill_base_multiplier + (avg_skill * self.skill_factor)
                
                # Production = employees × base_productivity × skill_multiplier
                production = firm['employees'] * self.productivity_base * skill_multiplier
                firm['production'] = production
                firm['inventory'] += production
            else:
                firm['production'] = 0.0
        
        # Phase 4: SEQUENTIAL PURCHASING WITH WEALTH-BASED UTILITY (NEW!)
        total_sales = {agent_id: 0.0 for agent_id in active_firms}
        
        # RANDOMIZE household order each step (fairness)
        households_random = self.households.copy()
        np.random.shuffle(households_random)
        
        for household in households_random:
            if household['money'] <= 0:
                continue
            
            # Calculate budget for this household
            budget = household['money'] * self.consumption_rate
            budget *= self.wealth_multipliers.get(household['wealth_type'], 1.0)
            
            remaining_budget = budget
            
            # Filter firms by price sensitivity
            max_price = household['max_acceptable_price']
            affordable_firms = [
                firm_id for firm_id in active_firms 
                if self.firms[firm_id]['price'] <= max_price
            ]
            
            if not affordable_firms:
                continue  # This household can't afford any firm
            
            # NEW: Wealth-based utility calculation!
            wealth_type = household['wealth_type']
            wealth_mods = self.wealth_utility_modifiers.get(wealth_type, {})
            
            # Get modifiers for this wealth type
            price_weight = self.utility_price_weight * wealth_mods.get('price_weight', 1.0)
            quality_weight = self.utility_quality_weight * wealth_mods.get('quality_weight', 1.0)
            marketing_weight = self.utility_marketing_weight * wealth_mods.get('marketing_weight', 1.0)
            
            # Calculate utility for affordable firms with WEALTH-BASED weights
            utilities = {}
            for firm_id in affordable_firms:
                firm = self.firms[firm_id]
                price = firm['price']
                quality = firm['quality']
                marketing = firm['marketing']
                
                if price > 0:
                    numerator = (quality * quality_weight + marketing * marketing_weight)
                    denominator = price * price_weight
                    utility = numerator / denominator
                    utilities[firm_id] = utility
                else:
                    utilities[firm_id] = 0.0
            
            if not utilities:
                continue
            
            # SEQUENTIAL PURCHASING: Sort firms by utility (best first)
            firms_sorted_by_utility = sorted(
                utilities.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Buy from firms in order until budget exhausted or all firms visited
            for firm_id, utility_score in firms_sorted_by_utility:
                if remaining_budget <= 0:
                    break  # No money left
                
                firm = self.firms[firm_id]
                
                # Skip if firm has no inventory
                if firm['inventory'] <= 0:
                    continue
                
                # Calculate how much this household can buy from this firm
                max_quantity_by_budget = remaining_budget / firm['price'] if firm['price'] > 0 else 0
                max_quantity_by_inventory = firm['inventory']
                
                # Actual purchase quantity (limited by budget OR inventory)
                quantity = min(max_quantity_by_budget, max_quantity_by_inventory)
                
                if quantity > 0:
                    actual_cost = quantity * firm['price']
                    
                    # Execute purchase
                    total_sales[firm_id] += quantity
                    household['money'] -= actual_cost
                    firm['inventory'] -= quantity
                    remaining_budget -= actual_cost
                
                # If this firm is sold out, move to next firm
                # If household still has budget, continue buying from next best firm
        
        # Update sales tracking
        for agent_id in active_firms:
            self.firms[agent_id]['sales'] = total_sales.get(agent_id, 0.0)
        
        # Phase 5: Calculate profits, update capital, check bankruptcies
        rewards = {}
        
        # Calculate total market sales for market share
        total_market_sales = sum(self.firms[aid]['sales'] for aid in active_firms)
        
        for agent_id in self._agent_ids:
            firm = self.firms[agent_id]
            
            if firm['bankrupt']:
                rewards[agent_id] = self.bankruptcy_penalty  # Severe penalty
                continue
            
            # Revenue from sales
            revenue = total_sales.get(agent_id, 0.0) * firm['price']
            
            # Costs
            production_costs = firm['production'] * self.production_cost_per_unit
            storage_costs = firm['inventory'] * self.storage_cost_per_unit
            fixed_costs = self.fixed_costs
            
            total_costs = production_costs + storage_costs + fixed_costs
            
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
                rewards[agent_id] = self.bankruptcy_penalty
            else:
                # REBALANCED REWARD SHAPING
                base_reward = profit / self.reward_scale  # REDUCED impact (scale 1000 vs 100)
                
                # Capital bonus
                capital_bonus = 0.0
                if firm['capital'] > 0:
                    capital_bonus = min(firm['capital'] / self.capital_bonus_divisor, self.capital_bonus_max)
                
                # Market share bonus
                market_share_bonus = 0.0
                if total_market_sales > 0:
                    own_market_share = firm['sales'] / total_market_sales
                    market_share_bonus = own_market_share * self.market_share_bonus_weight
                
                # Inventory penalty
                inventory_penalty = 0.0
                if firm['production'] > 0:
                    inventory_ratio = firm['inventory'] / firm['production']
                    if inventory_ratio > 1.0:
                        inventory_penalty = -inventory_ratio * self.inventory_penalty_weight
                
                # Exploration penalty
                exploration_penalty = 0.0
                if revenue == 0:
                    exploration_penalty = -self.exploration_penalty
                
                # Total reward
                rewards[agent_id] = (base_reward + capital_bonus + market_share_bonus + 
                                   inventory_penalty + exploration_penalty)
        
        # NEW: Survivor diversity penalty (end of episode)
        if self.timestep == self.max_steps - 1:
            active_count = sum(1 for f in self.firms.values() if not f['bankrupt'])
            
            if active_count < self.survivor_diversity_threshold:
                missing_survivors = self.survivor_diversity_threshold - active_count
                diversity_penalty = missing_survivors * self.survivor_diversity_penalty
                
                # Apply penalty to surviving firms (they should have kept competition alive)
                for agent_id in self._agent_ids:
                    if not self.firms[agent_id]['bankrupt']:
                        rewards[agent_id] -= diversity_penalty
        
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
                'sales': self.firms[agent_id]['sales'],
                'quality': self.firms[agent_id]['quality'],
                'marketing': self.firms[agent_id]['marketing'],
                'bankrupt': self.firms[agent_id]['bankrupt'],
            } 
            for agent_id in self._agent_ids
        }
        
        return obs, rewards, dones, dones, infos
    
    def _get_obs(self, agent_id):
        """Create observation with comprehensive market data (25 features)"""
        firm = self.firms[agent_id]
        
        # Only consider active (non-bankrupt) firms for market stats
        active_firms = [f for aid, f in self.firms.items() if not f['bankrupt']]
        
        if not active_firms:
            return np.zeros(25, dtype=np.float32)
        
        # Market statistics
        all_prices = [f['price'] for f in active_firms]
        all_wages = [f['wage'] for f in active_firms]
        
        market_avg_price = np.mean(all_prices)
        market_min_price = np.min(all_prices)
        market_max_price = np.max(all_prices)
        market_median_price = np.median(all_prices)
        
        market_avg_wage = np.mean(all_wages)
        market_min_wage = np.min(all_wages)
        market_max_wage = np.max(all_wages)
        market_median_wage = np.median(all_wages)
        
        # Employment statistics
        total_employed = sum(f['employees'] for f in active_firms)
        unemployment_rate = 1.0 - (total_employed / self.n_households)
        
        # Household statistics
        avg_household_money = np.mean([hh['money'] for hh in self.households])
        avg_household_skill = np.mean([hh['skill_level'] for hh in self.households])
        
        # Competition info
        competitors_alive = len(active_firms)
        
        # Market share
        total_market_sales = sum(f['sales'] for f in active_firms)
        if total_market_sales > 0:
            own_market_share = firm['sales'] / total_market_sales
        else:
            own_market_share = 1.0 / len(active_firms)
        
        # Sales trend
        sales_trend = firm['sales'] - firm['sales_last_step']
        
        # Inventory ratio
        if firm['production'] > 0:
            inventory_ratio = firm['inventory'] / firm['production']
        else:
            inventory_ratio = 0.0
        
        # Wage competitiveness
        if market_median_wage > 0:
            wage_competitiveness = firm['wage'] / market_median_wage
        else:
            wage_competitiveness = 1.0
        
        # Price competitiveness
        if market_median_price > 0:
            price_competitiveness = firm['price'] / market_median_price
        else:
            price_competitiveness = 1.0
        
        obs = np.array([
            # Own state (7)
            firm['price'],
            firm['wage'],
            firm['employees'],
            firm['inventory'],
            firm['capital'] / 100.0,
            firm['quality'],
            firm['marketing'],
            # Market statistics (6)
            market_avg_price,
            market_min_price,
            market_max_price,
            market_avg_wage,
            market_min_wage,
            market_max_wage,
            # Aggregates (4)
            total_employed,
            unemployment_rate,
            avg_household_money / 10.0,
            avg_household_skill,
            # Meta (3)
            firm['profit_last_step'] / self.reward_scale,
            competitors_alive,
            self.timestep / self.max_steps,
            # Strategic insights (5)
            own_market_share,
            sales_trend / 10.0,
            inventory_ratio,
            wage_competitiveness,
            price_competitiveness,
        ], dtype=np.float32)
        
        return np.clip(obs, -1000.0, 1000.0)

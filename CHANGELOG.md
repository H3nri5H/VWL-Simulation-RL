# Changelog

All notable changes to the VWL Simulation project are documented here.

## [5.2.0] - 2026-02-05

### Added - Enhanced Observation Space (25 Features)

#### Strategic Market Intelligence (5 New Features)
- **Market Share Tracking**
  - `own_market_share`: Percentage of total market sales controlled by firm
  - Calculation: `firm_sales / total_market_sales`
  - Range: 0.0 (no sales) to 1.0 (monopoly)
  - Enables firms to understand competitive position
  - Critical for strategic decision-making

- **Sales Trend Analysis**
  - `sales_trend`: Change in sales volume from last step
  - Calculation: `current_sales - previous_sales`
  - Positive values = growing market share
  - Negative values = losing customers
  - Helps AI learn momentum-based strategies

- **Inventory Efficiency**
  - `inventory_ratio`: Inventory-to-production ratio
  - Calculation: `current_inventory / production_this_step`
  - High values = overproduction (storage costs accumulating)
  - Low values = efficient production matching demand
  - Optimizes production decisions

- **Wage Competitiveness**
  - `wage_competitiveness`: Own wage relative to market median
  - Calculation: `own_wage / market_median_wage`
  - > 1.0 = paying above median (attracting better workers)
  - < 1.0 = paying below median (risk losing workers)
  - Directly impacts employee skill acquisition

- **Price Competitiveness**
  - `price_competitiveness`: Own price relative to market median
  - Calculation: `own_price / market_median_price`
  - > 1.0 = charging premium (better margins, fewer sales)
  - < 1.0 = discount pricing (lower margins, more sales)
  - Reveals pricing strategy effectiveness

#### Improved Market Statistics
- Added **median price** and **median wage** calculations
- More robust than mean for skewed distributions
- Better competitiveness benchmarks
- Reduces impact of outliers (bankrupt or extreme firms)

### Changed - Unemployment System Overhaul

#### From "State" to "Suppliers"
- **Old System**: Unemployed households employed by fictional "state"
  - Unrealistic government intervention
  - Didn't fit market economy simulation
  - Terminology confused users

- **New System**: Unemployed work for "suppliers"
  - Supplier companies = B2B firms in supply chain
  - Provide materials, logistics, services to main firms
  - More realistic economic model
  - Maintains purchasing power without government intervention

- **Wage Calculation** (unchanged):
  ```python
  supplier_wage = (min(active_firm_wages) + max(active_firm_wages)) / 2.0
  ```
  - Suppliers pay competitive market rate
  - Keeps unemployed households as consumers
  - Prevents market collapse from unemployment

### Enhanced - Sales Tracking System

#### Firms Now Track Sales History
- **New Attribute**: `sales_last_step`
  - Stores previous step's sales volume
  - Initialized to 0.0 at episode start
  - Updated before new step calculation

- **Current Sales**: `sales` attribute
  - Tracks current step sales in units
  - Used for market share calculation
  - Separate from revenue (sales × price)

- **Trend Calculation**:
  ```python
  sales_trend = firm['sales'] - firm['sales_last_step']
  # Positive = growing, Negative = shrinking, Zero = stable
  ```

### Impact on Learning

#### Better Strategic Decisions
With 25 features (up from 20), firms can now:
1. **Understand Market Position**: See exact market share percentage
2. **Track Performance Trends**: Know if strategies are working (sales increasing?)
3. **Optimize Production**: Match production to demand (inventory ratio)
4. **Benchmark Competitively**: Compare wage/price to median, not just average
5. **Avoid Overproduction**: See when inventory is building up

#### Expected Training Improvements
- **Faster Convergence**: More informative observations → better learning signals
- **Better Strategies**: Firms learn competitive positioning, not just random actions
- **Reduced Variance**: Median-based comparisons more stable than mean
- **Market Dynamics**: AI can learn to react to competitors (e.g., price wars)

### Technical Details

#### Complete Observation Vector (25 Features)
```python
[
  # Own State (7)
  0:  price,
  1:  wage,
  2:  employees,
  3:  inventory,
  4:  capital / 100.0,
  5:  quality,
  6:  marketing,
  
  # Market Statistics (6)
  7:  market_avg_price,
  8:  market_min_price,
  9:  market_max_price,
  10: market_avg_wage,
  11: market_min_wage,
  12: market_max_wage,
  
  # Aggregates (4)
  13: total_employed,
  14: unemployment_rate,
  15: avg_household_money / 10.0,
  16: avg_household_skill,
  
  # Meta (3)
  17: profit_last_step / reward_scale,
  18: competitors_alive,
  19: timestep / max_steps,
  
  # NEW Strategic Insights (5)
  20: own_market_share,           # % of market (0.0-1.0)
  21: sales_trend / 10.0,         # Sales momentum
  22: inventory_ratio,            # Production efficiency
  23: wage_competitiveness,       # Wage vs. median
  24: price_competitiveness,      # Price vs. median
]
```

#### Observation Space Update
```python
# Old:
_obs_space = Box(low=-1000.0, high=1000.0, shape=(20,), dtype=np.float32)

# New:
_obs_space = Box(low=-1000.0, high=1000.0, shape=(25,), dtype=np.float32)
```

#### Sales Tracking in Step Function
```python
# Before market phase:
firm['sales_last_step'] = firm['sales']
firm['sales'] = 0.0

# After market phase:
for agent_id in active_firms:
    self.firms[agent_id]['sales'] = total_sales.get(agent_id, 0.0)
```

### Migration Notes

#### For Existing Checkpoints
- **Checkpoints from v5.0/v5.1 are INCOMPATIBLE**
  - Old checkpoints expect 20 features
  - New environment provides 25 features
  - Must retrain from scratch with v5.2.0

#### For Config Files
- **No changes required to config.yaml**
  - All parameters remain the same
  - Observation space automatically adjusted
  - Action space unchanged

#### For Training
- **Same training command**:
  ```bash
  python train.py
  python train.py --resume  # Starts fresh if no v5.2 checkpoint
  ```

### Realistic Economic Model

#### Supply Chain Economics
- **Multi-Agent Firms** (our simulation): Consumer goods producers
  - Compete for household purchases
  - Employ skilled workers directly
  - Visible in simulation

- **Supplier Firms** (background economy): B2B service providers
  - Logistics, raw materials, professional services
  - Employ workers not hired by main firms
  - Pay competitive wages (market average)
  - Not directly simulated (simplified)

#### Why This is More Realistic
- **Real Economy**: Not everyone works for consumer-facing companies
  - Many work in supply chains (manufacturing, distribution)
  - B2B sector is huge (often larger than B2C)
  - Unemployment is never 100% (always some jobs available)

- **Our Model**: Suppliers represent entire B2B sector
  - Simplified but captures essence
  - Maintains economic circulation
  - No artificial government intervention needed

---

## [5.1.0] - 2026-02-05

### Added - Database-Ready CSV Export

#### Long Format (Normalized) CSV Output
- **Changed CSV structure from WIDE to LONG format**
  - Wide format (old): One row per step, all entities as columns
  - Long format (new): One row per entity per step
  - Database-friendly normalized structure
  - Ready for SQL import with proper foreign keys

#### Separate CSV Files
- **firms.csv**:
  - Columns: `version`, `seed`, `step`, `firm_id`, `price`, `wage`, `capital`, `employees`, `max_employees`, `quality`, `marketing`, `profit`, `revenue`, `costs`, `inventory`, `production`, `bankrupt`
  - One row per firm per timestep
  - Example: 10 firms × 100 steps = 1,000 rows
  
- **households.csv**:
  - Columns: `version`, `seed`, `step`, `household_id`, `money`, `employer`, `wage`, `skill_level`, `wealth_type`
  - One row per household per timestep
  - Example: 50 households × 100 steps = 5,000 rows

#### Multi-Simulation Tracking
- **Version Column**: Extracted from CHANGELOG.md (e.g., "v5.1.0")
  - Tracks which model version generated the data
  - Allows comparing different model iterations
  - Defaults to "v5.0" if CHANGELOG not found
  
- **Seed Column**: Simulation seed for reproducibility
  - Combines multiple simulations in single database
  - Query by `(version, seed)` to get specific simulation
  - Enables statistical analysis across many runs

#### Database Import Benefits
- **Easy Queries**:
  ```sql
  -- Get all firm data for specific simulation
  SELECT * FROM firms WHERE version = 'v5.2.0' AND seed = 123456;
  
  -- Compare firm performance across versions
  SELECT version, AVG(profit) FROM firms GROUP BY version;
  
  -- Analyze household employment over time
  SELECT step, COUNT(*) FROM households 
  WHERE employer != 'suppliers' GROUP BY step;
  ```

- **Scalability**: Can store thousands of simulations in two tables
- **Relationships**: Easy joins between firms and households by step
- **Analytics**: Direct import to data analysis tools (pandas, R, Tableau)

### Added - Unemployment Benefits System

#### State Employer for Jobless Households
- **Problem**: Previously unemployed households had no income → market collapse
- **Solution**: Fictional "state" employer provides unemployment benefits
- **Calculation**: 
  ```python
  state_wage = (min(active_firm_wages) + max(active_firm_wages)) / 2.0
  ```
- **Effect**: 
  - Unemployed households receive average wage
  - Maintains purchasing power in economy
  - Prevents market from stalling
  - Realistic economic safety net

#### Tracking State Employment
- `employer = 'state'` for unemployed households
- Separate statistics for firm vs state employment
- Display shows:
  - Firm Employment: X%
  - State Employment (Benefits): Y%
  - Total Employment: X + Y = 100%

---

## [5.0.0] - 2026-02-03

### Added - Major Economy Expansion

#### Labor Market & Skills System
- **Household Skill Levels** (0.3 to 1.0)
  - Each household has individual skill rating affecting productivity
  - Skill-based job matching algorithm
  - Best skilled workers are matched with highest-paying firms first
  - Employee skills directly affect firm productivity:
    - Productivity = Employees × Base (6.0) × Skill Multiplier (0.5 to 1.0)
    - Skill Multiplier = 0.5 + (avg_employee_skill × 0.5)
  - Creates realistic labor market competition

#### Bankruptcy Mechanism
- **Firm Capital System**
  - Starting capital: 1000-1500 (randomized)
  - Capital tracks all financial flows:
    - Wages paid immediately reduce capital
    - Revenue from sales increases capital
    - Production, storage, and fixed costs reduce capital
  - Bankruptcy threshold: Capital < -400
  - Bankrupt firms:
    - Receive penalty reward (-20)
    - Stop participating in markets
    - Episode ends if all firms bankrupt
  - Surviving firms get capital bonus in reward (+0 to +5)

#### Extended Action Space (5 Actions)
- **Price Adjustment** (5 levels): -10%, -5%, 0%, +5%, +10%
- **Wage Adjustment** (5 levels): -10%, -5%, 0%, +5%, +10%
- **Marketing Investment** (3 levels):
  - Decrease marketing level (-0.1, free)
  - Keep current level (no cost)
  - Increase marketing level (+0.1, costs 20)
  - Affects utility calculation for consumers
- **Quality Improvement** (2 levels):
  - Don't improve (no cost)
  - Improve quality (+0.05, costs 30)
  - Affects utility calculation for consumers
- **Capacity Management** (3 levels):
  - Decrease max employees by 1 (free)
  - Keep current capacity (no cost)
  - Increase max employees by 1 (costs 50)
  - Allows dynamic workforce scaling

#### Enhanced Utility-Based Market
- **Consumer Utility Function**:
  - Utility = (Quality × 0.5 + Marketing × 0.3) / (Price × 1.0)
  - Households buy from firm with highest utility
  - Strategic investments in quality/marketing pay off
- **Household Wealth Types**:
  - Low wealth (30%): 0.8× consumption rate
  - Medium wealth (50%): 1.0× consumption rate  
  - High wealth (20%): 1.2× consumption rate
  - Creates diverse consumer behavior

#### Expanded Observation Space (20 Features)
- Own firm state: price, wage, employees, inventory, capital, quality, marketing
- Market statistics: avg/min/max prices and wages
- Employment: total employed, unemployment rate
- Household data: average money, average skill level
- Competition: number of competitors still active
- Performance: profit from last step
- Progress: current timestep / max steps

### Changed - Balanced Economics

#### Production Parameters (BALANCED)
- **Productivity**: 5.0 → **6.0** units per employee
  - Increased output to improve profitability
  - Skill multiplier (0.5-1.0) adds variance
- **Fixed Costs**: 50.0 → **30.0** per step
  - Reduced cost pressure on firms
  - More room for strategic investments
- **Storage Costs**: 0.5 → **0.2** per unit
  - Less penalty for overproduction
  - Encourages production capacity utilization

#### Capital & Bankruptcy
- **Starting Capital**: 500-1000 → **1000-1500**
  - More initial buffer for firms
  - Allows early investments in marketing/quality
- **Bankruptcy Threshold**: -200 → **-400**
  - More forgiving for temporary losses
  - Firms can survive 90+ steps with 30% sales rate

#### Market Dynamics
- **Consumption Rate**: 60% → **70%**
  - Households spend more of their money
  - Increases market demand and sales opportunities
  - Market supports ~71% sales rate per firm on average

#### Economic Balance Results
- Break-even at ~24% sales rate (easily achievable)
- Positive profit from 30% sales rate onwards
- At 50% sales: +189 profit per step
- At 70% sales: +339 profit per step
- Challenge Rating: ⭐⭐⭐ (Challenging but fair)

---

## [4.0.0] - 2026-02-02
## [3.0.0] - 2026-02-01
## [2.0.0] - 2026-01-31
## [1.0.0] - 2026-01-30

(See previous versions above for full history)

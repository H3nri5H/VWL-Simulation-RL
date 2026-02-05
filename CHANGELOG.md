# Changelog

All notable changes to the VWL Simulation project are documented here.

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
  SELECT * FROM firms WHERE version = 'v5.1.0' AND seed = 123456;
  
  -- Compare firm performance across versions
  SELECT version, AVG(profit) FROM firms GROUP BY version;
  
  -- Analyze household employment over time
  SELECT step, COUNT(*) FROM households 
  WHERE employer != 'None' GROUP BY step;
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

### Changed - Simulation Runner Updates

#### Enhanced Output Display
- Version information in all displays
- Separate tracking of firm vs state employment
- Long format data collection during simulation
- Database import statistics at end

#### File Naming Convention
```
firms_checkpoint50_seed123456_20260205_143022.csv
households_checkpoint50_seed123456_20260205_143022.csv
summary_seed123456_20260205_143022.txt
```

#### Summary File Updates
- Includes CSV format description
- Row count information (firms × steps, households × steps)
- Version and seed for reference
- Database import instructions

### Technical Details

#### Version Extraction
```python
def get_version():
    # Reads CHANGELOG.md
    # Regex: ## [5.1.0] or ## Version 5.1.0
    # Returns "v5.1.0"
    # Default: "v5.0"
```

#### Data Collection Example
```python
# Old (wide format)
step_data = {
    'step': 1,
    'firm_0_price': 25.0, 'firm_0_wage': 20.0,
    'firm_1_price': 30.0, 'firm_1_wage': 18.0,
    'hh_0_money': 120.0, 'hh_1_money': 115.0
}

# New (long format)
firms_data.append({
    'version': 'v5.1.0', 'seed': 123456,
    'step': 1, 'firm_id': 0,
    'price': 25.0, 'wage': 20.0, ...
})
firms_data.append({
    'version': 'v5.1.0', 'seed': 123456,
    'step': 1, 'firm_id': 1,
    'price': 30.0, 'wage': 18.0, ...
})
households_data.append({
    'version': 'v5.1.0', 'seed': 123456,
    'step': 1, 'household_id': 0,
    'money': 120.0, ...
})
```

### Impact on Workflows

#### For Data Analysis
- Import CSVs directly to SQL database
- Use pandas for immediate analysis:
  ```python
  firms = pd.read_csv('firms_checkpoint50_seed123456.csv')
  households = pd.read_csv('households_checkpoint50_seed123456.csv')
  
  # Merge for analysis
  merged = firms.merge(households, on=['version', 'seed', 'step'])
  ```

#### For Multi-Simulation Studies
- Run multiple simulations with different seeds
- Append all data to single database
- Query across all simulations for statistical significance
- Compare model versions side-by-side

#### For Reproducibility
- Every row tagged with version and seed
- Complete traceability of simulation origins
- Easy to reproduce specific simulations
- Version control friendly (text-based CSVs)

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

#### Training Configuration
- **Iterations**: 10 → **50** (more thorough training)
- **Batch Size**: 400 → **4000** (better learning stability)
- **Minibatch Size**: 128 → **256** (improved gradient estimates)
- **Workers**: 2 → **4** (more diverse experiences)

### Fixed - Windows Compatibility

#### UTF-8 Encoding Issues
- **Problem**: Windows default encoding (CP1252) couldn't read config.yaml with Unicode characters
- **Solution**: Added explicit `encoding='utf-8'` to YAML file loading
- **Files Fixed**:
  - `train.py`: Line 34
  - `env/economy_env.py`: Line 23
- Now works on Windows, macOS, and Linux

#### Console Output Cleanup
- **Reduced Verbose Logging**:
  - Suppressed Ray initialization logs
  - Disabled RLlib verbose output
  - Hidden checkpoint saving details
  - Filtered deprecation warnings
- **Clean Training Output**:
  ```
  VWL SIMULATION - TRAINING
  Fresh training: 50 iterations
  Environment: 10 firms, 50 households
  
  Iter   Reward       Min        Max        EpLen
  1      -5.23        -8.50      -2.10      100
  ...
  ```

### Technical Details

#### Action Space Structure
```python
MultiDiscrete([5, 5, 3, 2, 3])
# [price_adjust, wage_adjust, marketing, quality, capacity]
```

#### Reward Function
```python
reward = (profit / 100.0) + capital_bonus
# capital_bonus = min(capital / 1000.0, 5.0) if capital > 0
# Bankrupt firms: -20 penalty
```

#### Observation Vector (20 features)
```python
[
  # Own state (7)
  price, wage, employees, inventory, capital/100, quality, marketing,
  # Market stats (6)
  market_avg_price, market_min_price, market_max_price,
  market_avg_wage, market_min_wage, market_max_wage,
  # Aggregates (4)
  total_employed, unemployment_rate, avg_household_money/10, avg_household_skill,
  # Meta (3)
  profit_last_step/100, competitors_alive, timestep/max_steps
]
```

#### Economic Formulas
- **Production**: `employees × 6.0 × (0.5 + avg_skill × 0.5)`
- **Utility**: `(quality × 0.5 + marketing × 0.3) / (price × 1.0)`
- **Capital**: Updated each step with `revenue - costs - wages`

### Performance Impact
- Training is now more complex but realistic
- Expected convergence: 30-50 iterations
- Firms must learn:
  - Competitive pricing strategies
  - Wage optimization for skill acquisition  
  - Strategic investments (marketing/quality)
  - Capital management to avoid bankruptcy
  - Capacity planning based on demand

---

## [4.0.0] - 2026-02-02

### Added
- **Expanded Action Space** (5 discrete levels per parameter)
  - Price adjustments: -10%, -5%, 0%, +5%, +10% (previously only 3 levels)
  - Wage adjustments: -10%, -5%, 0%, +5%, +10%
  - Total action combinations increased from 9 to 25
  - More granular control for better learning and strategy development
  - Hardcoded in `env/economy_env.py` for consistency

- **Console Simulation Runner** (`run_simulation.py`)
  - Interactive checkpoint selection from trained models
  - Two simulation modes:
    - **Seed Mode**: Provide seed → automatic initialization with default ranges
    - **Manual Mode**: Specify all ranges → system calculates matching seed
  - Detailed initial state display:
    - All firm parameters (price, wage, max_employees)
    - All household states (money, employer, wage)
  - Detailed final state display with complete simulation results
  - CSV export with full household tracking:
    - Every household's money, employer, and wage per timestep
    - All firm metrics (price, wage, employees, profit)
    - Aggregate statistics (employment rate, average money)
  - Seed-based reproducibility for scientific experiments

- **Enhanced Data Export**
  - `initial_firms_*.csv`: Starting conditions for all firms (with seed)
  - `initial_households_*.csv`: Starting conditions for all households (with seed)
  - `simulation_checkpoint*_seed*_*.csv`: Complete timestep data including individual households
  - `summary_seed*_*.txt`: Simulation summary with reproduction instructions
  - Timestamped filenames for organization
  - Seed included in all exports for full traceability

### Changed
- **Config Cleanup**
  - Removed obsolete `price_adjustment` and `wage_adjustment` parameters
  - Action space now fully defined in code (single source of truth)
  - Added explanatory comments about action space location
  - Kept only configurable economic parameters (bounds, consumption_rate, reward_scale)

- **Simulation Workflow**
  - Seed controls all randomization (max_employees, initial prices/wages, household money)
  - Manual mode generates seed that reproduces exact configuration
  - Environment reset uses explicit seed for full reproducibility
  - Initial values set after reset to preserve seed-based max_employees

### Technical Details
- **Action Space**: `MultiDiscrete([5, 5])` instead of `MultiDiscrete([3, 3])`
- **Adjustment Mapping**:
  ```python
  0: -10%, 1: -5%, 2: 0%, 3: +5%, 4: +10%
  ```
- **Seed Workflow**:
  - Seed provided → use default ranges from config.yaml
  - No seed → ask for ranges, generate reproducible seed
  - All randomization (environment + numpy) uses same seed

### Removed
- Hardcoded adjustment rates from config.yaml (now in code)
- Confusing parameter inputs (replaced with clear seed-based workflow)

---

## [3.0.0] - 2026-02-01

### Added
- **Central Configuration System** (`config.yaml`)
  - All simulation parameters in one file
  - Environment setup (firms, households, steps)
  - Initial value ranges for firms and households
  - Training hyperparameters (PPO settings)
  - Economic parameters (price/wage adjustment rates)
  - Easy modification without code changes

- **Improved Observation Space** (13 features instead of 7)
  - Market aggregates: avg/min/max prices and wages across all firms
  - Employment statistics: total employed, unemployment rate
  - Better scalability for multiple firms (not limited to 2)
  - Supports variable number of competing firms

- **Flexible Training System**
  - Command-line arguments override config.yaml defaults
  - Examples:
    - `python train.py` - use config.yaml defaults
    - `python train.py --n-firms 5 --iterations 100`
    - `python train.py --lr 0.001 --num-workers 4`
  - Better progress logging with formatted tables
  - Automatic cleanup of old training data

- **Enhanced Dashboard**
  - Loads checkpoint configuration automatically
  - Displays trained environment parameters (firms, households)
  - Configurable initial parameter ranges for simulation
  - Improved chart layouts with Plotly

### Changed
- Environment now loads defaults from `config.yaml`
- Observations include full market statistics instead of simple averages
- Training script more configurable via CLI arguments
- Dashboard matches trained environment structure (no manual firm/household selection)

### Technical Details
- **Observation Vector** (old vs new):
  - Old: `[price, wage, employees, profit, avg_other_price, avg_other_wage, time]`
  - New: `[price, wage, employees, profit, market_avg_price, market_min_price, market_max_price, market_avg_wage, market_min_wage, market_max_wage, total_employed, unemployment_rate, time]`

- **Configuration Hierarchy**:
  1. `config.yaml` defaults
  2. CLI arguments (override config)
  3. Function parameters (override both)

### Removed
- Hard-coded default values scattered across files (now in config.yaml)
- Deprecated backend/ and frontend/ folders (marked for deletion)
- Obsolete simulation scripts (simulate.py, test_env.py, train_simple.py)

---

## [2.0.0] - 2026-01-31

### Changed
- Complete rebuild of project for Ray 2.40.0 compatibility
- Switched to old API stack (enable_rl_module_and_learner=False) for stability
- Reason: New API stack in Ray 2.40.0 has fundamental incompatibilities with MultiAgentEnv interface
- Previous attempts failed due to property vs method conflicts in observation_space/action_space
- Clean slate approach ensures proper architecture from start
- Replaced React frontend with Streamlit for simpler Python-only dashboard

### Added
- SimpleEconomyEnv with Gymnasium MultiAgentEnv base class
- Training script using PPO with stable API configuration
- Environment supports 2 firms competing for 10 households
- Firms adjust prices, households choose cheapest provider
- Successfully trains with episode reward mean of 108.75
- Streamlit dashboard with real-time training visualization
- Plotly charts for episode reward and length
- Auto-refresh every 2 seconds
- No npm/JavaScript build required

### Fixed
- Deprecated API parameters: rollouts() replaced with env_runners()
- num_rollout_workers replaced with num_env_runners
- sgd_minibatch_size replaced with minibatch_size
- num_sgd_iter replaced with num_epochs
- Metric extraction from env_runners dict for correct display

### Removed
- React frontend and npm dependencies
- FastAPI backend (not needed with Streamlit)

---

## [1.0.0] - 2026-01-30

### Added
- Initial multi-agent economy environment
- PPO training implementation
- Web dashboard for visualization

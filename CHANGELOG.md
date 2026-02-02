# Changelog

All notable changes to the VWL Simulation project are documented here.

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

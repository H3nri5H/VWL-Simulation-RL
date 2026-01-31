# Changelog

## [2.0.0] - 2026-01-31

### Changed
- Complete rebuild of project for Ray 2.40.0 compatibility
- Switched to old API stack (enable_rl_module_and_learner=False) for stability
- Reason: New API stack in Ray 2.40.0 has fundamental incompatibilities with MultiAgentEnv interface
- Previous attempts failed due to property vs method conflicts in observation_space/action_space
- Clean slate approach ensures proper architecture from start

### Added
- SimpleEconomyEnv with Gymnasium MultiAgentEnv base class
- Training script using PPO with stable API configuration
- Environment supports 2 firms competing for 10 households
- Firms adjust prices, households choose cheapest provider
- Successfully trains with episode reward mean of 108.75
- FastAPI backend to serve training metrics from checkpoint results
- React frontend dashboard with real-time training visualization
- Recharts integration for episode reward and length charts
- Auto-refresh training metrics every 2 seconds

### Fixed
- Deprecated API parameters: rollouts() replaced with env_runners()
- num_rollout_workers replaced with num_env_runners
- sgd_minibatch_size replaced with minibatch_size
- num_sgd_iter replaced with num_epochs
- Metric extraction from env_runners dict for correct display

## [1.0.0] - 2026-01-30

### Added
- Initial multi-agent economy environment
- PPO training implementation
- Web dashboard for visualization

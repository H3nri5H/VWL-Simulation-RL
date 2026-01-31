# Changelog

## [2.0.0] - 2026-01-31

### Changed
- Complete rebuild of project for Ray 2.40.0 compatibility
- Switched to old API stack (enable_rl_module_and_learner=False) for stability
- Reason: New API stack in Ray 2.40.0 has fundamental incompatibilities with MultiAgentEnv interface
- Previous attempts failed due to property vs method conflicts in observation_space/action_space
- Clean slate approach ensures proper architecture from start

## [1.0.0] - 2026-01-30

### Added
- Initial multi-agent economy environment
- PPO training implementation
- Web dashboard for visualization

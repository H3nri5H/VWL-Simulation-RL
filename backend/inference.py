"""Inference Engine for trained RL policies.

Loads trained models and runs simulations without training.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.economy_env import EconomyEnv
import numpy as np


class SimulationRunner:
    """Runs simulations with trained or random policies."""
    
    def __init__(self, checkpoint_path=None):
        """Initialize simulation runner.
        
        Args:
            checkpoint_path: Path to trained model checkpoint (optional)
                           If None, uses random policy
        """
        self.checkpoint_path = checkpoint_path
        self.algo = None
        
        # Try to load trained model if path provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                # Import Ray only if we actually have a trained model
                from ray.rllib.algorithms.ppo import PPO
                self.algo = PPO.from_checkpoint(checkpoint_path)
                print(f"Loaded trained model from {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Could not load model from {checkpoint_path}: {e}")
                print("Falling back to random policy")
                self.algo = None
    
    def run_simulation(
        self, 
        n_firms=2, 
        n_households=10, 
        max_steps=100,
        start_params=None,
        seed=None
    ):
        """Run complete simulation.
        
        Args:
            n_firms: Number of firms
            n_households: Number of households
            max_steps: Number of quarters to simulate
            start_params: Optional dict with 'prices' and 'wages' lists
            seed: Random seed for reproducibility
            
        Returns:
            dict: Complete history with market data, firm data, household data
        """
        # Create environment
        env = EconomyEnv(
            n_firms=n_firms,
            n_households=n_households,
            max_steps=max_steps
        )
        
        # Reset environment
        obs, info = env.reset(seed=seed)
        
        # Override start parameters if provided
        if start_params:
            if 'prices' in start_params:
                for i, firm_name in enumerate(env.firms.keys()):
                    if i < len(start_params['prices']):
                        env.firms[firm_name]['price'] = start_params['prices'][i]
            
            if 'wages' in start_params:
                for i, firm_name in enumerate(env.firms.keys()):
                    if i < len(start_params['wages']):
                        env.firms[firm_name]['wage'] = start_params['wages'][i]
        
        # Initialize history storage
        history = {
            'metadata': {
                'n_firms': n_firms,
                'n_households': n_households,
                'max_steps': max_steps,
                'model': 'trained' if self.algo else 'random'
            },
            'market': [],
            'firms': {name: [] for name in env.firms.keys()},
            'households': [],
        }
        
        # Run simulation
        step = 0
        while not all(env.terminations.values()) and step < max_steps:
            # Each agent acts
            for agent in env.agents:
                if env.terminations[agent]:
                    continue
                
                obs = env.observe(agent)
                
                # Get action from trained policy or random
                if self.algo:
                    try:
                        action = self.algo.compute_single_action(
                            obs,
                            policy_id='firm_policy'
                        )
                    except:
                        # Fallback to random if inference fails
                        action = env.action_spaces[agent].sample()
                else:
                    # Random policy
                    action = env.action_spaces[agent].sample()
                
                env.step(action)
            
            # After all agents acted, store state
            step += 1
            
            # Market state
            history['market'].append({
                'step': step,
                'gdp': float(env.state['total_gdp']),
                'avg_price': float(env.state['avg_price']),
                'avg_wage': float(env.state['avg_wage']),
            })
            
            # Firm states
            for firm_name, firm in env.firms.items():
                history['firms'][firm_name].append({
                    'step': step,
                    'price': float(firm['price']),
                    'wage': float(firm['wage']),
                    'profit': float(firm['profit']),
                    'inventory': float(firm['inventory']),
                    'capital': float(firm['capital']),
                    'demand': float(firm['demand']),
                    'employees': int(firm['employees']),
                })
            
            # Household states (sample first 10 for performance)
            household_sample = []
            for i, household in enumerate(env.households[:min(10, len(env.households))]):
                household_sample.append({
                    'id': i,
                    'money': float(household['money']),
                    'income': float(household['income']),
                    'employer': household['employer'],
                })
            
            history['households'].append({
                'step': step,
                'sample': household_sample
            })
        
        return history


if __name__ == "__main__":
    # Test inference without trained model
    print("Testing SimulationRunner with random policy...")
    
    runner = SimulationRunner()
    history = runner.run_simulation(n_firms=2, n_households=10, max_steps=20)
    
    print(f"\nSimulation completed!")
    print(f"Steps: {len(history['market'])}")
    print(f"Final GDP: {history['market'][-1]['gdp']:.2f}€")
    print(f"Final Avg Price: {history['market'][-1]['avg_price']:.2f}€")

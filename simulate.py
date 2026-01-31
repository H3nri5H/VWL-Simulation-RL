import json
import numpy as np
from pathlib import Path
from ray.rllib.algorithms.ppo import PPO
from env.economy_env import SimpleEconomyEnv


def run_simulation(config):
    """
    Run a simulation with a trained checkpoint.
    
    Args:
        config: dict with keys:
            - checkpoint_path: path to checkpoint
            - n_firms: number of firms
            - n_households: number of households
            - n_steps: simulation steps
            - firm_configs: initial firm settings
    
    Returns:
        dict with simulation results
    """
    
    print(f"Loading checkpoint: {config['checkpoint_path']}")
    
    # Load the trained algorithm
    algo = PPO.from_checkpoint(config['checkpoint_path'])
    
    # Create environment with specified parameters
    env_config = {
        'n_firms': config['n_firms'],
        'n_households': config['n_households'],
        'max_steps': config['n_steps'],
    }
    
    env = SimpleEconomyEnv(env_config)
    
    # Override initial firm values if provided
    obs, info = env.reset()
    
    if 'firm_configs' in config:
        for i, firm_config in enumerate(config['firm_configs']):
            firm_id = f"firm_{i}"
            if firm_id in env.firms:
                env.firms[firm_id]['price'] = firm_config.get('price', 10.0)
                env.firms[firm_id]['wage'] = firm_config.get('wage', 8.0)
        
        # Get new observations after override
        obs = {agent_id: env._get_obs(agent_id) for agent_id in env._agent_ids}
    
    # Storage for results
    firm_history = {f"firm_{i}": [] for i in range(config['n_firms'])}
    household_history = []
    summary_history = []
    
    print(f"Running simulation for {config['n_steps']} steps...")
    
    # Run simulation
    for step in range(config['n_steps']):
        # Get actions from trained policy
        actions = {}
        for agent_id in env._agent_ids:
            action = algo.compute_single_action(obs[agent_id], policy_id="shared_policy")
            actions[agent_id] = action
        
        # Step environment
        obs, rewards, dones, truncated, infos = env.step(actions)
        
        # Record firm data
        for firm_id in env._agent_ids:
            firm = env.firms[firm_id]
            firm_history[firm_id].append({
                'step': step,
                'price': float(firm['price']),
                'wage': float(firm['wage']),
                'employees': int(firm['employees']),
                'profit': float(firm['profit']),
                'revenue': float(firm['revenue']),
                'costs': float(firm['costs']),
            })
        
        # Record household aggregate data
        total_money = sum(h['money'] for h in env.households)
        employed = sum(1 for h in env.households if h['employer'] is not None)
        avg_wage = np.mean([h['wage'] for h in env.households if h['wage'] > 0]) if employed > 0 else 0
        
        household_history.append({
            'step': step,
            'total_money': float(total_money),
            'employed': int(employed),
            'unemployed': int(len(env.households) - employed),
            'avg_wage': float(avg_wage),
        })
        
        # Summary statistics
        total_profit = sum(firm['profit'] for firm in env.firms.values())
        avg_price = np.mean([firm['price'] for firm in env.firms.values()])
        
        summary_history.append({
            'step': step,
            'total_profit': float(total_profit),
            'avg_price': float(avg_price),
            'total_money': float(total_money),
            'employment_rate': float(employed / len(env.households)),
        })
        
        if step % 20 == 0:
            print(f"  Step {step}/{config['n_steps']} - Avg Profit: {total_profit/len(env.firms):.2f}")
    
    print("Simulation complete!")
    
    # Compile results
    results = {
        'config': config,
        'firms': firm_history,
        'households': household_history,
        'summary': summary_history,
    }
    
    # Save to file
    output_dir = Path("./simulation_results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "latest_simulation.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    algo.stop()
    
    return results


if __name__ == "__main__":
    # Example usage
    config = {
        'checkpoint_path': './checkpoints/checkpoint_000020',
        'n_firms': 2,
        'n_households': 10,
        'n_steps': 100,
        'firm_configs': [
            {'price': 10.0, 'wage': 8.0},
            {'price': 12.0, 'wage': 9.0},
        ]
    }
    
    results = run_simulation(config)

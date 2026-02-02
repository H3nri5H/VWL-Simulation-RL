import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from ray.rllib.algorithms.ppo import PPO
from env.economy_env import SimpleEconomyEnv
import warnings
import numpy as np

warnings.filterwarnings('ignore')

def find_checkpoints():
    """Find all available checkpoints"""
    checkpoint_base = Path("./checkpoints").absolute()
    
    if not checkpoint_base.exists():
        return []
    
    checkpoints = []
    
    for checkpoint_dir in checkpoint_base.iterdir():
        if not checkpoint_dir.is_dir() or not checkpoint_dir.name.startswith('checkpoint_'):
            continue
        
        # Load metadata
        metadata_file = checkpoint_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {'iteration': 0, 'reward_mean': 0.0}
        
        # Load env config
        rllib_config_file = checkpoint_dir / "rllib_checkpoint.json"
        if rllib_config_file.exists():
            with open(rllib_config_file, 'r') as f:
                rllib_config = json.load(f)
                env_config = rllib_config.get('env_config', {})
        else:
            env_config = {'n_firms': 2, 'n_households': 10, 'max_steps': 100}
        
        checkpoints.append({
            'path': str(checkpoint_dir),
            'iteration': metadata.get('iteration', 0),
            'reward': metadata.get('reward_mean', 0.0),
            'n_firms': env_config.get('n_firms', 2),
            'n_households': env_config.get('n_households', 10),
            'max_steps': env_config.get('max_steps', 100),
        })
    
    checkpoints.sort(key=lambda x: x['iteration'])
    return checkpoints

def select_checkpoint(checkpoints):
    """Interactive checkpoint selection"""
    print("\n" + "="*70)
    print("  AVAILABLE CHECKPOINTS")
    print("="*70)
    
    for i, cp in enumerate(checkpoints):
        print(f"[{i+1}] Iteration {cp['iteration']:<6} | "
              f"Reward: {cp['reward']:<8.2f} | "
              f"Firms: {cp['n_firms']:<2} | "
              f"Households: {cp['n_households']:<3}")
    
    while True:
        try:
            choice = int(input("\nSelect checkpoint number: ")) - 1
            if 0 <= choice < len(checkpoints):
                return checkpoints[choice]
            print("Invalid choice!")
        except ValueError:
            print("Please enter a number!")

def get_simulation_config(checkpoint):
    """Get simulation parameters from user"""
    print("\n" + "="*70)
    print("  SIMULATION CONFIGURATION")
    print("="*70)
    
    print(f"\nCheckpoint trained with:")
    print(f"  - Firms: {checkpoint['n_firms']}")
    print(f"  - Households: {checkpoint['n_households']}")
    print(f"  - Steps: {checkpoint['max_steps']}")
    
    config = {}
    
    # Number of steps
    default_steps = checkpoint['max_steps']
    steps_input = input(f"\nNumber of simulation steps [{default_steps}]: ").strip()
    config['max_steps'] = int(steps_input) if steps_input else default_steps
    
    # Initial price range
    print("\nInitial price range per firm:")
    price_min = input("  Min price [8.0]: ").strip()
    price_max = input("  Max price [15.0]: ").strip()
    config['price_range'] = (
        float(price_min) if price_min else 8.0,
        float(price_max) if price_max else 15.0
    )
    
    # Initial wage range
    print("\nInitial wage range per firm:")
    wage_min = input("  Min wage [5.0]: ").strip()
    wage_max = input("  Max wage [12.0]: ").strip()
    config['wage_range'] = (
        float(wage_min) if wage_min else 5.0,
        float(wage_max) if wage_max else 12.0
    )
    
    # Initial money range
    print("\nInitial money range per household:")
    money_min = input("  Min money [40.0]: ").strip()
    money_max = input("  Max money [60.0]: ").strip()
    config['money_range'] = (
        float(money_min) if money_min else 40.0,
        float(money_max) if money_max else 60.0
    )
    
    return config

def display_initial_state(env, checkpoint):
    """Display detailed initial state of all agents"""
    print("\n" + "="*70)
    print("  INITIAL STATE")
    print("="*70)
    
    # Firms
    print("\n[FIRMS]")
    print("-" * 70)
    for i in range(checkpoint['n_firms']):
        firm_id = f"firm_{i}"
        firm = env.firms[firm_id]
        print(f"Firm {i}:")
        print(f"  - Initial Price: {firm['price']:.2f}")
        print(f"  - Initial Wage: {firm['wage']:.2f}")
        print(f"  - Max Employees: {firm['max_employees']}")
        print(f"  - Current Employees: {firm['employees']}")
        print()
    
    # Households
    print("\n[HOUSEHOLDS]")
    print("-" * 70)
    for i, hh in enumerate(env.households):
        employer_str = hh['employer'] if hh['employer'] else 'Unemployed'
        wage_str = f"{hh['wage']:.2f}" if hh['employer'] else 'N/A'
        print(f"Household {i:2d}: Money={hh['money']:.2f} | "
              f"Employer={employer_str:8s} | Wage={wage_str:>6s}")
    
    print("="*70)

def save_initial_state(env, checkpoint, output_dir):
    """Save initial state to CSV files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Firms initial state
    firms_data = []
    for i in range(checkpoint['n_firms']):
        firm_id = f"firm_{i}"
        firm = env.firms[firm_id]
        firms_data.append({
            'firm_id': i,
            'initial_price': firm['price'],
            'initial_wage': firm['wage'],
            'max_employees': firm['max_employees']
        })
    
    firms_df = pd.DataFrame(firms_data)
    firms_file = output_dir / f"initial_firms_{timestamp}.csv"
    firms_df.to_csv(firms_file, index=False)
    
    # Households initial state
    households_data = []
    for i, hh in enumerate(env.households):
        households_data.append({
            'household_id': i,
            'initial_money': hh['money'],
            'initial_employer': hh['employer'] if hh['employer'] else 'None',
            'initial_wage': hh['wage']
        })
    
    households_df = pd.DataFrame(households_data)
    households_file = output_dir / f"initial_households_{timestamp}.csv"
    households_df.to_csv(households_file, index=False)
    
    print(f"\n✅ Initial state saved:")
    print(f"   - Firms: {firms_file}")
    print(f"   - Households: {households_file}")

def display_final_state(env, checkpoint):
    """Display detailed final state of all agents"""
    print("\n" + "="*70)
    print("  FINAL STATE")
    print("="*70)
    
    # Firms
    print("\n[FIRMS]")
    print("-" * 70)
    for i in range(checkpoint['n_firms']):
        firm_id = f"firm_{i}"
        firm = env.firms[firm_id]
        print(f"Firm {i}:")
        print(f"  - Final Price: {firm['price']:.2f}")
        print(f"  - Final Wage: {firm['wage']:.2f}")
        print(f"  - Employees: {firm['employees']}/{firm['max_employees']}")
        print(f"  - Final Profit: {firm['profit']:.2f}")
        print(f"  - Revenue: {firm['revenue']:.2f}")
        print(f"  - Costs: {firm['costs']:.2f}")
        print()
    
    # Households
    print("\n[HOUSEHOLDS]")
    print("-" * 70)
    employed_count = 0
    total_money = 0.0
    
    for i, hh in enumerate(env.households):
        employer_str = hh['employer'] if hh['employer'] else 'Unemployed'
        wage_str = f"{hh['wage']:.2f}" if hh['employer'] else 'N/A'
        print(f"Household {i:2d}: Money={hh['money']:.2f} | "
              f"Employer={employer_str:8s} | Wage={wage_str:>6s}")
        
        if hh['employer']:
            employed_count += 1
        total_money += hh['money']
    
    print("\n[SUMMARY]")
    print("-" * 70)
    print(f"Employment Rate: {(employed_count/len(env.households)*100):.1f}%")
    print(f"Average Household Money: {total_money/len(env.households):.2f}")
    print("="*70)

def run_simulation(checkpoint, config):
    """Run the simulation and collect data"""
    print("\n" + "="*70)
    print("  RUNNING SIMULATION")
    print("="*70)
    
    # Create results directory
    results_dir = Path("./simulation_results")
    results_dir.mkdir(exist_ok=True)
    
    # Setup environment
    env_config = {
        'n_firms': checkpoint['n_firms'],
        'n_households': checkpoint['n_households'],
        'max_steps': config['max_steps'],
    }
    
    # Load trained model
    print("\nLoading model...")
    algo = PPO.from_checkpoint(checkpoint['path'])
    env = SimpleEconomyEnv(env_config)
    
    # Reset environment with RANDOM seed (randomizes max_employees!)
    random_seed = np.random.randint(0, 1000000)
    obs, info = env.reset(seed=random_seed)
    
    # NOW set initial values AFTER reset (so max_employees is already randomized)
    for i in range(checkpoint['n_firms']):
        firm_id = f"firm_{i}"
        env.firms[firm_id]['price'] = np.random.uniform(*config['price_range'])
        env.firms[firm_id]['wage'] = np.random.uniform(*config['wage_range'])
        # max_employees is already set by reset() - don't touch it!
    
    for hh in env.households:
        hh['money'] = np.random.uniform(*config['money_range'])
    
    # Display and save initial state
    display_initial_state(env, checkpoint)
    save_initial_state(env, checkpoint, results_dir)
    
    input("\nPress ENTER to start simulation...")
    
    # Storage for all steps
    all_data = []
    
    print("\nSimulation running...\n")
    print("="*70)
    
    done = False
    step = 0
    
    while not done and step < config['max_steps']:
        # Get actions from model
        actions = {}
        for agent_id in obs.keys():
            action, _, _ = algo.get_policy("shared_policy").compute_single_action(obs[agent_id])
            actions[agent_id] = action
        
        # Step environment
        obs, rewards, dones, truncated, info = env.step(actions)
        
        # Collect data for this step
        step_data = {'step': step + 1}
        
        # Firm data
        for i in range(checkpoint['n_firms']):
            firm_id = f"firm_{i}"
            firm = env.firms[firm_id]
            step_data[f'firm_{i}_price'] = firm['price']
            step_data[f'firm_{i}_wage'] = firm['wage']
            step_data[f'firm_{i}_employees'] = firm['employees']
            step_data[f'firm_{i}_max_employees'] = firm['max_employees']
            step_data[f'firm_{i}_profit'] = firm['profit']
        
        # Individual household data
        for i, hh in enumerate(env.households):
            step_data[f'hh_{i}_money'] = hh['money']
            step_data[f'hh_{i}_employer'] = hh['employer'] if hh['employer'] else 'None'
            step_data[f'hh_{i}_wage'] = hh['wage']
        
        # Aggregate stats
        total_money = sum(hh['money'] for hh in env.households)
        employed = sum(1 for hh in env.households if hh['employer'] is not None)
        step_data['avg_household_money'] = total_money / len(env.households)
        step_data['employment_rate'] = (employed / len(env.households)) * 100
        
        all_data.append(step_data)
        
        # Print table every 10 steps or first/last step
        if step % 10 == 0 or step == 0 or step == config['max_steps'] - 1:
            print(f"\nStep {step + 1}/{config['max_steps']}")
            print("-" * 70)
            
            # Firm table
            for i in range(checkpoint['n_firms']):
                firm_id = f"firm_{i}"
                firm = env.firms[firm_id]
                print(f"Firm {i}: Price={firm['price']:.2f} | "
                      f"Wage={firm['wage']:.2f} | "
                      f"Employees={firm['employees']}/{firm['max_employees']} | "
                      f"Profit={firm['profit']:.2f}")
            
            # Household stats
            print(f"\nHouseholds: Avg Money={step_data['avg_household_money']:.2f} | "
                  f"Employment={step_data['employment_rate']:.1f}%")
            print("=" * 70)
        
        done = dones.get('__all__', False)
        step += 1
    
    # Display final state
    display_final_state(env, checkpoint)
    
    algo.stop()
    
    return pd.DataFrame(all_data), results_dir

def save_results(df, checkpoint, results_dir):
    """Save results to CSV"""
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simulation_checkpoint{checkpoint['iteration']}_{timestamp}.csv"
    filepath = results_dir / filename
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    
    print(f"\n✅ Results saved to: {filepath}")
    
    # Also save summary
    summary_file = results_dir / f"summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Simulation Summary\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Checkpoint: Iteration {checkpoint['iteration']}\n")
        f.write(f"Firms: {checkpoint['n_firms']}\n")
        f.write(f"Households: {checkpoint['n_households']}\n")
        f.write(f"Steps: {len(df)}\n\n")
        f.write(f"Final Results:\n")
        f.write(f"-"*50 + "\n")
        f.write(df.tail(1).to_string(index=False))
    
    print(f"✅ Summary saved to: {summary_file}")
    
    return filepath

def main():
    print("\n" + "="*70)
    print("  VWL SIMULATION - CONSOLE RUNNER")
    print("="*70)
    
    # Find checkpoints
    checkpoints = find_checkpoints()
    
    if not checkpoints:
        print("\n❌ No checkpoints found!")
        print("\nRun training first: python train.py\n")
        return
    
    # Select checkpoint
    checkpoint = select_checkpoint(checkpoints)
    
    # Get configuration
    config = get_simulation_config(checkpoint)
    
    # Run simulation
    df, results_dir = run_simulation(checkpoint, config)
    
    # Save results
    save_results(df, checkpoint, results_dir)
    
    print("\n" + "="*70)
    print("  SIMULATION COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

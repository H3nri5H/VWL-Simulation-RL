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
        
        # Load env config from rllib_checkpoint.json
        rllib_config_file = checkpoint_dir / "rllib_checkpoint.json"
        env_config = None
        
        if rllib_config_file.exists():
            try:
                with open(rllib_config_file, 'r') as f:
                    rllib_config = json.load(f)
                    # Extract env_config from checkpoint
                    env_config = rllib_config.get('env_config', {})
            except:
                pass
        
        # Fallback to defaults if not found
        if env_config is None:
            env_config = {'n_firms': 2, 'n_households': 10, 'max_steps': 100}
        
        checkpoints.append({
            'path': str(checkpoint_dir.absolute()),  # Absolute path!
            'iteration': metadata.get('iteration', 0),
            'reward': metadata.get('reward_mean', 0.0),
            'n_firms': env_config.get('n_firms', 2),
            'n_households': env_config.get('n_households', 10),
            'max_steps': env_config.get('max_steps', 100),
            'env_config': env_config,  # Store full config
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
    
    # Ask for seed FIRST
    print("\n" + "="*70)
    print("[SIMULATION MODE]")
    print("="*70)
    print("\nOption A: Provide a SEED")
    print("  → Seed controls ALL initial values (max_employees, prices, wages, money)")
    print("  → Perfect for reproducing exact simulations")
    print("  → Uses default ranges from config.yaml")
    print("\nOption B: Leave EMPTY (press ENTER)")
    print("  → You specify all parameter ranges manually")
    print("  → System calculates matching seed at the end")
    print("  → Full control over initial conditions")
    
    seed_input = input("\nEnter seed (integer) or ENTER for manual control: ").strip()
    
    if seed_input:
        # MODE A: Seed provided - use defaults
        try:
            config['seed'] = int(seed_input)
            config['seed_mode'] = 'provided'
            print(f"\n✅ Mode A: Using seed {config['seed']} with default ranges")
            
            # Use default ranges from config.yaml
            config['price_range'] = (8.0, 15.0)
            config['wage_range'] = (5.0, 12.0)
            config['money_range'] = (100.0, 150.0)
            
        except ValueError:
            print("Invalid seed, switching to manual mode.")
            config['seed_mode'] = 'manual'
    else:
        # MODE B: Manual configuration
        config['seed_mode'] = 'manual'
        print("\n✅ Mode B: Manual configuration (seed will be calculated)")
    
    # Number of steps (always ask)
    default_steps = checkpoint['max_steps']
    steps_input = input(f"\nNumber of simulation steps [{default_steps}]: ").strip()
    config['max_steps'] = int(steps_input) if steps_input else default_steps
    
    # If manual mode, ask for ranges
    if config['seed_mode'] == 'manual':
        print("\n" + "="*70)
        print("[MANUAL PARAMETER CONFIGURATION]")
        print("="*70)
        
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
        money_min = input("  Min money [100.0]: ").strip()
        money_max = input("  Max money [150.0]: ").strip()
        config['money_range'] = (
            float(money_min) if money_min else 100.0,
            float(money_max) if money_max else 150.0
        )
        
        # Generate random seed for this configuration
        config['seed'] = np.random.randint(0, 1000000)
        print(f"\n✅ Generated seed: {config['seed']}")
        print("   (This seed will reproduce these exact ranges)")
    
    return config

def display_initial_state(env, checkpoint, config):
    """Display detailed initial state of all agents"""
    print("\n" + "="*70)
    print("  INITIAL STATE")
    print("="*70)
    print(f"\n[SEED: {config['seed']}]")
    if config['seed_mode'] == 'provided':
        print("(Seed controlled all initial values)")
    else:
        print("(Seed generated to match your manual configuration)")
    print("-" * 70)
    
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

def save_initial_state(env, checkpoint, config, output_dir):
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
            'max_employees': firm['max_employees'],
            'seed': config['seed'],
            'seed_mode': config['seed_mode']
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
            'initial_wage': hh['wage'],
            'seed': config['seed'],
            'seed_mode': config['seed_mode']
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
    
    # Use checkpoint's environment config!
    env_config = checkpoint['env_config'].copy()
    env_config['max_steps'] = config['max_steps']  # Override max_steps from user
    
    print(f"\nEnvironment config:")
    print(f"  - Firms: {env_config['n_firms']}")
    print(f"  - Households: {env_config['n_households']}")
    print(f"  - Max Steps: {env_config['max_steps']}")
    
    # Load trained model
    print(f"\nLoading model from: {checkpoint['path']}")
    algo = PPO.from_checkpoint(checkpoint['path'])
    
    # Create environment with checkpoint's config
    env = SimpleEconomyEnv(env_config)
    
    # Reset environment with specified seed
    obs, info = env.reset(seed=config['seed'])
    
    # Set initial values based on seed and ranges
    np.random.seed(config['seed'])
    
    for i in range(checkpoint['n_firms']):
        firm_id = f"firm_{i}"
        env.firms[firm_id]['price'] = np.random.uniform(*config['price_range'])
        env.firms[firm_id]['wage'] = np.random.uniform(*config['wage_range'])
    
    for hh in env.households:
        hh['money'] = np.random.uniform(*config['money_range'])
    
    # Display and save initial state
    display_initial_state(env, checkpoint, config)
    save_initial_state(env, checkpoint, config, results_dir)
    
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
        step_data = {'step': step + 1, 'seed': config['seed']}
        
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

def save_results(df, checkpoint, config, results_dir):
    """Save results to CSV"""
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simulation_checkpoint{checkpoint['iteration']}_seed{config['seed']}_{timestamp}.csv"
    filepath = results_dir / filename
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    
    print(f"\n✅ Results saved to: {filepath}")
    
    # Also save summary
    summary_file = results_dir / f"summary_seed{config['seed']}_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Simulation Summary\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Checkpoint: Iteration {checkpoint['iteration']}\n")
        f.write(f"Random Seed: {config['seed']}\n")
        f.write(f"Seed Mode: {config['seed_mode']}\n")
        if config['seed_mode'] == 'provided':
            f.write(f"  (Seed controlled all initial values)\n")
        else:
            f.write(f"  (Seed generated to match manual configuration)\n")
        f.write(f"\nFirms: {checkpoint['n_firms']}\n")
        f.write(f"Households: {checkpoint['n_households']}\n")
        f.write(f"Steps: {len(df)}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  - Price Range: {config['price_range']}\n")
        f.write(f"  - Wage Range: {config['wage_range']}\n")
        f.write(f"  - Money Range: {config['money_range']}\n\n")
        f.write(f"Final Results:\n")
        f.write(f"-"*50 + "\n")
        f.write(df.tail(1).to_string(index=False))
        f.write(f"\n\n")
        f.write(f"To reproduce this exact simulation:\n")
        f.write(f"  1. Use checkpoint iteration {checkpoint['iteration']}\n")
        f.write(f"  2. Enter seed: {config['seed']}\n")
        if config['seed_mode'] == 'manual':
            f.write(f"     (Seed will use ranges: {config['price_range']}, {config['wage_range']}, {config['money_range']})\n")
    
    print(f"✅ Summary saved to: {summary_file}")
    
    if config['seed_mode'] == 'provided':
        print(f"\n♻️  To reproduce: Use seed {config['seed']} (with default ranges)")
    else:
        print(f"\n♻️  To reproduce: Use seed {config['seed']}")
        print(f"   This seed produces your exact configuration:")
        print(f"   - Price Range: {config['price_range']}")
        print(f"   - Wage Range: {config['wage_range']}")
        print(f"   - Money Range: {config['money_range']}")
    
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
    save_results(df, checkpoint, config, results_dir)
    
    print("\n" + "="*70)
    print("  SIMULATION COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

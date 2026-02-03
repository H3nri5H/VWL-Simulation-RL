import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from ray.rllib.algorithms.ppo import PPO
from env.economy_env import SimpleEconomyEnv
import warnings
import numpy as np

warnings.filterwarnings('ignore')
os.environ['RAY_DEDUP_LOGS'] = '0'

# Suppress Ray output
import logging
logging.getLogger('ray').setLevel(logging.ERROR)

def find_checkpoints():
    """Find all available checkpoints"""
    checkpoint_base = Path("./checkpoints").absolute()
    
    if not checkpoint_base.exists():
        return []
    
    checkpoints = []
    
    for checkpoint_dir in checkpoint_base.iterdir():
        if not checkpoint_dir.is_dir() or not checkpoint_dir.name.startswith('checkpoint_'):
            continue
        
        metadata_file = checkpoint_dir / "metadata.json"
        metadata = None
        env_config = None
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    env_config = metadata.get('env_config', None)
            except Exception as e:
                print(f"Warning: Could not load metadata from {checkpoint_dir.name}: {e}")
        
        if metadata is None:
            metadata = {'iteration': 0, 'reward_mean': 0.0}
        
        if env_config is None:
            rllib_config_file = checkpoint_dir / "rllib_checkpoint.json"
            if rllib_config_file.exists():
                try:
                    with open(rllib_config_file, 'r') as f:
                        rllib_config = json.load(f)
                        env_config = rllib_config.get('env_config', None)
                except:
                    pass
        
        if env_config is None or not isinstance(env_config, dict):
            env_config = {'n_firms': 10, 'n_households': 50, 'max_steps': 100}
        
        env_config.setdefault('n_firms', 10)
        env_config.setdefault('n_households', 50)
        env_config.setdefault('max_steps', 100)
        
        checkpoints.append({
            'path': str(checkpoint_dir.absolute()),
            'iteration': metadata.get('iteration', 0),
            'reward': metadata.get('reward_mean', 0.0),
            'n_firms': env_config['n_firms'],
            'n_households': env_config['n_households'],
            'max_steps': env_config['max_steps'],
            'env_config': env_config,
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
    
    print("\n" + "="*70)
    print("[SIMULATION MODE]")
    print("="*70)
    print("\nOption A: Provide a SEED")
    print("  -> Seed controls ALL randomization (skills, capital, quality, etc.)")
    print("  -> Uses default ranges from config.yaml")
    print("\nOption B: Leave EMPTY")
    print("  -> Random seed will be generated")
    
    seed_input = input("\nEnter seed (integer) or ENTER for random: ").strip()
    
    if seed_input:
        try:
            config['seed'] = int(seed_input)
            config['seed_mode'] = 'provided'
            print(f"\nUsing seed {config['seed']}")
        except ValueError:
            print("Invalid seed, using random.")
            config['seed'] = np.random.randint(0, 1000000)
            config['seed_mode'] = 'random'
    else:
        config['seed'] = np.random.randint(0, 1000000)
        config['seed_mode'] = 'random'
        print(f"\nGenerated random seed: {config['seed']}")
    
    default_steps = checkpoint['max_steps']
    steps_input = input(f"\nNumber of simulation steps [{default_steps}]: ").strip()
    config['max_steps'] = int(steps_input) if steps_input else default_steps
    
    return config

def display_initial_state(env, checkpoint, config):
    """Display detailed initial state with NEW v5.0 features"""
    print("\n" + "="*70)
    print("  INITIAL STATE (v5.0 - Enhanced Economy)")
    print("="*70)
    print(f"\n[SEED: {config['seed']} | Mode: {config['seed_mode']}]")
    print("-" * 70)
    
    # Firms with NEW features
    print("\n[FIRMS]")
    print("-" * 70)
    for i in range(checkpoint['n_firms']):
        firm_id = f"firm_{i}"
        firm = env.firms[firm_id]
        print(f"Firm {i}:")
        print(f"  - Price: {firm['price']:.2f} | Wage: {firm['wage']:.2f}")
        print(f"  - Capital: {firm['capital']:.2f} | Capacity: {firm['max_employees']} employees")
        print(f"  - Quality: {firm['quality']:.2f} | Marketing: {firm['marketing']:.2f}")
        print(f"  - Employees: {firm['employees']} | Bankrupt: {firm['bankrupt']}")
        print()
    
    # Households with NEW skill levels
    print("\n[HOUSEHOLDS] (showing first 10 of {})\n".format(checkpoint['n_households']))
    print(f"{'ID':<4} {'Money':<8} {'Skill':<6} {'Wealth':<8} {'Employer':<12} {'Wage':<6}")
    print("-" * 70)
    for i, hh in enumerate(env.households[:10]):
        employer_str = hh['employer'] if hh['employer'] else 'Unemployed'
        wage_str = f"{hh['wage']:.2f}" if hh['employer'] else 'N/A'
        print(f"{i:<4} {hh['money']:<8.2f} {hh['skill_level']:<6.2f} "
              f"{hh['wealth_type']:<8} {employer_str:<12} {wage_str:<6}")
    
    if checkpoint['n_households'] > 10:
        print(f"... and {checkpoint['n_households'] - 10} more households")
    
    # Statistics
    employed = sum(1 for hh in env.households if hh['employer'])
    avg_skill = np.mean([hh['skill_level'] for hh in env.households])
    avg_money = np.mean([hh['money'] for hh in env.households])
    
    print("\n[INITIAL STATISTICS]")
    print("-" * 70)
    print(f"Employment Rate: {(employed/checkpoint['n_households']*100):.1f}%")
    print(f"Average Skill Level: {avg_skill:.2f}")
    print(f"Average Household Money: {avg_money:.2f}")
    print(f"Total Market Capital: {sum(f['capital'] for f in env.firms.values()):.2f}")
    print("="*70)

def save_initial_state(env, checkpoint, config, output_dir):
    """Save initial state with NEW v5.0 features"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Firms with NEW features
    firms_data = []
    for i in range(checkpoint['n_firms']):
        firm_id = f"firm_{i}"
        firm = env.firms[firm_id]
        firms_data.append({
            'firm_id': i,
            'initial_price': firm['price'],
            'initial_wage': firm['wage'],
            'initial_capital': firm['capital'],
            'max_employees': firm['max_employees'],
            'initial_quality': firm['quality'],
            'initial_marketing': firm['marketing'],
            'seed': config['seed'],
            'seed_mode': config['seed_mode']
        })
    
    firms_df = pd.DataFrame(firms_data)
    firms_file = output_dir / f"initial_firms_{timestamp}.csv"
    firms_df.to_csv(firms_file, index=False)
    
    # Households with NEW skill levels
    households_data = []
    for i, hh in enumerate(env.households):
        households_data.append({
            'household_id': i,
            'initial_money': hh['money'],
            'skill_level': hh['skill_level'],
            'wealth_type': hh['wealth_type'],
            'initial_employer': hh['employer'] if hh['employer'] else 'None',
            'initial_wage': hh['wage'],
            'seed': config['seed'],
            'seed_mode': config['seed_mode']
        })
    
    households_df = pd.DataFrame(households_data)
    households_file = output_dir / f"initial_households_{timestamp}.csv"
    households_df.to_csv(households_file, index=False)
    
    print(f"\nInitial state saved:")
    print(f"  - Firms: {firms_file}")
    print(f"  - Households: {households_file}")

def display_final_state(env, checkpoint):
    """Display final state with NEW v5.0 features"""
    print("\n" + "="*70)
    print("  FINAL STATE")
    print("="*70)
    
    # Firms with NEW metrics
    print("\n[FIRMS]")
    print("-" * 70)
    bankruptcies = 0
    for i in range(checkpoint['n_firms']):
        firm_id = f"firm_{i}"
        firm = env.firms[firm_id]
        status = "BANKRUPT" if firm['bankrupt'] else "Active"
        if firm['bankrupt']:
            bankruptcies += 1
        
        print(f"Firm {i} [{status}]:")
        print(f"  - Price: {firm['price']:.2f} | Wage: {firm['wage']:.2f}")
        print(f"  - Capital: {firm['capital']:.2f} | Employees: {firm['employees']}/{firm['max_employees']}")
        print(f"  - Quality: {firm['quality']:.2f} | Marketing: {firm['marketing']:.2f}")
        print(f"  - Profit: {firm['profit']:.2f} | Revenue: {firm['revenue']:.2f} | Costs: {firm['costs']:.2f}")
        print(f"  - Inventory: {firm['inventory']:.2f}")
        print()
    
    # Households summary
    print("\n[HOUSEHOLDS SUMMARY]")
    print("-" * 70)
    employed = sum(1 for hh in env.households if hh['employer'])
    total_money = sum(hh['money'] for hh in env.households)
    avg_money = total_money / len(env.households)
    
    # Skill distribution
    skills = [hh['skill_level'] for hh in env.households]
    avg_skill = np.mean(skills)
    
    # Wealth type distribution
    wealth_counts = {'low': 0, 'medium': 0, 'high': 0}
    for hh in env.households:
        wealth_counts[hh['wealth_type']] += 1
    
    print(f"Total Households: {len(env.households)}")
    print(f"Employed: {employed} ({(employed/len(env.households)*100):.1f}%)")
    print(f"Average Money: {avg_money:.2f}")
    print(f"Average Skill: {avg_skill:.2f}")
    print(f"Wealth Distribution: Low={wealth_counts['low']}, "
          f"Medium={wealth_counts['medium']}, High={wealth_counts['high']}")
    
    print("\n[FINAL STATISTICS]")
    print("-" * 70)
    print(f"Bankruptcies: {bankruptcies}/{checkpoint['n_firms']}")
    print(f"Active Firms: {checkpoint['n_firms'] - bankruptcies}")
    print(f"Employment Rate: {(employed/len(env.households)*100):.1f}%")
    print(f"Total Household Money: {total_money:.2f}")
    print(f"Total Firm Capital: {sum(f['capital'] for f in env.firms.values()):.2f}")
    print("="*70)

def run_simulation(checkpoint, config):
    """Run simulation with NEW v5.0 features"""
    print("\n" + "="*70)
    print("  RUNNING SIMULATION (v5.0)")
    print("="*70)
    
    results_dir = Path("./simulation_results")
    results_dir.mkdir(exist_ok=True)
    
    env_config = checkpoint['env_config'].copy()
    env_config['max_steps'] = config['max_steps']
    
    print(f"\nEnvironment: {env_config['n_firms']} firms, {env_config['n_households']} households, {env_config['max_steps']} steps")
    print(f"Loading model from checkpoint {checkpoint['iteration']}...")
    
    # Suppress Ray loading output
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        algo = PPO.from_checkpoint(checkpoint['path'])
        env = SimpleEconomyEnv(env_config)
        obs, info = env.reset(seed=config['seed'])
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    
    print("Model loaded successfully.")
    
    display_initial_state(env, checkpoint, config)
    save_initial_state(env, checkpoint, config, results_dir)
    
    input("\nPress ENTER to start simulation...")
    
    all_data = []
    
    print("\nSimulation running...\n")
    print("="*70)
    
    done = False
    step = 0
    
    while not done and step < config['max_steps']:
        # Get actions
        actions = {}
        for agent_id in obs.keys():
            action, _, _ = algo.get_policy("shared_policy").compute_single_action(obs[agent_id])
            actions[agent_id] = action
        
        # Step
        obs, rewards, dones, truncated, info = env.step(actions)
        
        # Collect data with NEW features
        step_data = {'step': step + 1, 'seed': config['seed']}
        
        # Firm data with NEW metrics
        bankruptcies = 0
        for i in range(checkpoint['n_firms']):
            firm_id = f"firm_{i}"
            firm = env.firms[firm_id]
            if firm['bankrupt']:
                bankruptcies += 1
            
            step_data[f'firm_{i}_price'] = firm['price']
            step_data[f'firm_{i}_wage'] = firm['wage']
            step_data[f'firm_{i}_capital'] = firm['capital']
            step_data[f'firm_{i}_employees'] = firm['employees']
            step_data[f'firm_{i}_max_employees'] = firm['max_employees']
            step_data[f'firm_{i}_quality'] = firm['quality']
            step_data[f'firm_{i}_marketing'] = firm['marketing']
            step_data[f'firm_{i}_profit'] = firm['profit']
            step_data[f'firm_{i}_revenue'] = firm['revenue']
            step_data[f'firm_{i}_inventory'] = firm['inventory']
            step_data[f'firm_{i}_bankrupt'] = firm['bankrupt']
        
        step_data['bankruptcies'] = bankruptcies
        
        # Household data
        total_money = 0
        employed = 0
        for i, hh in enumerate(env.households):
            step_data[f'hh_{i}_money'] = hh['money']
            step_data[f'hh_{i}_employer'] = hh['employer'] if hh['employer'] else 'None'
            step_data[f'hh_{i}_wage'] = hh['wage']
            total_money += hh['money']
            if hh['employer']:
                employed += 1
        
        step_data['avg_household_money'] = total_money / len(env.households)
        step_data['employment_rate'] = (employed / len(env.households)) * 100
        step_data['total_household_money'] = total_money
        step_data['total_firm_capital'] = sum(f['capital'] for f in env.firms.values())
        
        all_data.append(step_data)
        
        # Print every 10 steps
        if step % 10 == 0 or step == 0 or step == config['max_steps'] - 1:
            print(f"\nStep {step + 1}/{config['max_steps']}")
            print("-" * 70)
            
            for i in range(min(3, checkpoint['n_firms'])):  # Show first 3 firms
                firm_id = f"firm_{i}"
                firm = env.firms[firm_id]
                status = "[BANKRUPT]" if firm['bankrupt'] else ""
                print(f"Firm {i}{status}: Price={firm['price']:.2f} | Wage={firm['wage']:.2f} | "
                      f"Capital={firm['capital']:.2f} | Employees={firm['employees']}/{firm['max_employees']} | "
                      f"Profit={firm['profit']:.2f}")
            
            if checkpoint['n_firms'] > 3:
                print(f"  ... and {checkpoint['n_firms'] - 3} more firms")
            
            print(f"\nMarket: Bankruptcies={bankruptcies} | Employment={step_data['employment_rate']:.1f}% | "
                  f"Avg Money={step_data['avg_household_money']:.2f}")
            print("=" * 70)
        
        done = dones.get('__all__', False)
        step += 1
    
    display_final_state(env, checkpoint)
    
    algo.stop()
    
    return pd.DataFrame(all_data), results_dir

def save_results(df, checkpoint, config, results_dir):
    """Save results with NEW v5.0 data"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simulation_checkpoint{checkpoint['iteration']}_seed{config['seed']}_{timestamp}.csv"
    filepath = results_dir / filename
    
    df.to_csv(filepath, index=False)
    
    print(f"\nResults saved to: {filepath}")
    
    # Summary
    summary_file = results_dir / f"summary_seed{config['seed']}_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"VWL Simulation v5.0 - Summary\n")
        f.write(f"="*70 + "\n\n")
        f.write(f"Checkpoint: Iteration {checkpoint['iteration']}\n")
        f.write(f"Random Seed: {config['seed']} (Mode: {config['seed_mode']})\n")
        f.write(f"Firms: {checkpoint['n_firms']}\n")
        f.write(f"Households: {checkpoint['n_households']}\n")
        f.write(f"Steps: {len(df)}\n\n")
        
        # Final state summary
        last_row = df.iloc[-1]
        f.write(f"Final State:\n")
        f.write(f"-"*70 + "\n")
        f.write(f"Bankruptcies: {int(last_row['bankruptcies'])}/{checkpoint['n_firms']}\n")
        f.write(f"Employment Rate: {last_row['employment_rate']:.1f}%\n")
        f.write(f"Average Household Money: {last_row['avg_household_money']:.2f}\n")
        f.write(f"Total Household Money: {last_row['total_household_money']:.2f}\n")
        f.write(f"Total Firm Capital: {last_row['total_firm_capital']:.2f}\n\n")
        
        f.write(f"To reproduce:\n")
        f.write(f"  1. Use checkpoint iteration {checkpoint['iteration']}\n")
        f.write(f"  2. Enter seed: {config['seed']}\n")
    
    print(f"Summary saved to: {summary_file}")
    print(f"\nTo reproduce: Use seed {config['seed']} with checkpoint {checkpoint['iteration']}")
    
    return filepath

def main():
    print("\n" + "="*70)
    print("  VWL SIMULATION v5.0 - CONSOLE RUNNER")
    print("  Enhanced Economy: Skills | Bankruptcy | Quality | Marketing")
    print("="*70)
    
    checkpoints = find_checkpoints()
    
    if not checkpoints:
        print("\nNo checkpoints found!")
        print("Run training first: python train.py\n")
        return
    
    checkpoint = select_checkpoint(checkpoints)
    config = get_simulation_config(checkpoint)
    df, results_dir = run_simulation(checkpoint, config)
    save_results(df, checkpoint, config, results_dir)
    
    print("\n" + "="*70)
    print("  SIMULATION COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

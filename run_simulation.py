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
import re

warnings.filterwarnings('ignore')
os.environ['RAY_DEDUP_LOGS'] = '0'

# Suppress Ray output
import logging
logging.getLogger('ray').setLevel(logging.ERROR)

def get_version():
    """Extract version from CHANGELOG.md"""
    changelog_path = Path("./CHANGELOG.md")
    if changelog_path.exists():
        try:
            with open(changelog_path, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.search(r'##\s*(?:\[)?(?:Version\s+)?(\d+\.\d+\.\d+)', content)
                if match:
                    return f"v{match.group(1)}"
        except Exception as e:
            print(f"Warning: Could not read CHANGELOG: {e}")
    return "v5.1"  # Default version

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
    
    # Get version from CHANGELOG
    config['version'] = get_version()
    print(f"\nSimulation version: {config['version']}")
    
    return config

def display_initial_state(env, checkpoint, config):
    """Display initial state summary"""
    print("\n" + "="*70)
    print("  INITIAL STATE")
    print("="*70)
    print(f"Version: {config['version']} | Seed: {config['seed']}")
    print("-" * 70)
    
    # Firm statistics
    avg_price = np.mean([f['price'] for f in env.firms.values()])
    avg_wage = np.mean([f['wage'] for f in env.firms.values()])
    avg_capital = np.mean([f['capital'] for f in env.firms.values()])
    avg_quality = np.mean([f['quality'] for f in env.firms.values()])
    avg_marketing = np.mean([f['marketing'] for f in env.firms.values()])
    
    print(f"\nFirms ({checkpoint['n_firms']})")
    print(f"  Avg Price: {avg_price:.2f} | Avg Wage: {avg_wage:.2f}")
    print(f"  Avg Capital: {avg_capital:.2f} | Avg Quality: {avg_quality:.2f}")
    print(f"  Avg Marketing: {avg_marketing:.2f}")
    
    # Household statistics
    employed = sum(1 for hh in env.households if hh['employer'] and hh['employer'] != 'state')
    state_employed = sum(1 for hh in env.households if hh['employer'] == 'state')
    avg_skill = np.mean([hh['skill_level'] for hh in env.households])
    avg_money = np.mean([hh['money'] for hh in env.households])
    
    print(f"\nHouseholds ({checkpoint['n_households']})")
    print(f"  Firm Employed: {employed} ({(employed/checkpoint['n_households']*100):.1f}%)")
    print(f"  State Employed: {state_employed} ({(state_employed/checkpoint['n_households']*100):.1f}%)")
    print(f"  Avg Skill: {avg_skill:.2f} | Avg Money: {avg_money:.2f}")
    print("="*70)

def display_final_state(env, checkpoint, total_steps):
    """Display final state summary"""
    print("\n" + "="*70)
    print("  FINAL STATE")
    print("="*70)
    print(f"Simulation completed: {total_steps} steps")
    print("-" * 70)
    
    # Firm statistics
    bankruptcies = sum(1 for f in env.firms.values() if f['bankrupt'])
    active_firms = [f for f in env.firms.values() if not f['bankrupt']]
    
    if active_firms:
        avg_price = np.mean([f['price'] for f in active_firms])
        avg_wage = np.mean([f['wage'] for f in active_firms])
        avg_capital = np.mean([f['capital'] for f in active_firms])
        avg_quality = np.mean([f['quality'] for f in active_firms])
        avg_marketing = np.mean([f['marketing'] for f in active_firms])
        avg_profit = np.mean([f['profit'] for f in active_firms])
        
        print(f"\nFirms (Active: {len(active_firms)}/{checkpoint['n_firms']})")
        print(f"  Bankruptcies: {bankruptcies}")
        print(f"  Avg Price: {avg_price:.2f} | Avg Wage: {avg_wage:.2f}")
        print(f"  Avg Capital: {avg_capital:.2f} | Avg Profit: {avg_profit:.2f}")
        print(f"  Avg Quality: {avg_quality:.2f} | Avg Marketing: {avg_marketing:.2f}")
    else:
        print(f"\nFirms: ALL BANKRUPT ({bankruptcies}/{checkpoint['n_firms']})")
    
    # Household statistics
    employed = sum(1 for hh in env.households if hh['employer'] and hh['employer'] != 'state')
    state_employed = sum(1 for hh in env.households if hh['employer'] == 'state')
    total_money = sum(hh['money'] for hh in env.households)
    avg_money = total_money / len(env.households)
    
    print(f"\nHouseholds ({checkpoint['n_households']})")
    print(f"  Firm Employed: {employed} ({(employed/checkpoint['n_households']*100):.1f}%)")
    print(f"  State Employed: {state_employed} ({(state_employed/checkpoint['n_households']*100):.1f}%)")
    print(f"  Avg Money: {avg_money:.2f} | Total Money: {total_money:.2f}")
    print("="*70)

def run_simulation(checkpoint, config):
    """Run simulation with minimal console output"""
    print("\n" + "="*70)
    print("  RUNNING SIMULATION")
    print("="*70)
    
    results_dir = Path("./simulation_results")
    results_dir.mkdir(exist_ok=True)
    
    env_config = checkpoint['env_config'].copy()
    env_config['max_steps'] = config['max_steps']
    
    print(f"\nLoading checkpoint {checkpoint['iteration']}...")
    
    # Suppress ALL Ray output during loading and simulation
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    
    try:
        algo = PPO.from_checkpoint(checkpoint['path'])
        env = SimpleEconomyEnv(env_config)
        obs, info = env.reset(seed=config['seed'])
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    print("Model loaded.")
    
    display_initial_state(env, checkpoint, config)
    
    # Data storage
    firms_data = []
    households_data = []
    
    print("\nRunning simulation...")
    
    done = False
    step = 0
    
    # Suppress Ray output during simulation
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    
    try:
        while not done and step < config['max_steps']:
            # Get actions
            actions = {}
            for agent_id in obs.keys():
                action, _, _ = algo.get_policy("shared_policy").compute_single_action(obs[agent_id])
                actions[agent_id] = action
            
            # Step
            obs, rewards, dones, truncated, info = env.step(actions)
            
            current_step = step + 1
            
            # Collect data in LONG format
            for i in range(checkpoint['n_firms']):
                firm_id = f"firm_{i}"
                firm = env.firms[firm_id]
                
                firms_data.append({
                    'version': config['version'],
                    'seed': config['seed'],
                    'step': current_step,
                    'firm_id': i,
                    'price': firm['price'],
                    'wage': firm['wage'],
                    'capital': firm['capital'],
                    'employees': firm['employees'],
                    'max_employees': firm['max_employees'],
                    'quality': firm['quality'],
                    'marketing': firm['marketing'],
                    'profit': firm['profit'],
                    'revenue': firm['revenue'],
                    'costs': firm['costs'],
                    'inventory': firm['inventory'],
                    'production': firm['production'],
                    'bankrupt': firm['bankrupt']
                })
            
            for i, hh in enumerate(env.households):
                households_data.append({
                    'version': config['version'],
                    'seed': config['seed'],
                    'step': current_step,
                    'household_id': i,
                    'money': hh['money'],
                    'employer': hh['employer'] if hh['employer'] else 'None',
                    'wage': hh['wage'],
                    'skill_level': hh['skill_level'],
                    'wealth_type': hh['wealth_type']
                })
            
            done = dones.get('__all__', False)
            step += 1
    
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    print(f"Simulation complete: {step} steps")
    
    display_final_state(env, checkpoint, step)
    
    algo.stop()
    
    # Convert to DataFrames
    firms_df = pd.DataFrame(firms_data)
    households_df = pd.DataFrame(households_data)
    
    return firms_df, households_df, results_dir

def save_results(firms_df, households_df, checkpoint, config, results_dir):
    """Save results in long format for database import"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save firms data
    firms_filename = f"firms_checkpoint{checkpoint['iteration']}_seed{config['seed']}_{timestamp}.csv"
    firms_filepath = results_dir / firms_filename
    firms_df.to_csv(firms_filepath, index=False)
    
    # Save households data
    households_filename = f"households_checkpoint{checkpoint['iteration']}_seed{config['seed']}_{timestamp}.csv"
    households_filepath = results_dir / households_filename
    households_df.to_csv(households_filepath, index=False)
    
    print(f"\nResults saved:")
    print(f"  Firms: {firms_filename}")
    print(f"  Households: {households_filename}")
    
    # Summary
    final_step = firms_df['step'].max()
    final_firms = firms_df[firms_df['step'] == final_step]
    final_households = households_df[households_df['step'] == final_step]
    
    bankruptcies = final_firms['bankrupt'].sum()
    total_capital = final_firms['capital'].sum()
    total_money = final_households['money'].sum()
    employed = (final_households['employer'] != 'None').sum()
    state_employed = (final_households['employer'] == 'state').sum()
    firm_employed = employed - state_employed
    
    summary_file = results_dir / f"summary_seed{config['seed']}_{timestamp}.txt"
    
    with open(summary_file, 'w') as f:
        f.write(f"VWL Simulation {config['version']} - Summary\n")
        f.write(f"="*70 + "\n\n")
        f.write(f"Version: {config['version']}\n")
        f.write(f"Checkpoint: Iteration {checkpoint['iteration']}\n")
        f.write(f"Seed: {config['seed']} (Mode: {config['seed_mode']})\n")
        f.write(f"Firms: {checkpoint['n_firms']}\n")
        f.write(f"Households: {checkpoint['n_households']}\n")
        f.write(f"Steps: {final_step}\n\n")
        f.write(f"Final State:\n")
        f.write(f"-"*70 + "\n")
        f.write(f"Bankruptcies: {bankruptcies}/{checkpoint['n_firms']}\n")
        f.write(f"Firm Employment: {firm_employed} ({(firm_employed/checkpoint['n_households']*100):.1f}%)\n")
        f.write(f"State Employment: {state_employed} ({(state_employed/checkpoint['n_households']*100):.1f}%)\n")
        f.write(f"Avg Household Money: {final_households['money'].mean():.2f}\n")
        f.write(f"Total Household Money: {total_money:.2f}\n")
        f.write(f"Total Firm Capital: {total_capital:.2f}\n\n")
        f.write(f"Database Info:\n")
        f.write(f"  - {len(firms_df)} firm records\n")
        f.write(f"  - {len(households_df)} household records\n\n")
        f.write(f"To reproduce: Use seed {config['seed']} with checkpoint {checkpoint['iteration']}\n")
    
    print(f"  Summary: {summary_file.name}")
    print(f"\nDatabase records: {len(firms_df)} firms, {len(households_df)} households")
    
    return firms_filepath, households_filepath

def main():
    print("\n" + "="*70)
    print("  VWL SIMULATION v5.1 - CONSOLE RUNNER")
    print("  Database-Ready Output | Minimal Console Logging")
    print("="*70)
    
    checkpoints = find_checkpoints()
    
    if not checkpoints:
        print("\nNo checkpoints found!")
        print("Run training first: python train.py\n")
        return
    
    checkpoint = select_checkpoint(checkpoints)
    config = get_simulation_config(checkpoint)
    firms_df, households_df, results_dir = run_simulation(checkpoint, config)
    save_results(firms_df, households_df, checkpoint, config, results_dir)
    
    print("\n" + "="*70)
    print("  SIMULATION COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

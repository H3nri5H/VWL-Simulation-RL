import os
import sys
import json
import yaml
import uuid
import numpy as np
from pathlib import Path
from datetime import datetime
from ray.rllib.algorithms.ppo import PPO
from env.economy_env import SimpleEconomyEnv

# Suppress ALL Ray output
os.environ['RAY_DEDUP_LOGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['RAY_DISABLE_MEMORY_MONITOR'] = '1'
os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'

import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger('ray').setLevel(logging.CRITICAL)
logging.getLogger('ray.tune').setLevel(logging.CRITICAL)
logging.getLogger('ray.rllib').setLevel(logging.CRITICAL)
logging.getLogger('ray.serve').setLevel(logging.CRITICAL)


def load_config():
    """Load environment configuration"""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def find_checkpoint(checkpoint_path=None):
    """Find checkpoint to use for simulation"""
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            return os.path.abspath(checkpoint_path)
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return None
    
    checkpoint_dir = Path("./checkpoints")
    if not checkpoint_dir.exists():
        print("Error: No checkpoints directory found.")
        return None
    
    checkpoints = []
    for cp_dir in checkpoint_dir.iterdir():
        if cp_dir.is_dir() and cp_dir.name.startswith('checkpoint_'):
            metadata_file = cp_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    is_favorite = metadata.get('is_favorite', False)
                    iteration = metadata.get('iteration', 0)
                    checkpoints.append((os.path.abspath(str(cp_dir)), iteration, is_favorite))
    
    if not checkpoints:
        print("Error: No valid checkpoints found.")
        return None
    
    favorite = [cp for cp in checkpoints if cp[2]]
    if favorite:
        return favorite[0][0]
    
    checkpoints.sort(key=lambda x: x[1])
    return checkpoints[-1][0]


def run_simulation(checkpoint_path=None, seed=None, verbose=False):
    """Run single simulation episode with trained policies"""
    # Generate unique simulation ID
    simulation_id = str(uuid.uuid4())[:8]
    
    # Use provided seed or generate random one
    if seed is None:
        seed = np.random.randint(0, 1000000)
    
    checkpoint_path = find_checkpoint(checkpoint_path)
    if not checkpoint_path:
        return
    
    # Suppress Ray output during loading
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = open(os.devnull, 'w')
    try:
        algo = PPO.from_checkpoint(checkpoint_path)
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout, sys.stderr = old_stdout, old_stderr
    
    metadata_file = Path(checkpoint_path) / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            env_config = metadata.get('env_config', {})
            iteration = metadata.get('iteration', 0)
    else:
        config = load_config()
        env_cfg = config.get('environment', {})
        env_config = {
            'n_firms': env_cfg.get('n_firms', 10),
            'n_households': env_cfg.get('n_households', 3000),
            'max_steps': env_cfg.get('max_steps', 365),
        }
        iteration = 0
    
    env = SimpleEconomyEnv(env_config)
    
    print("\n" + "="*60)
    print("  VWL SIMULATION - RUNNING EPISODE")
    print("="*60)
    print(f"Simulation ID: {simulation_id}")
    print(f"Checkpoint: iteration {iteration}")
    print(f"Environment: {env_config['n_firms']} firms, {env_config['n_households']} households")
    print(f"Seed: {seed}")
    print("="*60 + "\n")
    
    obs, _ = env.reset(seed=seed)
    done = {'__all__': False}
    step = 0
    
    firm_history = {f"firm_{i}": [] for i in range(env_config['n_firms'])}
    household_history = []
    
    while not done['__all__']:
        actions = {}
        for agent_id in obs.keys():
            # Suppress deprecation warnings
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = open(os.devnull, 'w')
            try:
                actions[agent_id] = algo.compute_single_action(obs[agent_id], policy_id=agent_id)
            finally:
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout, sys.stderr = old_stdout, old_stderr
        
        obs, rewards, dones, _, infos = env.step(actions)
        done = dones
        step += 1
        
        # Record firm data
        for firm_id in firm_history.keys():
            firm = env.firms[firm_id]
            firm_history[firm_id].append({
                'step': step,
                'price': firm['price'],
                'wage': firm['wage'],
                'employees': firm['employees'],
                'inventory': firm['inventory'],
                'production': firm['production'],
                'capital': firm['capital'],
                'profit': firm['profit'],
                'revenue': firm['revenue'],
                'costs': firm['costs'],
                'sales': firm['sales'],
                'quality': firm['quality'],
                'marketing': firm['marketing'],
                'bankrupt': firm['bankrupt'],
            })
        
        # Record household data (sampled every 10 steps to reduce file size)
        if step % 10 == 0:
            for idx, household in enumerate(env.households):
                household_history.append({
                    'step': step,
                    'household_id': idx,
                    'money': household['money'],
                    'skill_level': household['skill_level'],
                    'max_acceptable_price': household['max_acceptable_price'],
                    'employer': household['employer'] if household['employer'] else 'unemployed',
                    'wage': household['wage'],
                    'wealth_type': household['wealth_type'],
                })
        
        if verbose and step % 50 == 0:
            print(f"Step {step}/{env_config['max_steps']}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("./simulation_results")
    results_dir.mkdir(exist_ok=True)
    
    import csv
    
    # Save firms data
    firms_file = results_dir / f"firms_sim{simulation_id}_cp{iteration}_seed{seed}_{timestamp}.csv"
    with open(firms_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['simulation_id', 'seed', 'firm_id', 'step', 'price', 'wage', 'employees', 
                        'inventory', 'production', 'capital', 'profit', 'revenue', 'costs', 
                        'sales', 'quality', 'marketing', 'bankrupt'])
        for firm_id, history in firm_history.items():
            for record in history:
                writer.writerow([simulation_id, seed, firm_id, record['step'], record['price'], 
                               record['wage'], record['employees'], record['inventory'], 
                               record['production'], record['capital'], record['profit'], 
                               record['revenue'], record['costs'], record['sales'], 
                               record['quality'], record['marketing'], record['bankrupt']])
    
    # Save households data
    households_file = results_dir / f"households_sim{simulation_id}_cp{iteration}_seed{seed}_{timestamp}.csv"
    with open(households_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['simulation_id', 'seed', 'step', 'household_id', 'money', 'skill_level', 
                        'max_acceptable_price', 'employer', 'wage', 'wealth_type'])
        for record in household_history:
            writer.writerow([simulation_id, seed, record['step'], record['household_id'], 
                           record['money'], record['skill_level'], record['max_acceptable_price'], 
                           record['employer'], record['wage'], record['wealth_type']])
    
    # Summary
    survivors = sum(1 for f in env.firms.values() if not f['bankrupt'])
    avg_capital = np.mean([f['capital'] for f in env.firms.values() if not f['bankrupt']]) if survivors > 0 else 0
    total_household_money = sum(hh['money'] for hh in env.households)
    avg_household_money = total_household_money / len(env.households)
    
    # Count segment distribution
    segments = {'budget': 0, 'mainstream': 0, 'premium': 0}
    for firm in env.firms.values():
        if not firm['bankrupt']:
            if firm['price'] < 40:
                segments['budget'] += 1
            elif firm['price'] > 70 or firm['quality'] > 1.5:
                segments['premium'] += 1
            else:
                segments['mainstream'] += 1
    
    summary_file = results_dir / f"summary_sim{simulation_id}_seed{seed}_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Simulation ID: {simulation_id}\n")
        f.write(f"Checkpoint: iteration {iteration}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"\n--- FIRMS ---\n")
        f.write(f"Survivors: {survivors}/{env_config['n_firms']}\n")
        f.write(f"Average Capital: {avg_capital:.2f}\n")
        f.write(f"\nSegment Distribution:\n")
        f.write(f"  Budget: {segments['budget']}\n")
        f.write(f"  Mainstream: {segments['mainstream']}\n")
        f.write(f"  Premium: {segments['premium']}\n")
        f.write(f"\n--- HOUSEHOLDS ---\n")
        f.write(f"Total Households: {env_config['n_households']}\n")
        f.write(f"Average Money: {avg_household_money:.2f}\n")
        f.write(f"Total Money: {total_household_money:.2f}\n")
    
    print(f"\nSimulation complete!")
    print(f"Survivors: {survivors}/{env_config['n_firms']}")
    print(f"Segments: Budget={segments['budget']}, Mainstream={segments['mainstream']}, Premium={segments['premium']}")
    print(f"Avg Capital: {avg_capital:.2f}\n")
    print(f"Results saved:")
    print(f"  - {firms_file.name}")
    print(f"  - {households_file.name}")
    print(f"  - {summary_file.name}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run VWL simulation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific checkpoint (default: latest)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    parser.add_argument("--verbose", action='store_true', help="Verbose output")
    args = parser.parse_args()
    
    run_simulation(checkpoint_path=args.checkpoint, seed=args.seed, verbose=args.verbose)

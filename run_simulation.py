import os
import sys
import json
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from ray.rllib.algorithms.ppo import PPO
from env.economy_env import SimpleEconomyEnv

os.environ['RAY_DEDUP_LOGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger('ray').setLevel(logging.ERROR)


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
    if seed is None:
        seed = np.random.randint(0, 1000000)
    
    checkpoint_path = find_checkpoint(checkpoint_path)
    if not checkpoint_path:
        return
    
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        algo = PPO.from_checkpoint(checkpoint_path)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    
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
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                actions[agent_id] = algo.compute_single_action(obs[agent_id], policy_id=agent_id)
            finally:
                sys.stdout.close()
                sys.stdout = old_stdout
        
        obs, rewards, dones, _, infos = env.step(actions)
        done = dones
        step += 1
        
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
        
        if verbose and step % 50 == 0:
            print(f"Step {step}/{env_config['max_steps']}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("./simulation_results")
    results_dir.mkdir(exist_ok=True)
    
    import csv
    
    # Save firms data
    firms_file = results_dir / f"firms_checkpoint{iteration}_seed{seed}_{timestamp}.csv"
    with open(firms_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['firm_id', 'step', 'price', 'wage', 'employees', 'inventory', 'production',
                        'capital', 'profit', 'revenue', 'costs', 'sales', 'quality', 'marketing', 'bankrupt'])
        for firm_id, history in firm_history.items():
            for record in history:
                writer.writerow([firm_id, record['step'], record['price'], record['wage'], 
                               record['employees'], record['inventory'], record['production'],
                               record['capital'], record['profit'], record['revenue'], record['costs'],
                               record['sales'], record['quality'], record['marketing'], record['bankrupt']])
    
    # Summary
    survivors = sum(1 for f in env.firms.values() if not f['bankrupt'])
    avg_capital = np.mean([f['capital'] for f in env.firms.values() if not f['bankrupt']]) if survivors > 0 else 0
    
    summary_file = results_dir / f"summary_seed{seed}_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Checkpoint: iteration {iteration}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Survivors: {survivors}/{env_config['n_firms']}\n")
        f.write(f"Average Capital: {avg_capital:.2f}\n")
    
    print(f"\nSimulation complete!")
    print(f"Survivors: {survivors}/{env_config['n_firms']}")
    print(f"Avg Capital: {avg_capital:.2f}\n")
    print(f"Results saved:")
    print(f"  - {firms_file.name}")
    print(f"  - {summary_file.name}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run VWL simulation")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--verbose", action='store_true', help="Verbose output")
    args = parser.parse_args()
    
    run_simulation(checkpoint_path=args.checkpoint, seed=args.seed, verbose=args.verbose)

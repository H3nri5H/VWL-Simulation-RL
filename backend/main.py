from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "VWL-Simulation Backend Running"}


@app.get("/api/training/latest")
def get_latest_training():
    checkpoint_dir = Path("./checkpoints")
    
    if not checkpoint_dir.exists():
        return {"error": "No checkpoints found"}
    
    result_files = list(checkpoint_dir.glob("*/result.json"))
    
    if not result_files:
        return {"error": "No training results found"}
    
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    return {
        "metrics": data,
        "checkpoint_path": str(latest_file.parent)
    }


@app.get("/api/training/history")
def get_training_history():
    checkpoint_dir = Path("./checkpoints")
    
    if not checkpoint_dir.exists():
        return {"iterations": []}
    
    result_files = sorted(
        checkpoint_dir.glob("*/result.json"),
        key=lambda p: p.stat().st_mtime
    )
    
    history = []
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            env_runners = data.get('env_runners', {})
            
            history.append({
                "iteration": data.get('training_iteration', 0),
                "reward_mean": env_runners.get('episode_reward_mean', 0),
                "reward_min": env_runners.get('episode_reward_min', 0),
                "reward_max": env_runners.get('episode_reward_max', 0),
                "episode_len": env_runners.get('episode_len_mean', 0),
                "timestamp": data.get('timestamp', 0),
            })
        except:
            continue
    
    return {"iterations": history}

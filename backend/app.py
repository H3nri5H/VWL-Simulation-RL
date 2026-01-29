"""FastAPI Backend for VWL-Simulation.

Provides REST API endpoints for:
- Listing trained models
- Running simulations with trained policies
- Retrieving simulation history
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import glob
from typing import List, Optional
from inference import SimulationRunner

app = FastAPI(title="VWL-Simulation API", version="1.0.0")

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulationRequest(BaseModel):
    """Request model for simulation."""
    model_name: str
    n_firms: int = 2
    n_households: int = 10
    max_steps: int = 100
    start_prices: Optional[List[float]] = None
    start_wages: Optional[List[float]] = None
    seed: Optional[int] = None


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "VWL-Simulation API"}


@app.get("/api/models")
def list_models():
    """Liste aller verfügbaren trainierten Modelle.
    
    Returns:
        dict: {"models": ["checkpoint_000100", "checkpoint_000200", ...]}
    """
    models_dir = "../models"
    
    if not os.path.exists(models_dir):
        return {"models": [], "message": "No models directory found"}
    
    # Finde alle Checkpoint-Ordner
    checkpoints = glob.glob(f"{models_dir}/checkpoint_*")
    model_names = [os.path.basename(cp) for cp in checkpoints]
    
    return {
        "models": sorted(model_names),
        "count": len(model_names)
    }


@app.post("/api/simulate")
def run_simulation(request: SimulationRequest):
    """Führt Simulation mit trainierter Policy aus.
    
    Args:
        request: SimulationRequest mit Modell-Name und Parametern
        
    Returns:
        dict: {"status": "success", "data": {...history...}}
    """
    model_path = f"../models/{request.model_name}"
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{request.model_name}' not found"
        )
    
    try:
        # Initialize simulation runner
        runner = SimulationRunner(checkpoint_path=model_path)
        
        # Prepare start parameters
        start_params = {}
        if request.start_prices:
            start_params['prices'] = request.start_prices
        if request.start_wages:
            start_params['wages'] = request.start_wages
        
        # Run simulation
        history = runner.run_simulation(
            n_firms=request.n_firms,
            n_households=request.n_households,
            max_steps=request.max_steps,
            start_params=start_params if start_params else None,
            seed=request.seed
        )
        
        return {
            "status": "success",
            "data": history,
            "metadata": {
                "model": request.model_name,
                "n_firms": request.n_firms,
                "n_households": request.n_households,
                "steps": request.max_steps
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {str(e)}"
        )


@app.get("/api/health")
def health_check():
    """Detaillierter Health Check."""
    return {
        "status": "healthy",
        "models_available": len(list_models()["models"]),
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

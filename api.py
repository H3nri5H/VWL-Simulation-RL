import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
VENV_PYTHON = ROOT_DIR / "venv" / "bin" / "python"
PYTHON_BIN = os.getenv("PYTHON_BIN") or (str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable)

app = FastAPI(title="VWL Simulation API", version="2.0.0")
RUNNING_TRAIN_PIDS: set[int] = set()
RUNNING_SIMULATION_PROCESSES: Dict[int, subprocess.Popen] = {}


class TrainRequest(BaseModel):
    resume: bool = Field(default=False, description="Resume from latest checkpoint")


class SimulationRequest(BaseModel):
    checkpoint: Optional[str] = Field(default=None, description="Checkpoint path")
    seed: Optional[int] = Field(default=None, description="Seed for reproducibility")
    steps: Optional[int] = Field(default=None, gt=0, description="Max simulation steps")


class CheckpointInfo(BaseModel):
    path: str
    iteration: int


class StartResponse(BaseModel):
    status: str
    pid: int
    command: List[str]
    log_file: str


class StopResponse(BaseModel):
    status: str
    stopped_pids: List[int]


class SimulationResponse(BaseModel):
    status: str
    pid: int
    command: List[str]
    log_file: str
    running: bool
    exit_code: Optional[int] = None


class ProcessStatusResponse(BaseModel):
    pid: int
    running: bool
    status: str
    exit_code: Optional[int] = None


def list_checkpoints() -> List[CheckpointInfo]:
    if not CHECKPOINT_DIR.exists():
        return []

    checkpoints = []
    for cp_dir in CHECKPOINT_DIR.iterdir():
        if not cp_dir.is_dir() or not cp_dir.name.startswith("checkpoint_"):
            continue

        metadata_file = cp_dir / "metadata.json"
        iteration = 0

        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                iteration = int(metadata.get("iteration", 0))
            except (ValueError, json.JSONDecodeError):
                pass

        checkpoints.append(
            CheckpointInfo(
                path=str(cp_dir.resolve()),
                iteration=iteration,
            )
        )

    checkpoints.sort(key=lambda c: c.iteration)
    return checkpoints


def latest_checkpoint_path() -> Optional[str]:
    checkpoints = list_checkpoints()
    if not checkpoints:
        return None
    return checkpoints[-1].path


def start_process(command: List[str], log_prefix: str) -> StartResponse:
    log_file = LOG_DIR / f"{log_prefix}_{uuid.uuid4()}.log"
    with open(log_file, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
    if log_prefix == "train":
        RUNNING_TRAIN_PIDS.add(process.pid)

    return StartResponse(
        status="started",
        pid=process.pid,
        command=command,
        log_file=str(log_file),
    )


def _spawn_process(command: List[str], log_prefix: str) -> tuple[subprocess.Popen, str]:
    log_file = LOG_DIR / f"{log_prefix}_{uuid.uuid4()}.log"
    with open(log_file, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
    return process, str(log_file)


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _scan_train_pids() -> List[int]:
    try:
        output = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
    except Exception:
        return []

    pids: List[int] = []
    root_str = str(ROOT_DIR)
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        pid_str, command = parts
        if "train.py" not in command:
            continue
        if root_str not in command and str(VENV_PYTHON) not in command:
            continue
        try:
            pids.append(int(pid_str))
        except ValueError:
            continue
    return pids


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/checkpoints", response_model=List[CheckpointInfo])
def get_checkpoints() -> List[CheckpointInfo]:
    return list_checkpoints()


@app.post("/train", response_model=StartResponse)
def start_training(payload: Optional[TrainRequest] = None) -> StartResponse:
    payload = payload or TrainRequest()
    command = [PYTHON_BIN, "train.py"]
    if payload.resume:
        command.append("--resume")
    return start_process(command, "train")


@app.post("/simulate", response_model=SimulationResponse)
def start_simulation(
    payload: Optional[SimulationRequest] = None,
    wait_for_finish: bool = False,
    timeout_seconds: Optional[int] = None,
) -> SimulationResponse:
    payload = payload or SimulationRequest()
    checkpoint = payload.checkpoint or latest_checkpoint_path()
    if checkpoint is None:
        raise HTTPException(status_code=400, detail="No checkpoint found. Train first or provide a checkpoint path.")

    command = [
        PYTHON_BIN,
        "run_simulation.py",
        "--no-interactive",
        "--checkpoint",
        checkpoint,
    ]

    if payload.seed is not None:
        command.extend(["--seed", str(payload.seed)])
    if payload.steps is not None:
        command.extend(["--steps", str(payload.steps)])

    process, log_file = _spawn_process(command, "simulation")
    RUNNING_SIMULATION_PROCESSES[process.pid] = process

    if not wait_for_finish:
        return SimulationResponse(
            status="started",
            pid=process.pid,
            command=command,
            log_file=log_file,
            running=True,
            exit_code=None,
        )

    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return SimulationResponse(
            status="running_timeout",
            pid=process.pid,
            command=command,
            log_file=log_file,
            running=True,
            exit_code=None,
        )

    exit_code = process.returncode
    RUNNING_SIMULATION_PROCESSES.pop(process.pid, None)
    return SimulationResponse(
        status="completed" if exit_code == 0 else "failed",
        pid=process.pid,
        command=command,
        log_file=log_file,
        running=False,
        exit_code=exit_code,
    )


@app.get("/simulation-status/{pid}", response_model=ProcessStatusResponse)
def simulation_status(pid: int) -> ProcessStatusResponse:
    process = RUNNING_SIMULATION_PROCESSES.get(pid)
    if process is None:
        running = _is_pid_alive(pid)
        return ProcessStatusResponse(
            pid=pid,
            running=running,
            status="running_untracked" if running else "finished_or_unknown",
            exit_code=None,
        )

    exit_code = process.poll()
    if exit_code is None:
        return ProcessStatusResponse(pid=pid, running=True, status="running", exit_code=None)

    RUNNING_SIMULATION_PROCESSES.pop(pid, None)
    return ProcessStatusResponse(
        pid=pid,
        running=False,
        status="completed" if exit_code == 0 else "failed",
        exit_code=exit_code,
    )


@app.post("/stop-training", response_model=StopResponse)
def stop_training(force: bool = False) -> StopResponse:
    candidate_pids = [pid for pid in RUNNING_TRAIN_PIDS if _is_pid_alive(pid)]
    if not candidate_pids:
        candidate_pids = _scan_train_pids()

    stopped_pids: List[int] = []
    sig = signal.SIGKILL if force else signal.SIGTERM
    for pid in candidate_pids:
        try:
            os.kill(pid, sig)
            stopped_pids.append(pid)
        except ProcessLookupError:
            continue

    RUNNING_TRAIN_PIDS.difference_update(stopped_pids)
    status = "stopped" if stopped_pids else "no_training_process_found"
    return StopResponse(status=status, stopped_pids=stopped_pids)

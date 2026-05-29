"""Template worker for RF-Agent-Python pending candidate folders.

Copy this file into your Python training project and replace train_one_candidate
with your real training/evaluation function.
"""

from __future__ import annotations

import importlib.util
import json
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dashboard import render_dashboard_for_task
from candidate_store import CandidateStore
from config import load_json
from python_log_reader import PythonLogReader
from recommendations import print_training_recommendations
from task_loader import load_task_folder
from tree import SearchTree


def main() -> None:
    args = parse_args()
    task_dir = Path(args.task_dir)
    if not task_dir.is_absolute():
        task_dir = PROJECT_ROOT / task_dir
    candidates_dir = task_dir / "candidates"

    if not candidates_dir.exists():
        print(f"No candidates directory found: {candidates_dir}")
        return

    for candidate_dir in sorted(candidates_dir.glob("candidate_*")):
        status_path = candidate_dir / "status.json"
        if not status_path.exists():
            continue

        status = read_json(status_path)
        if status.get("status") != "pending":
            continue

        try:
            mark_status(candidate_dir, "running")
            reward_path = candidate_dir / "reward_fcn.py"
            reward_fn = load_reward_function(reward_path)

            result = train_one_candidate(reward_fn, reward_path, candidate_dir)
            result.setdefault("status", "trained")

            write_json(candidate_dir / "summary.json", result)
            mark_status(candidate_dir, "trained")
        except Exception as exc:
            result = {
                "status": "failed",
                "error_message": str(exc),
            }
            write_json(candidate_dir / "summary.json", result)
            mark_status(candidate_dir, "failed", str(exc))

    render_dashboard_for_task(
        task_dir=task_dir,
        agent_config_path=PROJECT_ROOT / "configs" / "agent.json",
        show_window=True,
    )
    print_recommendations_for_task(task_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Template offline Python worker for RF-Agent-Python.")
    parser.add_argument("--task-dir", default="tasks/python_task")
    return parser.parse_args()


def train_one_candidate(reward_fn, reward_path: Path, candidate_dir: Path) -> Dict[str, Any]:
    """Replace this placeholder with your real Python RL training.

    Your trainer should:
    - use reward_fn or reward_path
    - write logs/train.csv and/or logs/eval.csv when available
    - return the metrics used by configs/task.json
    """

    # Tiny placeholder call to verify the generated function is importable.
    obs = {"position_error": 1.0, "velocity_error": 0.5}
    action = {"motor_torque": [0.1, -0.2]}
    next_obs = {"position_error": 0.8, "velocity_error": 0.4}
    info = {"constraint_violation": 0.0, "success": False}
    _ = reward_fn(obs, action, next_obs, info)

    return {
        "status": "trained",
        "max_task_score": 0.0,
        "final_task_score": 0.0,
        "mean_return": 0.0,
        "success_rate": 0.0,
        "constraint_violation": 0.0,
        "notes": "Placeholder result from template worker.",
    }


def load_reward_function(path: Path):
    spec = importlib.util.spec_from_file_location(f"reward_{path.parent.name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load reward module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "compute_reward"):
        raise RuntimeError(f"{path} does not define compute_reward")
    return module.compute_reward


def mark_status(candidate_dir: Path, state: str, error_message: str = "") -> None:
    write_json(
        candidate_dir / "status.json",
        {
            "status": state,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "error_message": error_message,
        },
    )


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def print_recommendations_for_task(task_dir: Path) -> None:
    task_config = load_task_folder(task_dir)
    agent_config = load_json(PROJECT_ROOT / "configs" / "agent.json")
    store = CandidateStore(task_dir / "candidates")
    log_reader = PythonLogReader(
        task_config.get("score", {}),
        dummy_failure=float(agent_config.get("dummy_failure", -10000.0)),
    )
    tree = SearchTree(
        store.scan(),
        log_reader,
        float(agent_config.get("dummy_failure", -10000.0)),
        max_simulations=int(agent_config.get("simulations", 80)),
    )
    latest_path = store.root / "latest_generation.json"
    latest_decisions = load_json(latest_path) if latest_path.exists() else []
    print_training_recommendations(tree, latest_decisions)


if __name__ == "__main__":
    main()

"""Template worker for RF-Agent-Python pending candidate folders.

Copy this file into your Python training project and replace train_one_candidate
with your real training/evaluation function.
"""

from __future__ import annotations

import importlib.util
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    task_dir = Path(args.experiments_dir) / args.task_name
    if not task_dir.is_absolute():
        task_dir = PROJECT_ROOT / task_dir

    if not task_dir.exists():
        print(f"No task directory found: {task_dir}")
        return

    for candidate_dir in sorted(task_dir.glob("candidate_*")):
        status_path = candidate_dir / "status.json"
        if not status_path.exists():
            continue

        status = read_json(status_path)
        if status.get("status") != "pending":
            continue

        try:
            mark_status(candidate_dir, "running")
            reward_path = candidate_dir / "reward.py"
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


def parse_args():
    parser = argparse.ArgumentParser(description="Template offline Python worker for RF-Agent-Python.")
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--task-name", default="python_task")
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


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from config import load_json


TASK_TEXT_FILES = {
    "description": "description.md",
    "observations": "observations.md",
    "environment": "environment.md",
    "original_reward": "original_reward.py",
}


def load_task_folder(task_dir: Path) -> dict:
    task_dir = Path(task_dir)
    task_config_path = task_dir / "task.json"
    if not task_config_path.exists():
        raise FileNotFoundError(f"Task config not found: {task_config_path}")

    config = load_json(task_config_path)
    config["_task_dir"] = str(task_dir)
    config.setdefault("task_name", task_dir.name)
    config.setdefault("reward_language", "python")
    config.setdefault("reward_function_name", "compute_reward")
    config.setdefault("reward_file", "reward_fcn.py")

    for key, filename in TASK_TEXT_FILES.items():
        path = task_dir / filename
        config[f"{key}_text"] = path.read_text(encoding="utf-8") if path.exists() else ""

    config["log_inventory"] = collect_log_inventory(task_dir / "logs")
    config["candidate_log_inventory"] = collect_log_inventory(task_dir / "candidates")
    config["task_context"] = build_task_context(config)
    return config


def collect_log_inventory(path: Path) -> List[str]:
    if not path.exists():
        return []
    files = []
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            try:
                files.append(str(file_path.relative_to(path)))
            except ValueError:
                files.append(str(file_path))
    return files


def build_task_context(config: Dict) -> str:
    sections = []
    sections.append(section("Purpose / Description", config.get("description_text") or config.get("description", "")))
    sections.append(section("Observations", config.get("observations_text", "")))
    sections.append(section("Environment", config.get("environment_text", "")))
    sections.append(section("Original Reward", config.get("original_reward_text", "")))

    available_variables = config.get("available_variables", [])
    if available_variables:
        sections.append(section("Available Variables", "\n".join(f"- {item}" for item in available_variables)))

    log_inventory = config.get("log_inventory", [])
    if log_inventory:
        sections.append(section("Task Log Files", "\n".join(f"- {item}" for item in log_inventory)))

    candidate_log_inventory = config.get("candidate_log_inventory", [])
    if candidate_log_inventory:
        sections.append(section("Candidate Log Files", "\n".join(f"- {item}" for item in candidate_log_inventory)))

    return "\n\n".join(part for part in sections if part.strip())


def section(title: str, content: str) -> str:
    content = (content or "").strip()
    if not content:
        return ""
    return f"## {title}\n{content}"


from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from config import load_json


DEFAULT_TASK_TEXT_FILES = {
    "description": {
        "path": "description.md",
        "title": "Purpose / Description",
    },
    "observations": {
        "path": "observations.md",
        "title": "Observations",
    },
    "environment": {
        "path": "environment.md",
        "title": "Environment",
    },
    "original_reward": {
        "path": "original_reward.py",
        "title": "Original Reward",
    },
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

    text_files = normalize_task_text_files(config.get("task_text_files", DEFAULT_TASK_TEXT_FILES))
    text_sections = []
    for item in text_files:
        path = task_dir / item["path"]
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        config[f"{item['key']}_text"] = content
        text_sections.append(
            {
                "key": item["key"],
                "title": item["title"],
                "path": item["path"],
                "content": content,
            }
        )
    config["_task_text_sections"] = text_sections

    config["log_inventory"] = collect_log_inventory(task_dir / "logs")
    config["candidate_log_inventory"] = collect_log_inventory(task_dir / "candidates")
    config["task_context"] = build_task_context(config)
    return config


def normalize_task_text_files(config_value) -> List[Dict[str, str]]:
    items = config_value.items() if isinstance(config_value, dict) else enumerate(config_value or [])
    normalized = []
    for key, value in items:
        if isinstance(value, str):
            text_key = str(key)
            path = value
            title = title_from_key(text_key)
        else:
            text_key = str(value.get("key", key))
            path = value.get("path") or value.get("file")
            title = value.get("title", title_from_key(text_key))
        if not path:
            continue
        normalized.append(
            {
                "key": text_key,
                "path": str(path),
                "title": str(title),
            }
        )
    return normalized


def title_from_key(key: str) -> str:
    if key == "description":
        return "Purpose / Description"
    return key.replace("_", " ").title()


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
    text_sections = config.get("_task_text_sections", [])
    if text_sections:
        for item in text_sections:
            fallback = config.get("description", "") if item["key"] == "description" else ""
            sections.append(section(item["title"], item.get("content") or fallback))
    else:
        sections.append(section("Purpose / Description", config.get("description", "")))

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

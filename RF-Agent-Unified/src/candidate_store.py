from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from config import load_json, save_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Candidate:
    candidate_id: str
    folder: Path
    metadata: dict
    status: dict
    summary: Optional[dict]
    reward_code: str

    @property
    def parent_id(self):
        return self.metadata.get("parent_id")

    @property
    def action_type(self):
        return self.metadata.get("action_type", "unknown")

    @property
    def action_index(self):
        return self.metadata.get("action_index", 0)

    @property
    def is_trained(self):
        return self.status.get("status") == "trained" and self.summary is not None

    @property
    def is_pending(self):
        return self.status.get("status") == "pending"

    @property
    def is_failed(self):
        return self.status.get("status") == "failed"


class CandidateStore:
    def __init__(self, candidates_dir: Path, task_name: Optional[str] = None):
        self.root = Path(candidates_dir) / task_name if task_name else Path(candidates_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def scan(self) -> List[Candidate]:
        candidates = []
        for folder in sorted(self.root.glob("candidate_*")):
            if not folder.is_dir():
                continue
            metadata_path = folder / "metadata.json"
            if not metadata_path.exists():
                continue
            metadata = load_json(metadata_path)
            status_path = folder / "status.json"
            status = load_json(status_path) if status_path.exists() else {"status": metadata.get("status", "unknown")}
            summary_path = folder / "summary.json"
            summary = load_json(summary_path) if summary_path.exists() else None
            reward_language = str(metadata.get("reward_language", "")).lower()
            default_reward = "reward_fcn.m" if reward_language in {"matlab", "m"} else "reward_fcn.py"
            reward_file = folder / metadata.get("reward_file", default_reward)
            if not reward_file.exists():
                for fallback in ("reward_fcn.py", "reward_fcn.m", "reward.py", "reward.m"):
                    fallback_path = folder / fallback
                    if fallback_path.exists():
                        reward_file = fallback_path
                        break
            reward_code = reward_file.read_text(encoding="utf-8") if reward_file.exists() else ""
            candidates.append(
                Candidate(
                    candidate_id=metadata["candidate_id"],
                    folder=folder,
                    metadata=metadata,
                    status=status,
                    summary=summary,
                    reward_code=reward_code,
                )
            )
        return candidates

    def next_candidate_id(self) -> str:
        max_id = 0
        pattern = re.compile(r"candidate_(\d+)$")
        for folder in self.root.glob("candidate_*"):
            match = pattern.match(folder.name)
            if match:
                max_id = max(max_id, int(match.group(1)))
        return f"candidate_{max_id + 1:03d}"

    def load_elite_ids(self) -> List[str]:
        elite_path = self.root / "elite_set.json"
        if elite_path.exists():
            data = load_json(elite_path)
            if isinstance(data, list):
                return [str(item) for item in data]
            return [str(item) for item in data.get("candidate_ids", [])]

        latest_path = self.root / "latest_generation.json"
        if not latest_path.exists():
            return []
        latest = load_json(latest_path)
        ids = []
        for decision in latest:
            selected_id = decision.get("selected_tree_node_id") or decision.get("parent_id")
            if selected_id and selected_id not in ids:
                ids.append(selected_id)
        return ids

    def save_elite_ids(self, candidate_ids: Iterable[str]) -> None:
        save_json(
            self.root / "elite_set.json",
            {
                "candidate_ids": list(candidate_ids),
                "updated_at": utc_now(),
            },
        )

    def create_candidate(
        self,
        *,
        parent_id: Optional[str],
        action_type: str,
        action_index: int,
        generation: int,
        reward_language: str,
        reward_code: str,
        design_thought: str,
        prompt_messages: list,
        source_node_ids: Iterable[str],
        extra_metadata: Optional[dict] = None,
    ) -> Candidate:
        candidate_id = self.next_candidate_id()
        folder = self.root / candidate_id
        logs_dir = folder / "logs"
        logs_dir.mkdir(parents=True, exist_ok=False)

        reward_language = reward_language.lower()
        reward_file = "reward_fcn.m" if reward_language in {"matlab", "m"} else "reward_fcn.py"
        metadata = {
            "candidate_id": candidate_id,
            "parent_id": parent_id,
            "action_type": action_type,
            "action_index": action_index,
            "generation": generation,
            "created_at": utc_now(),
            "created_by": "rf-agent-unified",
            "reward_language": reward_language,
            "reward_file": reward_file,
            "design_thought": design_thought,
            "source_node_ids": list(source_node_ids),
            "status": "pending",
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        status = {
            "status": "pending",
            "updated_at": utc_now(),
            "error_message": "",
        }

        (folder / reward_file).write_text(reward_code.rstrip() + "\n", encoding="utf-8")
        (folder / "description.txt").write_text(design_thought.rstrip() + "\n", encoding="utf-8")
        save_json(folder / "metadata.json", metadata)
        save_json(folder / "status.json", status)
        save_json(folder / "prompt_messages.json", prompt_messages)

        return Candidate(
            candidate_id=candidate_id,
            folder=folder,
            metadata=metadata,
            status=status,
            summary=None,
            reward_code=reward_code,
        )

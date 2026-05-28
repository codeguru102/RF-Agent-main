from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


class PythonLogReader:
    def __init__(self, score_config: dict, dummy_failure: float = -10000.0):
        self.score_config = score_config
        self.dummy_failure = dummy_failure

    def score_summary(self, summary: Optional[dict]) -> float:
        if not summary:
            return self.dummy_failure
        if summary.get("status") == "failed":
            return self.dummy_failure

        primary = self.score_config.get("primary", "max_task_score")
        maximize = self.score_config.get("maximize", True)
        value = summary.get(primary)
        if value is None:
            return self.dummy_failure

        score = float(value)
        for penalty in self.score_config.get("penalties", []):
            field = penalty.get("field")
            weight = float(penalty.get("weight", 0.0))
            if field in summary:
                score -= weight * float(summary[field])

        return score if maximize else -score

    def summarize_csv_logs(self, candidate_folder: Path) -> Dict[str, dict]:
        logs_dir = Path(candidate_folder) / "logs"
        if not logs_dir.exists():
            return {}
        summaries = {}
        for csv_path in sorted(logs_dir.glob("*.csv")):
            summaries[csv_path.name] = self._summarize_csv(csv_path)
        return summaries

    def _summarize_csv(self, csv_path: Path) -> Dict[str, dict]:
        numeric_values: Dict[str, List[float]] = defaultdict(list)
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                for key, value in row.items():
                    try:
                        numeric_values[key].append(float(value))
                    except (TypeError, ValueError):
                        continue

        summary = {}
        for key, values in numeric_values.items():
            if not values:
                continue
            summary[key] = {
                "first": values[0],
                "last": values[-1],
                "max": max(values),
                "min": min(values),
                "mean": sum(values) / len(values),
                "samples": values[:: max(len(values) // 10, 1)],
            }
        return summary

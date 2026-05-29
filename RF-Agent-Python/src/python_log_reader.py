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
            value = self._metric_fallback(summary, primary)
        if value is None:
            return self.dummy_failure

        score = float(value)
        for penalty in self.score_config.get("penalties", []):
            field = penalty.get("field")
            weight = float(penalty.get("weight", 0.0))
            if field in summary:
                score -= weight * float(summary[field])

        return score if maximize else -score

    def summary_from_csv_logs(self, candidate_folder: Path) -> Dict[str, object]:
        csv_summaries = self.summarize_csv_logs(candidate_folder)
        if not csv_summaries:
            return {}

        metrics = self._aggregate_metric_summaries(csv_summaries)
        summary: Dict[str, object] = {
            "status": "trained",
            "log_files": sorted(csv_summaries),
            "metrics": metrics,
        }

        for metric_name, stats in metrics.items():
            summary[f"max_{metric_name}"] = stats["max"]
            summary[f"final_{metric_name}"] = stats["last"]
            summary[f"mean_{metric_name}"] = stats["mean"]

        primary = self.score_config.get("primary", "max_task_score")
        primary_value = self._derive_primary_score(summary, primary)
        if primary_value is not None:
            summary[primary] = primary_value

        if "reward" in metrics:
            summary.setdefault("max_task_score", metrics["reward"]["max"])
            summary.setdefault("final_task_score", metrics["reward"]["last"])
            summary.setdefault("mean_return", metrics["reward"]["mean"])
        if "consecutive_successes" in metrics:
            summary.setdefault("max_task_score", metrics["consecutive_successes"]["max"])
            summary.setdefault("success_rate", metrics["consecutive_successes"]["last"])
        if "task_score" in metrics:
            summary.setdefault("max_task_score", metrics["task_score"]["max"])
            summary.setdefault("final_task_score", metrics["task_score"]["last"])
        if "success_rate" in metrics:
            summary.setdefault("success_rate", metrics["success_rate"]["last"])

        return summary

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

    def _aggregate_metric_summaries(self, csv_summaries: Dict[str, dict]) -> Dict[str, dict]:
        by_metric: Dict[str, List[dict]] = defaultdict(list)
        for file_summary in csv_summaries.values():
            for metric_name, stats in file_summary.items():
                by_metric[metric_name].append(stats)

        metrics = {}
        for metric_name, stats_list in by_metric.items():
            samples = []
            for stats in stats_list:
                samples.extend(stats.get("samples", []))
            metrics[metric_name] = {
                "max": max(stats["max"] for stats in stats_list),
                "min": min(stats["min"] for stats in stats_list),
                "mean": sum(stats["mean"] for stats in stats_list) / len(stats_list),
                "first": stats_list[0]["first"],
                "last": stats_list[-1]["last"],
                "samples": samples[:: max(len(samples) // 10, 1)] if samples else [],
            }
        return metrics

    def _derive_primary_score(self, summary: Dict[str, object], primary: str):
        direct = summary.get(primary)
        if direct is not None:
            return direct

        value = self._metric_fallback(summary, primary)
        if value is not None:
            return value

        metrics = summary.get("metrics", {})
        if not isinstance(metrics, dict):
            return None
        for metric_name in ("task_score", "consecutive_successes", "success_rate", "reward", "gt_reward"):
            stats = metrics.get(metric_name)
            if isinstance(stats, dict) and "max" in stats:
                return stats["max"]
        return None

    def _metric_fallback(self, summary: Dict[str, object], primary: str):
        metrics = summary.get("metrics", {})
        if not isinstance(metrics, dict):
            return None

        prefix_map = {
            "max_": "max",
            "final_": "last",
            "last_": "last",
            "mean_": "mean",
            "min_": "min",
        }
        for prefix, stat_name in prefix_map.items():
            if not primary.startswith(prefix):
                continue
            metric_name = primary[len(prefix):]
            stats = metrics.get(metric_name)
            if isinstance(stats, dict):
                return stats.get(stat_name)
        return None

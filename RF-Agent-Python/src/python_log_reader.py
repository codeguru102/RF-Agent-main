from __future__ import annotations

import ast
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


class PythonLogReader:
    def __init__(self, score_config: dict, dummy_failure: float = -10000.0, q_value_config: Optional[dict] = None):
        self.score_config = score_config
        self.dummy_failure = dummy_failure
        self.q_value_config = q_value_config or {}

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

    def q_value_for_candidate(self, candidate_folder: Path, fallback_score: float) -> float:
        if not self.q_value_config:
            return fallback_score

        metrics = self.read_eval_metrics(candidate_folder)
        if metrics is None:
            return fallback_score

        try:
            return float(evaluate_metric_formula(metrics, self.q_value_config))
        except (ValueError, TypeError, SyntaxError):
            return fallback_score

    def read_eval_metrics(self, candidate_folder: Path) -> Optional[List[float]]:
        eval_path = Path(candidate_folder) / "logs" / "eval.txt"
        if not eval_path.exists():
            return None

        lines = [line.strip() for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return None

        values = [float(item) for item in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", lines[-1])]
        num_metrics = int(self.q_value_config.get("num_metrics", len(values)))
        if len(values) < num_metrics:
            return None
        return values[:num_metrics]

    def read_feedback_text(self, candidate_folder: Path, max_chars: int = 6000) -> str:
        feedback_path = Path(candidate_folder) / "logs" / "feedback.txt"
        if not feedback_path.exists():
            return ""

        text = feedback_path.read_text(encoding="utf-8").strip()
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars].rstrip() + "\n[feedback.txt truncated]"
        return text

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


def evaluate_metric_formula(metrics: List[float], q_value_config: dict) -> float:
    num_metrics = int(q_value_config.get("num_metrics", len(metrics)))
    if len(metrics) < num_metrics:
        raise ValueError("Not enough eval metrics for q_value formula.")

    formula = str(q_value_config.get("calc_formula", "")).strip()
    if not formula:
        raise ValueError("Missing q_value calc_formula.")

    variables = {f"m{index}": metrics[index - 1] for index in range(1, num_metrics + 1)}
    for index in range(1, num_metrics + 1):
        formula = formula.replace(f"%{index}", f"m{index}")

    for offset, name in enumerate("abcdefghijklmnopqrstuvwxyz", start=1):
        if offset > num_metrics:
            break
        variables[name] = metrics[offset - 1]

    tree = ast.parse(formula, mode="eval")
    _validate_formula_ast(tree, set(variables))
    return float(eval(compile(tree, "<q_value_formula>", "eval"), {"__builtins__": {}, "max": max, "min": min, "abs": abs}, variables))


def _validate_formula_ast(tree: ast.AST, variable_names: set) -> None:
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
    )
    allowed_calls = {"max", "min", "abs"}

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported q_value formula expression: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls:
                raise ValueError("Only max, min, and abs calls are allowed in q_value formulas.")
        if isinstance(node, ast.Name) and node.id not in variable_names and node.id not in allowed_calls:
            raise ValueError(f"Unknown q_value formula variable: {node.id}")

from __future__ import annotations

from candidate_store import Candidate
from python_log_reader import PythonLogReader


class FeedbackBuilder:
    def __init__(self, log_reader: PythonLogReader):
        self.log_reader = log_reader

    def build(self, candidate: Candidate) -> str:
        parts = []
        parts.append(f"candidate_id: {candidate.candidate_id}")
        parts.append(f"action_type: {candidate.action_type}")
        parts.append(f"status: {candidate.status.get('status', 'unknown')}")

        if candidate.summary:
            parts.append("summary:")
            for key, value in sorted(candidate.summary.items()):
                parts.append(f"- {key}: {value}")

        score = self.log_reader.score_summary(candidate.summary)
        parts.append(f"computed_selection_score: {score:.6f}")

        feedback_text = self.log_reader.read_feedback_text(candidate.folder)
        if feedback_text:
            parts.append("result feedback:")
            parts.append(feedback_text)

        csv_summaries = self.log_reader.summarize_csv_logs(candidate.folder)
        if csv_summaries:
            parts.append("csv log summaries:")
            for filename, metrics in csv_summaries.items():
                parts.append(f"- {filename}:")
                for metric_name, stats in metrics.items():
                    samples = ", ".join(f"{v:.4g}" for v in stats["samples"])
                    parts.append(
                        f"  {metric_name}: samples=[{samples}], "
                        f"max={stats['max']:.4g}, mean={stats['mean']:.4g}, min={stats['min']:.4g}, last={stats['last']:.4g}"
                    )

        if candidate.status.get("error_message"):
            parts.append(f"error_message: {candidate.status['error_message']}")

        return "\n".join(parts)

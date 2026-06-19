from __future__ import annotations

from candidate_store import Candidate
from log_reader import OfflineLogReader


class FeedbackBuilder:
    def __init__(self, log_reader: OfflineLogReader):
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

        fallback_score = self.log_reader.score_summary(candidate.summary)
        score = self.log_reader.q_value_for_candidate(candidate.folder, fallback_score)
        parts.append(f"computed_selection_score: {score:.6f}")

        inventory = self.log_reader.log_file_inventory(candidate.folder)
        if any(inventory.values()):
            parts.append("logs folder inventory:")
            for kind in ("text", "csv", "images", "other"):
                files = inventory.get(kind) or []
                if files:
                    parts.append(f"- {kind}: {', '.join(files)}")

        feedback_text = self.log_reader.read_feedback_text(candidate.folder)
        if feedback_text:
            parts.append("result feedback from logs/feedback.txt:")
            parts.append(feedback_text)

        text_logs = self.log_reader.read_text_log_files(candidate.folder, exclude_names={"feedback.txt"})
        if text_logs:
            parts.append("text/markdown log feedback:")
            for filename, text in text_logs.items():
                parts.append(f"--- logs/{filename} ---")
                parts.append(text)

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

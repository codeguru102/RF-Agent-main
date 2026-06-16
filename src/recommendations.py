from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from tree import SearchTree


def best_trained_summary(tree: SearchTree):
    best = tree.best_node()
    if best is None or best.candidate is None:
        return None
    return {
        "candidate_id": best.candidate_id,
        "score": best.q_leaf_value,
        "q_value": best.q_value,
        "visits": best.visits,
        "reward_file": str(best.candidate.folder / _reward_file_name(best.candidate.metadata)),
        "summary": best.candidate.summary,
    }


def training_recommendations(tree: SearchTree, latest_decisions: Optional[List[dict]] = None) -> List[dict]:
    latest_decisions = latest_decisions or []
    latest_order = {
        decision.get("candidate_id"): index
        for index, decision in enumerate(latest_decisions)
        if decision.get("candidate_id")
    }
    pending = [node for node in tree.pending_nodes() if node.candidate and node.candidate.good_to_train]
    incomplete = [
        node for node in tree.nodes.values()
        if node is not tree.root
        and node.candidate is not None
        and node.candidate.status.get("status") == "trained"
        and node.candidate.summary is None
    ]

    def sort_key(node):
        latest_rank = latest_order.get(node.candidate_id, 10_000)
        generation = int(node.metadata.get("generation", node.depth))
        action_index = int(node.action_index or 0)
        return (latest_rank, generation, node.depth, action_index, node.candidate_id)

    recommendations = []
    rank = 1
    for node in sorted(incomplete, key=lambda item: item.candidate_id):
        recommendations.append(
            {
                "rank": rank,
                "candidate_id": node.candidate_id,
                "action_type": node.action_type,
                "action_index": node.action_index,
                "parent_id": node.parent_id,
                "status": node.candidate.status.get("status", "unknown"),
                "good_to_train": node.candidate.good_to_train,
                "score": node.q_leaf_value,
                "uct_score": None,
                "reward_file": str(node.candidate.folder / _reward_file_name(node.candidate.metadata)),
                "logs_dir": str(node.candidate.folder / "logs"),
                "summary_file": str(node.candidate.folder / "summary.json"),
                "reason": "status is trained, but summary.json is missing; add summary/logs so RF-Agent can score it",
            }
        )
        rank += 1

    for node in sorted(pending, key=sort_key):
        latest = node.candidate_id in latest_order
        reason = "latest RF-Agent generated candidate" if latest else "pending candidate without training summary"
        if node.action_type == "initialize":
            reason += "; initial candidates need offline training before scores/UCT can be meaningful"
        else:
            reason += f"; generated from parent {node.parent_id}"

        recommendations.append(
            {
                "rank": rank,
                "candidate_id": node.candidate_id,
                "action_type": node.action_type,
                "action_index": node.action_index,
                "parent_id": node.parent_id,
                "status": node.candidate.status.get("status", "unknown"),
                "good_to_train": node.candidate.good_to_train,
                "score": node.q_leaf_value,
                "uct_score": None,
                "reward_file": str(node.candidate.folder / _reward_file_name(node.candidate.metadata)),
                "logs_dir": str(node.candidate.folder / "logs"),
                "summary_file": str(node.candidate.folder / "summary.json"),
                "reason": reason,
            }
        )
        rank += 1
    return recommendations


def print_training_recommendations(tree: SearchTree, latest_decisions: Optional[List[dict]] = None):
    best = best_trained_summary(tree)
    print("")
    print("Training recommendation:")

    if best:
        print("Current best trained reward:")
        print(f"- Candidate: {best['candidate_id']}")
        print(f"- Score: {best['score']:.6f}")
        print(f"- Q: {best['q_value']:.6f}")
        print(f"- Reward file: {best['reward_file']}")
    else:
        print("Current best trained reward: none yet")

    recs = training_recommendations(tree, latest_decisions)
    if not recs:
        print("")
        print("No pending reward candidates to train right now.")
        print("Next step: run generation again to create a new candidate from the trained tree.")
        return

    print("")
    print("Train or complete next:")
    for rec in recs:
        print(f"{rec['rank']}. {rec['candidate_id']} ({rec['action_type']}[{rec['action_index']}])")
        print(f"   reward: {rec['reward_file']}")
        print(f"   put logs in: {rec['logs_dir']}")
        print(f"   write summary: {rec['summary_file']}")
        print(f"   why: {rec['reason']}")

    if best is None:
        print("")
        print("Because no candidate has been trained yet, train all pending initial candidates if possible.")
        print("If you can only train one first, start with rank 1, then add its summary/logs and run inspect again.")


def _reward_file_name(metadata: dict) -> str:
    reward_language = str(metadata.get("reward_language", "")).lower()
    default_file = "reward_fcn.m" if reward_language in {"matlab", "m"} else "reward_fcn.py"
    return metadata.get("reward_file", default_file)

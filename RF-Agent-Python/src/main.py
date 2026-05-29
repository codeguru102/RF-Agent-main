from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from candidate_store import CandidateStore
from candidate_store import utc_now
from config import load_json, save_json
from dashboard import render_dashboard
from feedback_builder import FeedbackBuilder
from llm_client import LLMClient
from python_log_reader import PythonLogReader
from recommendations import print_training_recommendations
from rf_agent import OfflineRFAgent
from task_loader import load_task_folder
from tree import SearchTree


def parse_args():
    parser = argparse.ArgumentParser(description="Offline RF-Agent for Python training logs.")
    parser.add_argument("--mode", choices=["inspect", "generate", "sync"], default="inspect")
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=0,
        help="Optional cap for generated candidates. Use 0 to expand the full RF-Agent action bundle.",
    )
    parser.add_argument("--task-dir", default="tasks/python_task")
    parser.add_argument("--agent-config", default="configs/agent.json")
    parser.add_argument("--prompt-dir", default="prompts")
    parser.add_argument("--dry-run", action="store_true", help="Generate placeholder rewards without calling an LLM.")
    return parser.parse_args()


def build_context(args):
    task_config = load_task_folder(Path(args.task_dir))
    agent_config = load_json(args.agent_config)
    store = CandidateStore(Path(args.task_dir) / "candidates")
    log_reader = PythonLogReader(
        task_config.get("score", {}),
        dummy_failure=float(agent_config.get("dummy_failure", -10000.0)),
    )
    candidates = store.scan()
    tree = SearchTree(
        candidates,
        log_reader,
        float(agent_config.get("dummy_failure", -10000.0)),
        max_simulations=int(agent_config.get("simulations", 80)),
    )
    feedback_builder = FeedbackBuilder(log_reader)
    llm_client = LLMClient(
        model=agent_config.get("model", "gpt-4o-mini"),
        temperature=float(agent_config.get("temperature", 1.0)),
        dry_run=args.dry_run,
    )
    agent = OfflineRFAgent(
        task_config=task_config,
        agent_config=agent_config,
        prompt_dir=Path(args.prompt_dir),
        store=store,
        tree=tree,
        feedback_builder=feedback_builder,
        llm_client=llm_client,
    )
    return task_config, agent_config, store, tree, agent


def inspect_state(tree: SearchTree, task_name: str, elite_ids: List[str], elite_limit: int):
    trained = tree.trained_nodes()
    pending = tree.pending_nodes()
    elites = tree.elite_set_nodes(elite_ids, elite_limit)
    print(f"Task: {task_name}")
    print(f"Total candidates: {len(tree.nodes) - 1}")
    print(f"Trained candidates: {len(trained)}")
    print(f"Pending candidates: {len(pending)}")
    print("")
    print("Elite candidates:")
    if not elites:
        print("- none")
    for node in elites:
        print(f"- {node.candidate_id}: score={node.reward_cur:.6f}, action={node.action_type}, parent={node.parent_id}")


def main():
    args = parse_args()
    task_config, agent_config, store, tree, agent = build_context(args)

    if args.mode == "inspect":
        elite_limit = int(agent_config.get("dashboard_elite_max", agent_config.get("elite_control_num", 4)))
        inspect_state(tree, task_config["task_name"], store.load_elite_ids(), elite_limit)
        render_and_print(args, task_config, agent_config, store, tree)
        return

    if args.mode == "sync":
        synced = sync_candidate_summaries(store, tree.log_reader)
        print(f"Synced candidates from CSV logs: {len(synced)}")
        for candidate_id in synced:
            print(f"- {candidate_id}")
        refreshed = CandidateStore(Path(args.task_dir) / "candidates").scan()
        tree = SearchTree(
            refreshed,
            tree.log_reader,
            float(agent_config.get("dummy_failure", -10000.0)),
            max_simulations=int(agent_config.get("simulations", 80)),
        )
        render_and_print(args, task_config, agent_config, store, tree)
        return

    created = agent.generate_batch(args.num_candidates)
    save_json(store.root / "latest_generation.json", agent.last_generation_decisions)
    print("Created pending candidates:")
    for candidate_id in created:
        print(f"- {candidate_id}")
    render_and_print(args, task_config, agent_config, store, agent.tree, agent.last_generation_decisions)


def render_and_print(args, task_config, agent_config, store, tree, latest_decisions=None):
    if latest_decisions is None:
        latest_path = store.root / "latest_generation.json"
        latest_decisions = load_json(latest_path) if latest_path.exists() else []

    task_dir = Path(task_config["_task_dir"])
    best_paths = export_best_reward(task_dir, tree)
    output_dir = task_dir / "visualization"
    paths = render_dashboard(
        tree=tree,
        task_name=task_config["task_name"],
        agent_config=agent_config,
        output_dir=output_dir,
        latest_decisions=latest_decisions,
        elite_node_ids=store.load_elite_ids(),
        best_node_id=best_paths.get("candidate_id"),
        show_window=True,
    )
    print("")
    if best_paths:
        print("Best reward:")
        print(f"- Candidate: {best_paths['candidate_id']}")
        print(f"- Score: {best_paths['score']:.6f}")
        print(f"- File: {best_paths['reward_path']}")
    print("Dashboard written:")
    print(f"- PNG: {paths['png']}")
    print(f"- JSON: {paths['json']}")
    print_training_recommendations(tree, latest_decisions)


def export_best_reward(task_dir: Path, tree: SearchTree):
    best = tree.best_node()
    if best is None or best.candidate is None:
        return {}

    reward_path = task_dir / "best_reward_fcn.py"
    summary_path = task_dir / "best_reward_summary.json"
    reward_path.write_text(best.candidate.reward_code.rstrip() + "\n", encoding="utf-8")
    save_json(
        summary_path,
        {
            "candidate_id": best.candidate_id,
            "score": best.reward_cur,
            "q_value": best.q_value,
            "visits": best.visits,
            "summary": best.candidate.summary,
            "metadata": best.candidate.metadata,
            "reward_file": str(reward_path),
        },
    )
    return {
        "candidate_id": best.candidate_id,
        "score": best.reward_cur,
        "reward_path": reward_path,
    }


def sync_candidate_summaries(store: CandidateStore, log_reader: PythonLogReader):
    synced = []
    for candidate in store.scan():
        if candidate.summary is not None or candidate.is_failed:
            continue

        summary = log_reader.summary_from_csv_logs(candidate.folder)
        if not summary:
            continue

        save_json(candidate.folder / "summary.json", summary)

        status = dict(candidate.status)
        status["status"] = "trained"
        status["updated_at"] = utc_now()
        status.setdefault("error_message", "")
        save_json(candidate.folder / "status.json", status)

        metadata = dict(candidate.metadata)
        metadata["status"] = "trained"
        save_json(candidate.folder / "metadata.json", metadata)

        synced.append(candidate.candidate_id)
    return synced


if __name__ == "__main__":
    main()

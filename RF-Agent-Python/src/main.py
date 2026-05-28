from __future__ import annotations

import argparse
from pathlib import Path

from candidate_store import CandidateStore
from config import load_json
from feedback_builder import FeedbackBuilder
from llm_client import LLMClient
from python_log_reader import PythonLogReader
from rf_agent import OfflineRFAgent
from tree import SearchTree


def parse_args():
    parser = argparse.ArgumentParser(description="Offline RF-Agent for Python training logs.")
    parser.add_argument("--mode", choices=["inspect", "generate"], default="inspect")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--task-config", default="configs/task.json")
    parser.add_argument("--agent-config", default="configs/agent.json")
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--prompt-dir", default="prompts")
    parser.add_argument("--dry-run", action="store_true", help="Generate placeholder rewards without calling an LLM.")
    return parser.parse_args()


def build_context(args):
    task_config = load_json(args.task_config)
    agent_config = load_json(args.agent_config)
    store = CandidateStore(Path(args.experiments_dir), task_config["task_name"])
    log_reader = PythonLogReader(
        task_config.get("score", {}),
        dummy_failure=float(agent_config.get("dummy_failure", -10000.0)),
    )
    candidates = store.scan()
    tree = SearchTree(candidates, log_reader, float(agent_config.get("dummy_failure", -10000.0)))
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


def inspect_state(tree: SearchTree, task_name: str):
    trained = tree.trained_nodes()
    pending = tree.pending_nodes()
    elites = tree.elite_nodes(10)
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
    task_config, _, _, tree, agent = build_context(args)

    if args.mode == "inspect":
        inspect_state(tree, task_config["task_name"])
        return

    created = agent.generate_batch(args.num_candidates)
    print("Created pending candidates:")
    for candidate_id in created:
        print(f"- {candidate_id}")


if __name__ == "__main__":
    main()

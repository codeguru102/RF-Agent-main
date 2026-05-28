from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from candidate_store import CandidateStore
from feedback_builder import FeedbackBuilder
from llm_client import LLMClient
from reward_parser import parse_reward_response
from tree import SearchNode, SearchTree


ACTION_ORDER = [
    "mutation_mechanism",
    "mutation_param",
    "crossover_elite",
    "tree_reasoning",
    "different_thought",
]


class OfflineRFAgent:
    def __init__(
        self,
        *,
        task_config: dict,
        agent_config: dict,
        prompt_dir: Path,
        store: CandidateStore,
        tree: SearchTree,
        feedback_builder: FeedbackBuilder,
        llm_client: LLMClient,
    ):
        self.task_config = task_config
        self.agent_config = agent_config
        self.prompt_dir = Path(prompt_dir)
        self.store = store
        self.tree = tree
        self.feedback_builder = feedback_builder
        self.llm_client = llm_client
        self.action_weights = agent_config.get("action_weights", {})

    def generate_batch(self, num_candidates: int) -> List[str]:
        created = []
        for _ in range(num_candidates):
            parent, action_type, action_index, source_nodes = self._choose_next_action()
            messages = self._build_messages(parent, action_type, source_nodes)
            response = self.llm_client.complete(messages)
            design_thought, reward_code = parse_reward_response(response)
            if not reward_code:
                raise RuntimeError("LLM response did not contain reward code.")

            generation = 0 if parent is None else parent.depth + 1
            parent_id = None if parent is None else parent.candidate_id
            candidate = self.store.create_candidate(
                parent_id=parent_id,
                action_type=action_type,
                action_index=action_index,
                generation=generation,
                reward_language=self.task_config.get("reward_language", "python"),
                reward_code=reward_code,
                design_thought=design_thought,
                prompt_messages=messages,
                source_node_ids=[node.candidate_id for node in source_nodes],
            )
            created.append(candidate.candidate_id)

            # Add a lightweight pending node so the same action slot is not reused in this batch.
            refreshed = self.store.scan()
            self.tree = SearchTree(
                refreshed,
                self.tree.log_reader,
                self.agent_config.get("dummy_failure", -10000.0),
            )
        return created

    def _choose_next_action(self) -> Tuple[Optional[SearchNode], str, int, List[SearchNode]]:
        initial_size = int(self.agent_config.get("initial_size", 6))
        root_initial_count = sum(1 for child in self.tree.root.children if child.action_type == "initialize")
        if root_initial_count < initial_size:
            return None, "initialize", root_initial_count, []

        parent = self._select_parent_for_expansion()
        if parent is None:
            raise RuntimeError(
                "No trained candidates are available for mutation or selection. "
                "Run your Python trainer on the pending initial candidates before generating more."
            )

        action_type, action_index = self._next_untried_action(parent)
        source_nodes = self._source_nodes_for_action(parent, action_type)
        return parent, action_type, action_index, source_nodes

    def _select_parent_for_expansion(self) -> Optional[SearchNode]:
        candidates = [
            node for node in self.tree.trained_nodes()
            if self._next_untried_action(node)[0] is not None
        ]
        if not candidates:
            elites = self.tree.elite_nodes(1)
            return elites[0] if elites else None

        progress = len(self.tree.trained_nodes()) / max(len(self.tree.nodes), 1)
        c_param_init = float(self.agent_config.get("c_param_init", 0.4))
        c_param_final = float(self.agent_config.get("c_param_final", 0.1))
        c_param = (c_param_init - c_param_final) * (1.0 - progress) + c_param_final
        return max(candidates, key=lambda node: self.tree.uct_score(node, c_param))

    def _next_untried_action(self, parent: SearchNode) -> Tuple[Optional[str], int]:
        used = {}
        for child in parent.children:
            key = child.action_type
            used[key] = max(used.get(key, -1), int(child.action_index or 0))

        for action_type in ACTION_ORDER:
            max_count = int(self.action_weights.get(action_type, 1))
            for action_index in range(max_count):
                if used.get(action_type, -1) < action_index:
                    return action_type, action_index
        return None, 0

    def _source_nodes_for_action(self, parent: SearchNode, action_type: str) -> List[SearchNode]:
        if action_type in {"mutation_mechanism", "mutation_param"}:
            return [parent]

        if action_type == "crossover_elite":
            elites = [node for node in self.tree.elite_nodes(int(self.agent_config.get("elite_control_num", 4))) if node != parent]
            return [parent] + elites[: max(int(self.agent_config.get("elite_control_num", 4)) - 1, 0)]

        if action_type == "tree_reasoning":
            path = self.tree.path_to_root(parent)
            max_len = int(self.agent_config.get("tree_reasoning_max_length", 4))
            return path[-max_len:]

        if action_type == "different_thought":
            different = self.tree.branch_excluding(parent)
            max_len = int(self.agent_config.get("different_thought_max_control_num", 4))
            return [parent] + different[: max(max_len - 1, 0)]

        return []

    def _build_messages(self, parent: Optional[SearchNode], action_type: str, source_nodes: List[SearchNode]) -> List[dict]:
        system = self._prompt("initial_system.txt") + "\n\n" + self._prompt("output_format.txt")
        user_parts = [self._base_user_prompt()]

        if action_type == "initialize":
            pass
        elif action_type in {"mutation_mechanism", "mutation_param"}:
            prompt_file = "mutation_mechanism.txt" if action_type == "mutation_mechanism" else "mutation_param.txt"
            source = source_nodes[0]
            user_parts.append(
                self._prompt(prompt_file).format(
                    design_thought=source.metadata.get("design_thought", ""),
                    reward_function=source.candidate.reward_code,
                    trained_results=self.feedback_builder.build(source.candidate),
                )
            )
        elif action_type == "crossover_elite":
            user_parts.append(
                self._prompt("crossover_elite.txt").format(
                    reward_func_group=self._format_node_group(source_nodes)
                )
            )
        elif action_type == "tree_reasoning":
            user_parts.append(
                self._prompt("tree_reasoning.txt").format(
                    reward_func_group=self._format_node_group(source_nodes)
                )
            )
        elif action_type == "different_thought":
            user_parts.append(
                self._prompt("different_thought.txt").format(
                    reward_func_group=self._format_node_group(source_nodes)
                )
            )
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        user_parts.append(self._prompt("result_analysis.txt"))
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(part for part in user_parts if part.strip())},
        ]

    def _base_user_prompt(self) -> str:
        variables = "\n".join(f"- {item}" for item in self.task_config.get("available_variables", []))
        score = self.task_config.get("score", {})
        score_description = f"primary={score.get('primary', 'max_task_score')}, maximize={score.get('maximize', True)}"
        return self._prompt("initial_user.txt").format(
            task_description=self.task_config.get("description", ""),
            reward_signature=self.task_config.get("reward_signature", ""),
            available_variables=variables,
            score_description=score_description,
        )

    def _format_node_group(self, nodes: List[SearchNode]) -> str:
        chunks = []
        for index, node in enumerate(nodes, start=1):
            chunks.append(
                "\n".join(
                    [
                        f"Candidate {index}: {node.candidate_id}",
                        f"Action: {node.action_type}",
                        f"Score: {node.reward_cur:.6f}",
                        f"Design thought: {node.metadata.get('design_thought', '')}",
                        "Reward:",
                        node.candidate.reward_code,
                        "Training feedback:",
                        self.feedback_builder.build(node.candidate),
                    ]
                )
            )
        return "\n\n".join(chunks)

    def _prompt(self, filename: str) -> str:
        return (self.prompt_dir / filename).read_text(encoding="utf-8").strip()

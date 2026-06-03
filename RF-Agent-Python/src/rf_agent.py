from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from candidate_store import CandidateStore
from console import Spinner
from feedback_builder import FeedbackBuilder
from llm_client import LLMClient
from reward_parser import parse_reward_response
from reward_validator import validate_reward_code
from tree import SearchNode, SearchTree


ACTION_ORDER = [
    "mutation",
    "crossover_elite",
    "tree_reasoning",
    "different_thought",
]

DEFAULT_ACTION_PROBABILITIES = {
    "mutation": 0.30,
    "crossover_elite": 0.30,
    "tree_reasoning": 0.20,
    "different_thought": 0.20,
}


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
        self.expansion_size = int(agent_config.get("expansion_size", 4))
        self.action_probabilities = self._normalize_action_probabilities(
            agent_config.get("action_probabilities", DEFAULT_ACTION_PROBABILITIES)
        )
        self.max_try_num = int(agent_config.get("max_try_num", 9))
        self.max_same_try_cnt = int(agent_config.get("max_same_try_cnt", 3))
        self.enable_self_verify = bool(agent_config.get("enable_self_verify", True))
        random_seed = agent_config.get("random_seed")
        self.random = random.Random(random_seed)
        self.last_generation_decisions: List[dict] = []
        self.last_selected_node_id: Optional[str] = None
        self.last_selection_path: List[dict] = []

    def generate_batch(self, num_candidates: int) -> List[str]:
        created = []
        self.last_generation_decisions = []
        selection_c_param = self._current_c_param()
        parent = self._select_node_for_expansion(selection_c_param)
        self.last_selected_node_id = None if parent is None else parent.candidate_id
        if parent is not None:
            self._remember_elite_parent(parent)

        action_plan = self._action_plan_for_selected_node(parent, num_candidates)
        total_actions = len(action_plan)
        parent_id = "root" if parent is None else parent.candidate_id
        model = self.agent_config.get("model", "unknown")
        print(
            f"PROGRESS total={total_actions} step=0 from={parent_id} model={model}",
            flush=True,
        )
        for action_type, action_index, source_nodes in action_plan:
            print(
                f"PROGRESS total={total_actions} step={len(created)} "
                f"from={parent_id} action={action_type}[{action_index}] "
                f"model={model} phase=generating",
                flush=True,
            )
            parent_uct_at_selection = None if parent is None else self.tree.uct_score(parent, selection_c_param)
            messages = self._build_messages(parent, action_type, source_nodes)
            label = self._operation_label("OpenAI reward generation", parent, action_type, action_index)
            with Spinner(label, enabled=not self.llm_client.dry_run) as spinner:
                design_thought, reward_code, response, validation_attempts = self._generate_valid_reward(
                    messages,
                    status=spinner.update,
                )
                spinner.succeed(f"Generated {action_type}[{action_index}] in {validation_attempts} attempt(s)")

            self_verify_score = 0.0
            if self.enable_self_verify and not self.llm_client.dry_run:
                label = self._operation_label("OpenAI self verification", parent, action_type, action_index)
                with Spinner(label) as spinner:
                    self_verify_score = self._self_verify_reward(design_thought, reward_code)
                    spinner.succeed(f"Self-verify score for {action_type}[{action_index}]: {self_verify_score:.3f}")

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
                extra_metadata={
                    "self_verify_score": self_verify_score,
                    "validation_attempts": validation_attempts,
                    "raw_response": response,
                },
            )
            created.append(candidate.candidate_id)
            print(
                f"PROGRESS total={total_actions} step={len(created)} "
                f"id={candidate.candidate_id} action={action_type}[{action_index}]",
                flush=True,
            )
            self.last_generation_decisions.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "parent_id": parent_id,
                    "action_type": action_type,
                    "action_index": action_index,
                    "source_node_ids": [node.candidate_id for node in source_nodes],
                    "selection_c_param": selection_c_param,
                    "parent_uct_at_selection": parent_uct_at_selection,
                    "selected_tree_node_id": self.last_selected_node_id,
                    "selection_path": self.last_selection_path,
                    "self_verify_score": self_verify_score,
                    "validation_attempts": validation_attempts,
                }
            )
        refreshed = self.store.scan()
        self.tree = SearchTree(
            refreshed,
            self.tree.log_reader,
            self.agent_config.get("dummy_failure", -10000.0),
            max_simulations=int(self.agent_config.get("simulations", 80)),
        )
        return created

    def _select_node_for_expansion(self, c_param: float) -> Optional[SearchNode]:
        initial_size = int(self.agent_config.get("initial_size", 6))
        root_initial_count = sum(1 for child in self.tree.root.children if child.action_type == "initialize")
        if root_initial_count == 0:
            self.last_selection_path = [
                {
                    "node_id": "root",
                    "uct_score": None,
                    "reason": "root initial expansion",
                }
            ]
            return None

        if not self.tree.trained_nodes():
            raise RuntimeError(
                "No trained candidates are available for mutation or selection. "
                "Run your Python trainer on the pending initial candidates before generating more."
            )

        node = self.tree.root
        path = [{"node_id": "root", "uct_score": None, "reason": "start"}]
        max_depth = int(self.agent_config.get("tree_max_depth", 16))
        while node.children and node.depth < max_depth:
            pending_children = [child for child in node.children if child.candidate and child.candidate.is_pending]
            if pending_children:
                pending_ids = ", ".join(child.candidate_id for child in pending_children)
                raise RuntimeError(
                    f"Node {node.candidate_id} has pending children. Original RF-Agent waits for the "
                    f"whole expanded batch to finish before selecting again. Train these first: {pending_ids}"
                )
            selectable_children = self.tree.selectable_children(node)
            if not selectable_children:
                break

            best_child = max(selectable_children, key=lambda child: self.tree.uct_score(child, c_param))
            best_child_uct = self.tree.uct_score(best_child, c_param)
            path.append(
                {
                    "node_id": best_child.candidate_id,
                    "uct_score": best_child_uct,
                    "reason": "max UCT child",
                }
            )
            node = best_child

        if node is self.tree.root:
            raise RuntimeError("No trained child can be selected from the root.")

        if node.children:
            raise RuntimeError(
                f"Selected node {node.candidate_id} is not a leaf. Original RF-Agent expands only selected leaves."
            )

        self.last_selection_path = path
        return node

    def _action_plan_for_selected_node(self, parent: Optional[SearchNode], num_candidates: int) -> List[Tuple[str, int, List[SearchNode]]]:
        initial_size = int(self.agent_config.get("initial_size", 6))
        if parent is None:
            root_initial_count = sum(1 for child in self.tree.root.children if child.action_type == "initialize")
            actions = [("initialize", root_initial_count + index, []) for index in range(initial_size - root_initial_count)]
            return self._maybe_cap_action_plan(actions, num_candidates)

        if parent.children:
            raise RuntimeError(
                f"Selected node {parent.candidate_id} already has children. Original RF-Agent expands a leaf once."
            )
        actions = self._full_action_bundle()
        planned = []
        for action_type, action_index in actions:
            planned.append((action_type, action_index, self._source_nodes_for_action(parent, action_type)))
        return self._maybe_cap_action_plan(planned, num_candidates)

    def _current_c_param(self) -> float:
        max_simulations = int(self.agent_config.get("simulations", 80))
        progress = min(len(self.tree.trained_nodes()) / max(max_simulations, 1), 1.0)
        c_param_init = float(self.agent_config.get("c_param_init", 0.4))
        c_param_final = float(self.agent_config.get("c_param_final", 0.1))
        return (c_param_init - c_param_final) * (1.0 - progress) + c_param_final

    def _available_actions(self, parent: SearchNode) -> List[Tuple[str, int]]:
        used = set()
        for child in parent.children:
            if child.action_type is None:
                continue
            used.add((child.action_type, int(child.action_index or 0)))

        return [
            (action_type, action_index)
            for action_type, action_index in self._full_action_bundle()
            if (action_type, action_index) not in used
        ]

    def _full_action_bundle(self) -> List[Tuple[str, int]]:
        action_types = list(self.action_probabilities)
        weights = [self.action_probabilities[action_type] for action_type in action_types]
        sampled = self.random.choices(action_types, weights=weights, k=max(self.expansion_size, 0))
        counts: Dict[str, int] = {}
        actions = []
        for action_type in sampled:
            action_index = counts.get(action_type, 0)
            counts[action_type] = action_index + 1
            actions.append((action_type, action_index))
        return actions

    def _normalize_action_probabilities(self, probabilities: Dict[str, float]) -> Dict[str, float]:
        probabilities = probabilities or DEFAULT_ACTION_PROBABILITIES
        normalized = {}
        for action_type in ACTION_ORDER:
            value = probabilities.get(action_type, DEFAULT_ACTION_PROBABILITIES[action_type])
            value = max(float(value), 0.0)
            if value > 0:
                normalized[action_type] = value

        total = sum(normalized.values())
        if total <= 0:
            return dict(DEFAULT_ACTION_PROBABILITIES)
        return {action_type: value / total for action_type, value in normalized.items()}

    def _source_nodes_for_action(self, parent: SearchNode, action_type: str) -> List[SearchNode]:
        if action_type in {"mutation", "mutation_mechanism", "mutation_param"}:
            return [parent]

        if action_type == "crossover_elite":
            elites = self._elite_set_nodes()
            max_count = int(self.agent_config.get("elite_control_num", 4))
            count = self.random.randint(2, max(max_count, 2)) - 1
            return self._weighted_elite_choices(elites, count) + [parent]

        if action_type == "tree_reasoning":
            path = self.tree.path_to_root(parent)
            max_len = int(self.agent_config.get("tree_reasoning_max_length", 4))
            return path[-max_len:]

        if action_type == "different_thought":
            return self._different_thought_sources(parent)

        return []

    def _build_messages(self, parent: Optional[SearchNode], action_type: str, source_nodes: List[SearchNode]) -> List[dict]:
        system = self._prompt("initial_system.txt") + "\n\n" + self._prompt("output_format.txt")
        user_parts = [self._base_user_prompt()]

        if action_type == "initialize":
            pass
        elif action_type == "mutation":
            source = source_nodes[0]
            user_parts.append(self._format_mutation_prompt(source))
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
            task_context=self.task_config.get("task_context", ""),
            reward_signature=self.task_config.get("reward_signature", ""),
            available_variables=variables,
            score_description=score_description,
        )

    def _format_mutation_prompt(self, source: SearchNode) -> str:
        return "\n".join(
            [
                "Improve the reward with one integrated mutation pass.",
                "You may change reward mechanisms, add/remove/restructure components, and tune numeric scales, temperatures, tolerances, bonuses, or penalties in the same revision.",
                "",
                "Parent design thought:",
                source.metadata.get("design_thought", ""),
                "",
                "Parent reward:",
                source.candidate.reward_code,
                "",
                "Parent training feedback:",
                self.feedback_builder.build(source.candidate),
                "",
                "Focus on a useful combined update, not on preserving a strict boundary between mechanism changes and parameter tuning.",
            ]
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

    def _generate_valid_reward(
        self,
        messages: List[dict],
        status: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, str, str, int]:
        current_messages = list(messages)
        same_try_cnt = 0
        last_error = ""
        for attempt in range(1, self.max_try_num + 1):
            if same_try_cnt >= self.max_same_try_cnt:
                current_messages = list(messages)
                same_try_cnt = 0

            if status is not None:
                status(f"OpenAI reward generation attempt {attempt}/{self.max_try_num}")
            response = self.llm_client.complete(current_messages)
            design_thought, reward_code = parse_reward_response(response)
            if not reward_code:
                last_error = "LLM response did not contain reward code."
            else:
                validation = validate_reward_code(reward_code, self.task_config)
                if validation.valid:
                    return design_thought, reward_code, response, attempt
                last_error = validation.message

            current_messages = self._build_retry_messages(messages, reward_code, last_error)
            same_try_cnt += 1
            if status is not None:
                status(f"Retrying with validation feedback: {last_error}")

        raise RuntimeError(
            "Could not generate a valid reward function after "
            f"{self.max_try_num} attempts. Last validation error: {last_error}"
        )

    def _build_retry_messages(self, base_messages: List[dict], reward_code: str, error_message: str) -> List[dict]:
        feedback_path = self.prompt_dir / "initial_failed_feedback.txt"
        if feedback_path.exists():
            feedback = feedback_path.read_text(encoding="utf-8").strip().format(
                reward_function=reward_code or "(no reward code parsed)",
                traceback_msg=error_message,
            )
        else:
            feedback = "\n".join(
                [
                    "The generated reward function is invalid.",
                    f"Reward function:\n{reward_code or '(no reward code parsed)'}",
                    f"Validation error:\n{error_message}",
                    "Return one corrected JSON object with design_thought and reward_code.",
                ]
            )
        return [
            base_messages[0],
            {
                "role": "user",
                "content": base_messages[1]["content"] + "\n\n" + feedback,
            },
        ]

    def _self_verify_reward(self, design_thought: str, reward_code: str) -> float:
        if not self.enable_self_verify or self.llm_client.dry_run:
            return 0.0

        system_path = self.prompt_dir / "initial_system_verify.txt"
        prompt_path = self.prompt_dir / "self_node_value_verify_single.txt"
        if not system_path.exists() or not prompt_path.exists():
            return 0.0

        messages = [
            {"role": "system", "content": system_path.read_text(encoding="utf-8").strip()},
            {
                "role": "user",
                "content": self._base_user_prompt()
                + "\n\n"
                + prompt_path.read_text(encoding="utf-8").strip().format(
                    design_thought=design_thought,
                    reward_function=reward_code,
                ),
            },
        ]
        response = self.llm_client.complete(messages)
        bracket_values = re.findall(r"\[([-+]?\d*\.?\d+)\]", response)
        if bracket_values:
            return float(bracket_values[-1])
        number = re.search(r"[-+]?\d*\.?\d+", response)
        return float(number.group(0)) if number else 0.0

    def _weighted_elite_choices(self, nodes: List[SearchNode], count: int) -> List[SearchNode]:
        if not nodes or count <= 0:
            return []
        bias = float(self.agent_config.get("elite_weight_bias", 1.0))
        weights = [1.0 / (rank + 1.0 + bias) for rank in range(len(nodes))]
        return self.random.choices(nodes, weights=weights, k=count)

    def _different_thought_sources(self, parent: SearchNode) -> List[SearchNode]:
        max_count = int(self.agent_config.get("different_thought_max_control_num", 4))
        count = self.random.randint(2, max(max_count, 2)) - 1
        current_root_child = self._root_branch_child(parent)
        available_subtrees = [
            child
            for child in self.tree.root.children
            if child.is_trained and child is not current_root_child
        ]
        selected_subtrees = self.random.sample(available_subtrees, min(count, len(available_subtrees)))

        nodes = []
        for subtree in selected_subtrees:
            explore_depth = self.random.randint(0, self._max_trained_depth_below(subtree))
            selected = subtree
            for _ in range(explore_depth):
                trained_children = [child for child in selected.children if child.is_trained]
                if not trained_children:
                    break
                selected = max(trained_children, key=lambda child: child.reward_cur)
            nodes.append(selected)

        nodes.append(parent)
        return nodes

    def _root_branch_child(self, node: SearchNode) -> SearchNode:
        current = node
        while current.parent_id and current.parent_id != "root":
            current = self.tree.nodes.get(current.parent_id, current)
        return current

    def _max_trained_depth_below(self, node: SearchNode) -> int:
        trained_children = [child for child in node.children if child.is_trained]
        if not trained_children:
            return 0
        return 1 + max(self._max_trained_depth_below(child) for child in trained_children)

    def _elite_set_nodes(self) -> List[SearchNode]:
        return self.tree.elite_set_nodes(
            self.store.load_elite_ids(),
            int(self.agent_config.get("elite_max_length", 10)),
        )

    def _remember_elite_parent(self, parent: SearchNode) -> None:
        elite_ids = self.store.load_elite_ids()
        if parent.candidate_id not in elite_ids:
            elite_ids.append(parent.candidate_id)

        elite_nodes = self.tree.elite_set_nodes(elite_ids)
        max_length = int(self.agent_config.get("elite_max_length", 10))
        elite_nodes = sorted(elite_nodes, key=lambda node: node.reward_cur, reverse=True)[:max_length]
        self.store.save_elite_ids(node.candidate_id for node in elite_nodes)

    def _maybe_cap_action_plan(
        self,
        action_plan: List[Tuple[str, int, List[SearchNode]]],
        num_candidates: int,
    ) -> List[Tuple[str, int, List[SearchNode]]]:
        if num_candidates and num_candidates > 0:
            return action_plan[:num_candidates]
        return action_plan

    def _operation_label(
        self,
        prefix: str,
        parent: Optional[SearchNode],
        action_type: str,
        action_index: int,
    ) -> str:
        if parent is None:
            return f"{prefix}: {action_type}[{action_index}] from root"
        return f"{prefix}: {action_type}[{action_index}] from {parent.candidate_id}"

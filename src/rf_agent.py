from __future__ import annotations

import csv
import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from candidate_store import CandidateStore
from console import Spinner
from feedback_builder import FeedbackBuilder
from llm_client import IMAGE_EXTENSIONS, LLMClient
from reward_parser import parse_reward_response
from reward_validator import validate_reward_code
from tree import SearchNode, SearchTree


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return ""


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


ACTION_ORDER = [
    "mutation",
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
        self.max_try_num = int(agent_config.get("max_try_num", 1))
        self.max_same_try_cnt = int(agent_config.get("max_same_try_cnt", 3))
        self.enable_self_verify = bool(agent_config.get("enable_self_verify", True))
        self.include_images = bool(agent_config.get("include_images", True))
        self.max_images_per_request = int(agent_config.get("max_images_per_request", 8))
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

        action_plan = self._action_plan_for_selected_node(parent, num_candidates)
        total_actions = len(action_plan)
        parent_id = "root" if parent is None else parent.candidate_id
        model = self.agent_config.get("model", "unknown")
        llm_name = self.llm_client.display_name()
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
            label = self._operation_label(f"{llm_name} reward generation", parent, action_type, action_index)
            with Spinner(label, enabled=not self.llm_client.dry_run) as spinner:
                design_thought, reward_code, response, validation_attempts = self._generate_valid_reward(
                    messages,
                    status=spinner.update,
                )
                spinner.succeed(f"Generated {action_type}[{action_index}] in {validation_attempts} attempt(s)")

            self_verify_score = 0.0
            if self.enable_self_verify and not self.llm_client.dry_run:
                label = self._operation_label(f"{llm_name} self verification", parent, action_type, action_index)
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
            reward_language = self.task_config.get("reward_language", "python")
            raise RuntimeError(
                "No trained candidates are available for mutation or selection. "
                f"Run your {reward_language} trainer on the pending initial candidates before generating more."
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

    def _full_action_bundle(self) -> List[Tuple[str, int]]:
        return [(action_type, 0) for action_type in ACTION_ORDER]

    def _source_nodes_for_action(self, parent: SearchNode, action_type: str) -> List[SearchNode]:
        if action_type in {"mutation", "mutation_mechanism", "mutation_param"}:
            return [parent]

        if action_type == "crossover_elite":
            elites = self._elite_nodes()
            max_count = int(self.agent_config.get("elite_control_num", 4))
            count = self.random.randint(2, max(max_count, 2)) - 1
            elites = [node for node in elites if node.candidate_id != parent.candidate_id]
            return self._weighted_elite_choices(elites, count) + [parent]

        if action_type == "tree_reasoning":
            path = self.tree.path_to_root(parent)
            max_len = int(self.agent_config.get("tree_reasoning_max_length", 4))
            return path[-max_len:]

        if action_type == "different_thought":
            return self._different_thought_sources(parent)

        return []

    def _build_messages(self, parent: Optional[SearchNode], action_type: str, source_nodes: List[SearchNode]) -> List[dict]:
        system_parts = [
            self._prompt("initial_system.txt"),
            self._optional_prompt("code_output_tip.txt"),
            self._prompt("output_format.txt"),
        ]
        system = "\n\n".join(part for part in system_parts if part.strip())
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
                self._format_prompt(
                    prompt_file,
                    source_node=source,
                    source_nodes=[source],
                    reward_function=source.candidate.reward_code,
                    trained_results=self.feedback_builder.build(source.candidate),
                )
            )
        elif action_type == "crossover_elite":
            user_parts.append(
                self._format_prompt(
                    "crossover_elite.txt",
                    nums=len(source_nodes),
                    source_nodes=source_nodes,
                    reward_func_group=self._format_node_group(source_nodes)
                )
            )
        elif action_type == "tree_reasoning":
            user_parts.append(
                self._format_prompt(
                    "tree_reasoning.txt",
                    nums=len(source_nodes),
                    source_nodes=source_nodes,
                    reward_func_group=self._format_node_group(source_nodes)
                )
            )
        elif action_type == "different_thought":
            user_parts.append(
                self._format_prompt(
                    "different_thought.txt",
                    nums=len(source_nodes),
                    source_nodes=source_nodes,
                    reward_func_group=self._format_node_group(source_nodes)
                )
            )
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        user_parts.append(self._prompt("result_analysis.txt"))
        user_message = {
            "role": "user",
            "content": "\n\n".join(part for part in user_parts if part.strip()),
        }
        images = self._collect_source_images(parent, source_nodes)
        if images:
            user_message["images"] = images
        return [
            {"role": "system", "content": system},
            user_message,
        ]

    def _collect_source_images(self, parent: Optional[SearchNode], source_nodes: List[SearchNode]) -> List[str]:
        if not self.include_images:
            return []

        nodes: List[SearchNode] = []
        if parent is not None:
            nodes.append(parent)
        nodes.extend(source_nodes)

        images: List[str] = []
        seen = set()
        for node in nodes:
            candidate = getattr(node, "candidate", None)
            if candidate is None:
                continue
            for image_path in self._discover_images(candidate.folder):
                key = str(image_path)
                if key in seen:
                    continue
                seen.add(key)
                images.append(key)
                if len(images) >= self.max_images_per_request:
                    return images
        return images

    def _discover_images(self, folder: Optional[Path]) -> List[Path]:
        if folder is None or not Path(folder).exists():
            return []
        found = [
            path
            for path in sorted(Path(folder).rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        return found

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
                        f"Score: {node.q_leaf_value:.6f}",
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

    def _optional_prompt(self, filename: str) -> str:
        path = self.prompt_dir / filename
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def _format_prompt(self, filename: str, **kwargs) -> str:
        source_nodes = kwargs.pop("source_nodes", None)
        source_node = kwargs.pop("source_node", None)
        if source_nodes is None and source_node is not None:
            source_nodes = [source_node]
        context = self._prompt_context(source_nodes)
        if source_node is not None:
            design_idea = source_node.metadata.get("design_thought", "")
            context.update(
                {
                    "design_idea": design_idea,
                    "design_thought": design_idea,
                }
            )
        context.update(kwargs)
        return self._prompt(filename).format_map(_SafeFormatDict(context))

    def _prompt_context(self, source_nodes: Optional[List[SearchNode]] = None) -> Dict[str, object]:
        return {
            "epoch_freq": self._csv_cadence_description(source_nodes or []),
            "trained_result_analysis_tip": self._optional_prompt("result_analysis.txt"),
        }

    def _csv_cadence_description(self, source_nodes: List[SearchNode]) -> str:
        explicit = self._config_value("csv_row_cadence", "csv_cadence", "training_log_cadence")
        if explicit not in (None, ""):
            return str(explicit)

        row_interval = self._config_value("csv_row_interval_steps", "csv_step_interval", "csv_row_interval")
        if row_interval not in (None, ""):
            unit = str(self._config_value("csv_row_unit", "csv_unit") or "environment step")
            return f"each CSV row represents {self._format_number(row_interval)} {self._pluralize(unit, row_interval)}"

        legacy_epoch_freq = self.agent_config.get("epoch_freq", self.agent_config.get("eval_epoch_freq", self.agent_config.get("train_epoch_freq")))
        if legacy_epoch_freq not in (None, ""):
            return f"each CSV row represents {self._format_number(legacy_epoch_freq)} training {self._pluralize('epoch', legacy_epoch_freq)}"

        inferred = self._infer_csv_cadence(source_nodes)
        if inferred:
            return inferred

        return "each CSV row represents 1 environment step"

    def _config_value(self, *names: str):
        for config in (self.task_config, self.agent_config):
            for name in names:
                if name in config:
                    return config[name]
        return None

    def _infer_csv_cadence(self, source_nodes: List[SearchNode]) -> str:
        for node in source_nodes:
            candidate = node.candidate
            if candidate is None:
                continue
            logs_dir = candidate.folder / "logs"
            if not logs_dir.exists():
                continue
            for csv_path in sorted(logs_dir.glob("*.csv")):
                inferred = self._infer_csv_cadence_from_file(csv_path)
                if inferred:
                    return inferred
        return ""

    def _infer_csv_cadence_from_file(self, csv_path: Path) -> str:
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.reader(file)
                fieldnames = next(reader, None)
                if not fieldnames:
                    return ""
                cadence_column = self._cadence_column(fieldnames)
                if not cadence_column:
                    return ""
                cadence_index = fieldnames.index(cadence_column)
                values = []
                for row_index, row in enumerate(reader):
                    if row_index >= 1000:
                        break
                    if cadence_index >= len(row):
                        continue
                    try:
                        values.append(float(row[cadence_index]))
                    except (TypeError, ValueError):
                        continue
        except OSError:
            return ""

        diffs = [
            values[index] - values[index - 1]
            for index in range(1, len(values))
            if values[index] > values[index - 1]
        ]
        if not diffs:
            return ""
        diffs = sorted(diffs)
        delta = diffs[len(diffs) // 2]
        unit = self._cadence_unit(cadence_column)
        return (
            f"each CSV row represents about {self._format_number(delta)} "
            f"{self._pluralize(unit, delta)} based on the '{cadence_column}' column"
        )

    def _cadence_column(self, fieldnames: List[str]) -> str:
        priority = [
            "step",
            "global_step",
            "env_step",
            "environment_step",
            "timestep",
            "time_step",
            "simulation_step",
            "epoch",
            "episode",
            "iteration",
            "iter",
            "time",
            "t",
        ]
        normalized = {_normalize_name(name): name for name in fieldnames}
        for name in priority:
            if name in normalized:
                return normalized[name]
        return ""

    def _cadence_unit(self, column_name: str) -> str:
        normalized = _normalize_name(column_name)
        if "epoch" in normalized:
            return "training epoch"
        if "episode" in normalized:
            return "episode"
        if normalized in {"time", "t"} or "time" in normalized:
            return "time unit"
        return "environment step"

    def _format_number(self, value) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.6g}"

    def _pluralize(self, unit: str, value) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 2.0
        if abs(numeric - 1.0) < 1e-9:
            return unit
        if unit.endswith("s"):
            return unit
        return unit + "s"

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
                status(f"{self.llm_client.display_name()} reward generation attempt {attempt}/{self.max_try_num}")
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
            feedback = feedback_path.read_text(encoding="utf-8").strip().format_map(
                _SafeFormatDict(
                    {
                        "reward_function": reward_code or "(no reward code parsed)",
                        "traceback_msg": error_message,
                    }
                )
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
        retry_user = {
            "role": "user",
            "content": base_messages[1]["content"] + "\n\n" + feedback,
        }
        if base_messages[1].get("images"):
            retry_user["images"] = base_messages[1]["images"]
        return [
            base_messages[0],
            retry_user,
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
                + prompt_path.read_text(encoding="utf-8").strip().format_map(
                    _SafeFormatDict(
                        {
                            "design_thought": design_thought,
                            "design_idea": design_thought,
                            "reward_function": reward_code,
                        }
                    )
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
                selected = max(trained_children, key=lambda child: child.q_leaf_value)
            nodes.append(selected)

        nodes.append(parent)
        return nodes

    def _root_branch_child(self, node: SearchNode) -> SearchNode:
        current = node
        visited = set()
        chain = []
        while current.parent_id and current.parent_id != "root":
            if current.candidate_id in visited:
                chain_text = " -> ".join(chain + [current.candidate_id])
                raise RuntimeError(
                    f"Parent cycle detected while finding root branch for {node.candidate_id}: {chain_text}"
                )
            visited.add(current.candidate_id)
            chain.append(current.candidate_id)

            parent = self.tree.nodes.get(current.parent_id)
            if parent is None:
                raise RuntimeError(
                    f"Candidate {current.candidate_id} references missing parent {current.parent_id}."
                )
            if parent is current:
                raise RuntimeError(f"Candidate {current.candidate_id} is its own parent.")

            current = parent
        return current

    def _max_trained_depth_below(self, node: SearchNode) -> int:
        trained_children = [child for child in node.children if child.is_trained]
        if not trained_children:
            return 0
        return 1 + max(self._max_trained_depth_below(child) for child in trained_children)

    def _elite_nodes(self) -> List[SearchNode]:
        return self.tree.elite_nodes(int(self.agent_config.get("elite_max_length", 10)))

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

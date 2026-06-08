from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from candidate_store import Candidate


@dataclass
class SearchNode:
    candidate: Optional[Candidate]
    candidate_id: str
    parent_id: Optional[str]
    children: List["SearchNode"] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    q_value: float = 0.0
    q_leaf_value: float = 0.0
    reward_cur: float = -10000.0
    depth: int = 0

    @property
    def metadata(self):
        return self.candidate.metadata if self.candidate else {}

    @property
    def action_type(self):
        return self.metadata.get("action_type")

    @property
    def action_index(self):
        return self.metadata.get("action_index")

    @property
    def self_verify_score(self):
        try:
            return float(self.metadata.get("self_verify_score", 0.0))
        except (TypeError, ValueError):
            return 0.0

    @property
    def is_trained(self):
        return self.candidate.is_trained if self.candidate else False


class SearchTree:
    def __init__(
        self,
        candidates: List[Candidate],
        log_reader: PythonLogReader,
        dummy_failure: float,
        max_simulations: Optional[int] = None,
    ):
        self.log_reader = log_reader
        self.dummy_failure = dummy_failure
        self.update_best_child_gamma = 0.7
        self.update_mean_gamma = 0.15
        self.max_simulations = max_simulations
        self.min_q = 0.0
        self.max_q = 0.0
        self.root = SearchNode(candidate=None, candidate_id="root", parent_id=None, reward_cur=dummy_failure)
        self.nodes: Dict[str, SearchNode] = {"root": self.root}
        self._build(candidates)
        self._recompute_statistics()

    def _build(self, candidates: List[Candidate]):
        for candidate in candidates:
            score = self.log_reader.score_summary(candidate.summary)
            q_leaf_value = self.log_reader.q_value_for_candidate(candidate.folder, score)
            node = SearchNode(
                candidate=candidate,
                candidate_id=candidate.candidate_id,
                parent_id=candidate.parent_id,
                reward_cur=score,
                q_value=q_leaf_value,
                q_leaf_value=q_leaf_value,
            )
            self.nodes[candidate.candidate_id] = node

        for node in list(self.nodes.values()):
            if node is self.root:
                continue
            parent = self.nodes.get(node.parent_id or "root", self.root)
            parent.children.append(node)

        self._assign_depths(self.root, 0)

    def _assign_depths(self, node: SearchNode, depth: int):
        node.depth = depth
        for child in node.children:
            self._assign_depths(child, depth + 1)

    def _recompute_statistics(self, record_q_history: bool = False) -> Optional[Dict[str, List[dict]]]:
        histories: Optional[Dict[str, List[dict]]] = (
            {nid: [] for nid in self.nodes if nid != "root"} if record_q_history else None
        )

        for node in self.nodes.values():
            node.visits = 0
            node.total_reward = 0.0
            node.q_value = 0.0

        self.min_q = 0.0
        self.max_q = 0.0
        trained = sorted(self.trained_nodes(), key=self._replay_sort_key)
        max_simulations = self.max_simulations or max(len(trained), 1)

        for sim_index, node in enumerate(trained, start=1):
            reward = node.reward_cur
            q_value = node.q_leaf_value
            self._backpropagate_result(
                node,
                reward,
                q_value,
                sim_index,
                max_simulations,
                histories=histories,
            )
        return histories

    def recompute_q_histories(self) -> Dict[str, List[dict]]:
        """Replay trained nodes and return per-node Q snapshots after each simulation."""
        return self._recompute_statistics(record_q_history=True) or {}

    def _record_q_snapshot(
        self,
        histories: Optional[Dict[str, List[dict]]],
        node_id: str,
        sim_index: int,
        role: str,
        trained_leaf_id: str,
    ):
        if histories is None or node_id == "root" or node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        histories[node_id].append(
            {
                "sim": sim_index,
                "q": node.q_value,
                "visits": node.visits,
                "role": role,
                "trained_leaf": trained_leaf_id,
            }
        )

    def _backpropagate_result(
        self,
        node: SearchNode,
        reward: float,
        q_value: float,
        sim_index: int,
        max_simulations: int,
        histories: Optional[Dict[str, List[dict]]] = None,
    ):
        leaf_id = node.candidate_id
        node.reward_cur = reward
        node.q_value = q_value
        node.visits += 1
        node.total_reward += q_value
        self.min_q = min(self.min_q, node.q_value)
        self.max_q = max(self.max_q, node.q_value)
        self._record_q_snapshot(histories, leaf_id, sim_index, "leaf", leaf_id)

        parent = self.nodes.get(node.parent_id or "root")
        while parent and parent is not self.root:
            parent.visits += 1
            parent.total_reward += q_value
            q_mean_value = parent.total_reward / parent.visits
            trained_child_qs = [child.q_value for child in parent.children if child.is_trained or child.visits > 0]
            best_child_q = max(trained_child_qs) if trained_child_qs else 0.0
            decay = max(0.0, 1.0 - float(sim_index / max(max_simulations, 1)))
            update_mean_gamma = self.update_mean_gamma * decay
            parent.q_value = (
                (1.0 - self.update_best_child_gamma - update_mean_gamma) * parent.q_value
                + self.update_best_child_gamma * best_child_q
                + update_mean_gamma * q_mean_value
            )
            self._record_q_snapshot(histories, parent.candidate_id, sim_index, "ancestor", leaf_id)
            parent = self.nodes.get(parent.parent_id or "root")

        self.root.visits += 1

    def _replay_sort_key(self, node: SearchNode):
        status = node.candidate.status if node.candidate else {}
        metadata = node.metadata
        timestamp = (
            status.get("updated_at")
            or metadata.get("trained_at")
            or metadata.get("created_at")
            or ""
        )
        return (_parse_timestamp(timestamp), node.candidate_id)

    def trained_nodes(self) -> List[SearchNode]:
        return [node for node in self.nodes.values() if node is not self.root and node.is_trained]

    def pending_nodes(self) -> List[SearchNode]:
        return [
            node for node in self.nodes.values()
            if node is not self.root and node.candidate and node.candidate.is_pending
        ]

    def elite_nodes(self, limit: int) -> List[SearchNode]:
        nodes = self.trained_nodes()
        return sorted(nodes, key=lambda node: node.q_leaf_value, reverse=True)[:limit]

    def elite_set_nodes(self, elite_ids: List[str], limit: Optional[int] = None) -> List[SearchNode]:
        nodes = [
            self.nodes[candidate_id]
            for candidate_id in elite_ids
            if candidate_id in self.nodes and self.nodes[candidate_id].is_trained
        ]
        nodes = sorted(nodes, key=lambda node: node.q_leaf_value, reverse=True)
        return nodes[:limit] if limit is not None else nodes

    def best_node(self) -> Optional[SearchNode]:
        elites = self.elite_nodes(1)
        return elites[0] if elites else None

    def path_to_root(self, node: SearchNode) -> List[SearchNode]:
        path = []
        cur = node
        while cur and cur is not self.root:
            path.append(cur)
            cur = self.nodes.get(cur.parent_id or "root")
        return list(reversed(path))

    def branch_excluding(self, node: SearchNode, randomize: bool = False, rng=None) -> List[SearchNode]:
        root_child = node
        while root_child.parent_id and root_child.parent_id != "root":
            root_child = self.nodes.get(root_child.parent_id, root_child)
        excluded = root_child.candidate_id
        result = []
        for child in self.root.children:
            if child.candidate_id == excluded:
                continue
            branch_nodes = self._collect_trained(child)
            if randomize and branch_nodes:
                max_depth = max(item.depth for item in branch_nodes)
                target_depth = rng.randint(child.depth, max_depth) if rng else child.depth
                depth_nodes = [item for item in branch_nodes if item.depth <= target_depth]
                result.append(max(depth_nodes, key=lambda item: item.q_leaf_value))
            else:
                result.extend(branch_nodes)
        if randomize and rng:
            rng.shuffle(result)
            return result
        return sorted(result, key=lambda item: item.q_leaf_value, reverse=True)

    def _collect_trained(self, node: SearchNode) -> List[SearchNode]:
        result = [node] if node.is_trained else []
        for child in node.children:
            result.extend(self._collect_trained(child))
        return result

    def min_max_q(self):
        return self.min_q, self.max_q

    def sibling_min_max_q(self, node: SearchNode):
        parent = self.nodes.get(node.parent_id or "root", self.root)
        sibling_q_values = [
            child.q_value
            for child in parent.children
            if child.is_trained or child.visits > 0
        ]
        if not sibling_q_values:
            return self.min_max_q()
        return min(sibling_q_values), max(sibling_q_values)

    def uct_score(self, node: SearchNode, c_param: float) -> float:
        q_min, q_max = self.sibling_min_max_q(node)
        eps = 1e-8
        q_norm = (node.q_value - q_min) / (q_max - q_min + eps)
        parent = self.nodes.get(node.parent_id or "root", self.root)
        visit_part = math.sqrt(2.0 * math.log(parent.visits + 1.0) / max(node.visits, eps))
        verify_part = self._self_verify_softmax(node, parent)
        return q_norm + c_param * visit_part + c_param * verify_part

    def selectable_children(self, node: SearchNode) -> List[SearchNode]:
        return [child for child in node.children if child.is_trained]

    def _self_verify_softmax(self, node: SearchNode, parent: SearchNode) -> float:
        siblings = [child for child in parent.children if child.is_trained]
        if not siblings:
            return 0.0
        scores = [child.self_verify_score for child in siblings]
        max_score = max(scores)
        exp_scores = [math.exp(score - max_score) for score in scores]
        total = sum(exp_scores)
        if total <= 0:
            return 0.0
        for sibling, exp_score in zip(siblings, exp_scores):
            if sibling is node:
                return exp_score / total
        return 0.0


def _parse_timestamp(value: str) -> datetime:
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return datetime.min

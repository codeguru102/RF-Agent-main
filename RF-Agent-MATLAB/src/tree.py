from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from candidate_store import Candidate
from matlab_log_reader import MatlabLogReader


@dataclass
class SearchNode:
    candidate: Optional[Candidate]
    candidate_id: str
    parent_id: Optional[str]
    children: List["SearchNode"] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    q_value: float = 0.0
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
    def is_trained(self):
        return self.candidate.is_trained if self.candidate else False


class SearchTree:
    def __init__(self, candidates: List[Candidate], log_reader: MatlabLogReader, dummy_failure: float):
        self.log_reader = log_reader
        self.dummy_failure = dummy_failure
        self.root = SearchNode(candidate=None, candidate_id="root", parent_id=None, reward_cur=dummy_failure)
        self.nodes: Dict[str, SearchNode] = {"root": self.root}
        self._build(candidates)
        self._recompute_statistics()

    def _build(self, candidates: List[Candidate]):
        for candidate in candidates:
            score = self.log_reader.score_summary(candidate.summary)
            node = SearchNode(
                candidate=candidate,
                candidate_id=candidate.candidate_id,
                parent_id=candidate.parent_id,
                reward_cur=score,
                q_value=score,
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

    def _recompute_statistics(self):
        for node in self.nodes.values():
            node.visits = 0
            node.total_reward = 0.0
            node.q_value = node.reward_cur

        for node in self.nodes.values():
            if node is self.root or not node.is_trained:
                continue
            reward = node.reward_cur
            cur = node
            while cur is not None:
                cur.visits += 1
                cur.total_reward += reward
                parent_id = cur.parent_id or "root"
                cur = self.nodes.get(parent_id) if cur.candidate_id != "root" else None

        self._update_q_values(self.root)

    def _update_q_values(self, node: SearchNode) -> float:
        child_values = [self._update_q_values(child) for child in node.children]
        if node.is_trained:
            own_value = node.reward_cur
        elif node.visits > 0:
            own_value = node.total_reward / node.visits
        else:
            own_value = self.dummy_failure
        if child_values:
            node.q_value = max([own_value] + child_values)
        else:
            node.q_value = own_value
        return node.q_value

    def trained_nodes(self) -> List[SearchNode]:
        return [node for node in self.nodes.values() if node is not self.root and node.is_trained]

    def pending_nodes(self) -> List[SearchNode]:
        return [
            node for node in self.nodes.values()
            if node is not self.root and node.candidate and node.candidate.is_pending
        ]

    def elite_nodes(self, limit: int) -> List[SearchNode]:
        nodes = self.trained_nodes()
        return sorted(nodes, key=lambda node: node.reward_cur, reverse=True)[:limit]

    def path_to_root(self, node: SearchNode) -> List[SearchNode]:
        path = []
        cur = node
        while cur and cur is not self.root:
            path.append(cur)
            cur = self.nodes.get(cur.parent_id or "root")
        return list(reversed(path))

    def branch_excluding(self, node: SearchNode) -> List[SearchNode]:
        root_child = node
        while root_child.parent_id and root_child.parent_id != "root":
            root_child = self.nodes.get(root_child.parent_id, root_child)
        excluded = root_child.candidate_id
        result = []
        for child in self.root.children:
            if child.candidate_id == excluded:
                continue
            result.extend(self._collect_trained(child))
        return sorted(result, key=lambda item: item.reward_cur, reverse=True)

    def _collect_trained(self, node: SearchNode) -> List[SearchNode]:
        result = [node] if node.is_trained else []
        for child in node.children:
            result.extend(self._collect_trained(child))
        return result

    def min_max_q(self):
        trained = self.trained_nodes()
        if not trained:
            return 0.0, 1.0
        values = [node.q_value for node in trained]
        return min(values), max(values)

    def uct_score(self, node: SearchNode, c_param: float) -> float:
        q_min, q_max = self.min_max_q()
        eps = 1e-8
        q_norm = (node.q_value - q_min) / (q_max - q_min + eps)
        parent = self.nodes.get(node.parent_id or "root", self.root)
        visit_part = math.sqrt(2.0 * math.log(parent.visits + 1.0) / (node.visits + 1.0))
        return q_norm + c_param * visit_part


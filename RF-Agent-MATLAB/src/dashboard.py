from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from config import load_json, save_json
from candidate_store import CandidateStore
from matlab_log_reader import MatlabLogReader
from recommendations import best_trained_summary, training_recommendations
from task_loader import load_task_folder
from tree import SearchNode, SearchTree


NODE_WIDTH = 286
NODE_HEIGHT = 124
X_GAP = 54
Y_GAP = 76
MARGIN_X = 40
MARGIN_Y = 40
NODE_TEXT_LIMIT = 36

ACTION_LABELS = {
    "initialize": "init",
    "mutation": "mutation",
    "mutation_mechanism": "mut-mech",
    "mutation_param": "mut-param",
    "crossover_elite": "cross-elite",
    "tree_reasoning": "tree-reason",
    "different_thought": "diff-thought",
}


def render_dashboard_for_task(
    *,
    task_dir: Path,
    agent_config_path: Path,
    latest_decisions: Optional[List[dict]] = None,
    show_window: bool = True,
):
    task_config = load_task_folder(task_dir)
    agent_config = load_json(agent_config_path)
    store = CandidateStore(Path(task_dir) / "candidates")
    log_reader = MatlabLogReader(
        task_config.get("score", {}),
        dummy_failure=float(agent_config.get("dummy_failure", -10000.0)),
    )
    tree = SearchTree(
        store.scan(),
        log_reader,
        float(agent_config.get("dummy_failure", -10000.0)),
        max_simulations=int(agent_config.get("simulations", 80)),
    )
    elite_ids = store.load_elite_ids()
    return render_dashboard(
        tree=tree,
        task_name=task_config["task_name"],
        agent_config=agent_config,
        output_dir=Path(task_dir) / "visualization",
        latest_decisions=latest_decisions,
        elite_node_ids=elite_ids,
        show_window=show_window,
    )


def render_dashboard(
    *,
    tree: SearchTree,
    task_name: str,
    agent_config: dict,
    output_dir: Path,
    latest_decisions: Optional[List[dict]] = None,
    elite_node_ids: Optional[List[str]] = None,
    best_node_id: Optional[str] = None,
    show_window: bool = True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latest_decisions = latest_decisions or []
    selected_update_ids = {
        decision["parent_id"]
        for decision in latest_decisions
        if decision.get("parent_id")
    }
    latest_training_ids = {
        decision["candidate_id"]
        for decision in latest_decisions
        if decision.get("candidate_id")
    }
    elite_display_limit = int(agent_config.get("dashboard_elite_max", agent_config.get("elite_control_num", 4)))
    elite_display_nodes = tree.elite_set_nodes(elite_node_ids or [], elite_display_limit)
    elite_ids = {node.candidate_id for node in elite_display_nodes}
    pending_ids = {node.candidate_id for node in tree.pending_nodes()}
    best = tree.best_node()
    if best_node_id is None:
        best_node_id = best.candidate_id if best else None
    c_param = compute_current_c_param(tree, agent_config)

    nodes_json = [
        node_to_dict(node, tree, c_param, elite_ids, selected_update_ids, latest_training_ids, pending_ids, best_node_id)
        for node in sorted(tree.nodes.values(), key=lambda item: (item.depth, item.candidate_id))
    ]
    edges_json = [
        {"from": node.candidate_id, "to": child.candidate_id}
        for node in tree.nodes.values()
        for child in node.children
    ]
    json_path = output_dir / "tree.json"
    save_json(
        json_path,
        {
            "task_name": task_name,
            "c_param": c_param,
            "latest_decisions": latest_decisions,
            "elite_display_limit": elite_display_limit,
            "elite_candidates": [
                {"candidate_id": node.candidate_id, "score": node.reward_cur}
                for node in elite_display_nodes
            ],
            "best_trained": best_trained_summary(tree),
            "train_next": training_recommendations(tree, latest_decisions),
            "nodes": nodes_json,
            "edges": edges_json,
        },
    )

    figure_path = output_dir / "tree_dashboard.png"
    draw_matplotlib_dashboard(
        tree=tree,
        task_name=task_name,
        c_param=c_param,
        figure_path=figure_path,
        latest_decisions=latest_decisions,
        selected_update_ids=selected_update_ids,
        latest_training_ids=latest_training_ids,
        elite_ids=elite_ids,
        pending_ids=pending_ids,
        best_node_id=best_node_id,
        show_window=show_window,
    )
    return {"json": json_path, "png": figure_path}


def draw_matplotlib_dashboard(
    *,
    tree: SearchTree,
    task_name: str,
    c_param: float,
    figure_path: Path,
    latest_decisions: List[dict],
    selected_update_ids: Iterable[str],
    latest_training_ids: Iterable[str],
    elite_ids: Iterable[str],
    pending_ids: Iterable[str],
    best_node_id: Optional[str],
    show_window: bool,
):
    if not show_window or os.getenv("RF_AGENT_NO_DASHBOARD") == "1":
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    positions = layout_tree(tree.root)
    max_x = max((x + NODE_WIDTH for x, _ in positions.values()), default=NODE_WIDTH)
    max_y = max((y + NODE_HEIGHT for _, y in positions.values()), default=NODE_HEIGHT)

    figure_width = max(14, min(34, (max_x + 180) / 150))
    figure_height = max(8, min(24, (max_y + 140) / 145))
    fig = plt.figure(figsize=(figure_width, figure_height), constrained_layout=True)
    fig.canvas.manager.set_window_title(f"RF-Agent Dashboard - {task_name}")
    ax_tree = fig.add_subplot(1, 1, 1)

    ax_tree.set_title(f"Search Tree: {task_name}", fontsize=14, fontweight="bold")
    ax_tree.set_xlim(-40, max_x + 80)
    ax_tree.set_ylim(max_y + 120, -40)
    ax_tree.axis("off")

    for parent in tree.nodes.values():
        for child in parent.children:
            draw_edge(ax_tree, positions[parent.candidate_id], positions[child.candidate_id])

    for node in sorted(tree.nodes.values(), key=lambda item: (item.depth, item.candidate_id)):
        draw_node(
            ax_tree,
            node,
            tree,
            c_param,
            positions[node.candidate_id],
            selected_update_ids,
            latest_training_ids,
            elite_ids,
            pending_ids,
            best_node_id,
            FancyBboxPatch,
        )

    fig.savefig(figure_path, dpi=180)
    if show_window and os.getenv("RF_AGENT_NO_DASHBOARD") != "1":
        plt.show(block=True)
    else:
        plt.close(fig)


def compute_current_c_param(tree: SearchTree, agent_config: dict) -> float:
    max_simulations = int(agent_config.get("simulations", 80))
    progress = min(len(tree.trained_nodes()) / max(max_simulations, 1), 1.0)
    c_param_init = float(agent_config.get("c_param_init", 0.4))
    c_param_final = float(agent_config.get("c_param_final", 0.1))
    return (c_param_init - c_param_final) * (1.0 - progress) + c_param_final


def layout_tree(root: SearchNode):
    leaf_index = 0
    positions = {}

    def place(node: SearchNode) -> float:
        nonlocal leaf_index
        if not node.children:
            x_center = MARGIN_X + leaf_index * (NODE_WIDTH + X_GAP) + NODE_WIDTH / 2
            leaf_index += 1
        else:
            child_centers = [place(child) for child in node.children]
            x_center = (min(child_centers) + max(child_centers)) / 2
        y = MARGIN_Y + node.depth * (NODE_HEIGHT + Y_GAP)
        positions[node.candidate_id] = (int(x_center - NODE_WIDTH / 2), y)
        return x_center

    place(root)
    return positions


def node_to_dict(
    node: SearchNode,
    tree: SearchTree,
    c_param: float,
    elite_ids: Iterable[str],
    selected_update_ids: Iterable[str],
    latest_training_ids: Iterable[str],
    pending_ids: Iterable[str],
    best_node_id: Optional[str],
) -> dict:
    status = node_status(node)
    uct = node_uct_or_none(node, tree, c_param)
    return {
        "candidate_id": node.candidate_id,
        "parent_id": node.parent_id,
        "status": status,
        "action_type": node.action_type,
        "action_index": node.action_index,
        "depth": node.depth,
        "score": node.reward_cur,
        "q_value": node.q_value,
        "visits": node.visits,
        "total_reward": node.total_reward,
        "self_verify_score": node.self_verify_score,
        "uct_score": uct,
        "uct_note": "selectable" if uct is not None else "not selectable until trained",
        "is_elite": node.candidate_id in elite_ids,
        "selected_for_update": node.candidate_id in selected_update_ids,
        "selected_for_training": node.candidate_id in pending_ids,
        "new_training_candidate": node.candidate_id in latest_training_ids,
        "is_best": node.candidate_id == best_node_id,
    }


def node_uct_or_none(node: SearchNode, tree: SearchTree, c_param: float):
    if node.candidate_id == "root" or not node.is_trained:
        return None
    return tree.uct_score(node, c_param)


def draw_edge(ax, parent_pos: Tuple[int, int], child_pos: Tuple[int, int]):
    px, py = parent_pos
    cx, cy = child_pos
    x1 = px + NODE_WIDTH / 2
    y1 = py + NODE_HEIGHT
    x2 = cx + NODE_WIDTH / 2
    y2 = cy
    ax.plot([x1, x2], [y1, y2], color="#94a3b8", linewidth=1.4, zorder=1)


def draw_node(
    ax,
    node: SearchNode,
    tree: SearchTree,
    c_param: float,
    position: Tuple[int, int],
    selected_update_ids: Iterable[str],
    latest_training_ids: Iterable[str],
    elite_ids: Iterable[str],
    pending_ids: Iterable[str],
    best_node_id: Optional[str],
    box_cls,
):
    x, y = position
    status = node_status(node)
    fill = status_fill(status)
    edge = "#64748b"
    width = 1.2
    if node.candidate_id in selected_update_ids:
        edge = "#2563eb"
        width = 3.2
    if node.candidate_id == best_node_id:
        edge = "#111827"
        width = 3.4

    box = box_cls(
        (x, y),
        NODE_WIDTH,
        NODE_HEIGHT,
        boxstyle="round,pad=0.02,rounding_size=8",
        facecolor=fill,
        edgecolor=edge,
        linewidth=width,
        zorder=2,
    )
    ax.add_patch(box)

    uct = tree.uct_score(node, c_param) if node.candidate_id != "root" and node.is_trained else None
    score = "n/a" if node.candidate_id == "root" else f"{node.reward_cur:.3f}"
    q_value = "n/a" if node.candidate_id == "root" else f"{node.q_value:.3f}"
    uct_text = "n/a" if uct is None else f"{uct:.3f}"
    action = format_action(node.action_type, node.action_index)

    lines = [
        node.candidate_id,
        f"{status} | {action}",
        f"score: {score}  Q: {q_value}",
        f"UCT: {uct_text}  verify: {node.self_verify_score:.2f}",
        f"parent: {short_id(node.parent_id)}  visits: {node.visits}",
    ]
    ax.text(
        x + 12,
        y + 18,
        ellipsize(lines[0], NODE_TEXT_LIMIT),
        fontsize=9,
        fontweight="bold",
        va="top",
        zorder=3,
        clip_on=True,
    )
    for idx, line in enumerate(lines[1:], start=1):
        ax.text(
            x + 12,
            y + 18 + idx * 19,
            ellipsize(line, NODE_TEXT_LIMIT),
            fontsize=8,
            va="top",
            zorder=3,
            clip_on=True,
        )

    badges = []
    if node.candidate_id == best_node_id:
        badges.append(("BEST", "#111827", "#ffffff"))
    if node.candidate_id in elite_ids:
        badges.append(("ELITE", "#facc15", "#713f12"))
    if node.candidate_id in pending_ids:
        badges.append(("TRAIN", "#fde68a", "#78350f"))
    if node.candidate_id in latest_training_ids:
        badges.append(("NEW", "#ddd6fe", "#4c1d95"))
    for idx, (label, color, text_color) in enumerate(badges):
        by = y + 8 + idx * 21
        badge = box_cls(
            (x + NODE_WIDTH - 60, by),
            48,
            16,
            boxstyle="round,pad=0.02,rounding_size=4",
            facecolor=color,
            edgecolor="#94a3b8",
            linewidth=0.8,
            zorder=4,
        )
        ax.add_patch(badge)
        ax.text(x + NODE_WIDTH - 36, by + 8, label, fontsize=7, color=text_color, fontweight="bold", ha="center", va="center", zorder=5)


def node_status(node: SearchNode) -> str:
    if node.candidate_id == "root":
        return "root"
    if node.candidate is None:
        return "unknown"
    return node.candidate.status.get("status", node.candidate.metadata.get("status", "unknown"))


def status_fill(status: str) -> str:
    if status == "trained":
        return "#eefbf3"
    if status in {"pending", "running"}:
        return "#fff7dc"
    if status == "failed":
        return "#ffe7e7"
    if status == "root":
        return "#eaf6ff"
    return "#f8fafc"


def format_action(action_type, action_index) -> str:
    if not action_type:
        return "root"
    label = ACTION_LABELS.get(action_type, str(action_type).replace("_", " "))
    if action_index is None:
        return label
    return f"{label}[{action_index}]"


def ellipsize(value: str, limit: int) -> str:
    value = str(value)
    if len(value) <= limit:
        return value
    return value[: max(limit - 3, 1)] + "..."


def short_id(candidate_id) -> str:
    if not candidate_id:
        return "-"
    text = str(candidate_id)
    return "c" + text.rsplit("_", 1)[-1] if "_" in text else text

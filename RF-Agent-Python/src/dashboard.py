from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from config import load_json, save_json
from candidate_store import CandidateStore
from python_log_reader import PythonLogReader
from recommendations import best_trained_summary, training_recommendations
from task_loader import load_task_folder
from tree import SearchNode, SearchTree


NODE_WIDTH = 250
NODE_HEIGHT = 132
X_GAP = 46
Y_GAP = 84
MARGIN_X = 40
MARGIN_Y = 40


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
    log_reader = PythonLogReader(
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
        elite_display_nodes=elite_display_nodes,
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
    elite_display_nodes: List[SearchNode],
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

    fig = plt.figure(figsize=(17, 9), constrained_layout=True)
    fig.canvas.manager.set_window_title(f"RF-Agent Dashboard - {task_name}")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0])
    ax_tree = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])

    ax_tree.set_title(f"Search Tree: {task_name}", fontsize=14, fontweight="bold")
    ax_tree.set_xlim(-40, max_x + 80)
    ax_tree.set_ylim(max_y + 180, -40)
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

    draw_info_panel(ax_info, tree, task_name, c_param, latest_decisions, best_node_id, elite_display_nodes)
    fig.savefig(figure_path, dpi=160)
    if show_window and os.getenv("RF_AGENT_NO_DASHBOARD") != "1":
        plt.show(block=True)
    else:
        plt.close(fig)


def compute_current_c_param(tree: SearchTree, agent_config: dict) -> float:
    progress = len(tree.trained_nodes()) / max(len(tree.nodes), 1)
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
    edge = "#475569"
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
    action = node.action_type or "root"

    lines = [
        node.candidate_id,
        f"status: {status}",
        f"action: {action}",
        f"score: {score}  Q: {q_value}",
        f"UCT: {uct_text}  verify: {node.self_verify_score:.2f}",
        f"parent: {node.parent_id or '-'}",
    ]
    ax.text(x + 10, y + 18, lines[0], fontsize=9, fontweight="bold", va="top", zorder=3)
    for idx, line in enumerate(lines[1:], start=1):
        ax.text(x + 10, y + 18 + idx * 18, line, fontsize=8, va="top", zorder=3)

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


def draw_info_panel(
    ax,
    tree: SearchTree,
    task_name: str,
    c_param: float,
    latest_decisions: List[dict],
    best_node_id: Optional[str],
    elite_display_nodes: List[SearchNode],
):
    ax.axis("off")
    trained = tree.trained_nodes()
    pending = tree.pending_nodes()
    failed = [node for node in tree.nodes.values() if node.candidate and node.candidate.is_failed]
    best_node = tree.nodes.get(best_node_id) if best_node_id else tree.best_node()

    lines = [
        f"RF-Agent Dashboard",
        f"Task: {task_name}",
        f"Nodes: {len(tree.nodes) - 1}",
        f"Trained: {len(trained)}",
        f"Pending training: {len(pending)}",
        f"Failed: {len(failed)}",
        f"Current UCT c: {c_param:.4f}",
        "",
        "Latest Choices",
    ]
    if latest_decisions:
        for decision in latest_decisions:
            lines.extend(
                [
                    f"- new: {decision.get('candidate_id')}",
                    f"  update parent: {decision.get('parent_id') or 'root initialization'}",
                    f"  action: {decision.get('action_type')}[{decision.get('action_index')}]",
                    f"  parent UCT: {format_optional_float(decision.get('parent_uct_at_selection'))}",
                    f"  self verify: {format_optional_float(decision.get('self_verify_score'))}",
                ]
            )
    else:
        lines.append("- none")

    lines.extend(["", "Elite Set"])
    if elite_display_nodes:
        for node in elite_display_nodes:
            lines.append(f"- {node.candidate_id}: {node.reward_cur:.4f}")
    else:
        lines.append("- none")

    lines.extend(["", "Final Best Reward"])
    if best_node and best_node.candidate:
        lines.extend(
            [
                f"candidate: {best_node.candidate_id}",
                f"score: {best_node.reward_cur:.6f}",
                f"Q: {best_node.q_value:.6f}",
                f"visits: {best_node.visits}",
                "",
                "Reward code:",
            ]
        )
        code = best_node.candidate.reward_code
        for line in code.splitlines()[:28]:
            lines.extend(textwrap.wrap(line, width=54, replace_whitespace=False) or [""])
    else:
        lines.append("No trained candidate yet.")

    lines.extend(["", "Train Next"])
    recs = training_recommendations(tree, latest_decisions)
    if recs:
        for rec in recs[:8]:
            lines.append(f"{rec['rank']}. {rec['candidate_id']} -> {Path(rec['reward_file']).name}")
    else:
        lines.append("No pending candidates.")

    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=9, family="monospace", va="top")


def node_status(node: SearchNode) -> str:
    if node.candidate_id == "root":
        return "root"
    if node.candidate is None:
        return "unknown"
    return node.candidate.status.get("status", node.candidate.metadata.get("status", "unknown"))


def status_fill(status: str) -> str:
    if status == "trained":
        return "#dcfce7"
    if status in {"pending", "running"}:
        return "#fef3c7"
    if status == "failed":
        return "#fee2e2"
    if status == "root":
        return "#e0f2fe"
    return "#f8fafc"


def format_optional_float(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from config import load_json, save_json
from candidate_store import CandidateStore
from matlab_log_reader import MatlabLogReader
from recommendations import best_trained_summary, training_recommendations
from task_loader import load_task_folder
from tree import SearchNode, SearchTree


NODE_WIDTH = 248
NODE_HEIGHT = 108
X_GAP = 48
Y_GAP = 64
MARGIN_X = 48
MARGIN_Y = 48
NODE_TEXT_LIMIT = 34
DASHBOARD_DPI = 100
INTERACTIVE_VIEW_WIDTH = 1600
INTERACTIVE_VIEW_HEIGHT = 920
ZOOM_FACTOR = 1.22

THEME = {
    "page_bg": "#f8fafc",
    "canvas_bg": "#ffffff",
    "grid_dot": "#e2e8f0",
    "header_bg": "#0f172a",
    "header_text": "#f8fafc",
    "header_muted": "#94a3b8",
    "footer_bg": "#f1f5f9",
    "footer_text": "#64748b",
    "edge": "#94a3b8",
    "edge_highlight": "#3b82f6",
    "text_primary": "#0f172a",
    "text_secondary": "#475569",
    "text_muted": "#64748b",
    "status": {
        "trained": {"fill": "#ecfdf5", "accent": "#10b981", "label": "Trained"},
        "pending": {"fill": "#fffbeb", "accent": "#f59e0b", "label": "Pending"},
        "running": {"fill": "#fffbeb", "accent": "#f59e0b", "label": "Running"},
        "failed": {"fill": "#fef2f2", "accent": "#ef4444", "label": "Failed"},
        "root": {"fill": "#eff6ff", "accent": "#3b82f6", "label": "Root"},
        "unknown": {"fill": "#f8fafc", "accent": "#94a3b8", "label": "Unknown"},
    },
    "highlight": {
        "best": "#059669",
        "selected": "#2563eb",
        "elite": "#ca8a04",
    },
}


@dataclass(frozen=True)
class DashboardStyle:
    node_width: int
    node_height: int
    x_gap: int
    y_gap: int
    margin_x: int
    margin_y: int
    text_limit: int
    title_font: float
    body_font: float
    badge_font: float
    line_step: int
    pad_x: int
    compact: bool = False


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
        q_value_config=task_config.get("q_value_settings") or task_config.get("fitness_score_settings"),
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
                {"candidate_id": node.candidate_id, "score": node.q_leaf_value}
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
        show_window=False,
    )

    if show_window and os.getenv("RF_AGENT_NO_DASHBOARD") != "1":
        launch_interactive_dashboard(json_path)

    return {"json": json_path, "png": figure_path}


def launch_interactive_dashboard(tree_json_path: Path):
    """Open the clean PyQt dashboard window for the saved tree.json."""
    try:
        from dashboard_qt import show_dashboard_window
    except Exception as exc:  # pragma: no cover - optional GUI dependency
        print(f"[dashboard] interactive window unavailable ({exc}); PNG was still written.")
        return
    show_dashboard_window(tree_json_path, block=True)


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
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    style = dashboard_style(len(tree.nodes))
    positions = layout_tree(tree.root, style)
    bounds = dashboard_bounds(positions, style)
    stats = dashboard_stats(tree, c_param, best_node_id)

    export_fig = create_dashboard_figure(
        interactive=False,
        task_name=task_name,
        stats=stats,
        tree=tree,
        c_param=c_param,
        positions=positions,
        selected_update_ids=selected_update_ids,
        latest_training_ids=latest_training_ids,
        elite_ids=elite_ids,
        pending_ids=pending_ids,
        best_node_id=best_node_id,
        style=style,
        bounds=bounds,
    )
    export_fig.savefig(figure_path, dpi=DASHBOARD_DPI, facecolor=THEME["page_bg"])
    plt.close(export_fig)


def create_dashboard_figure(
    *,
    interactive: bool,
    task_name: str,
    stats: dict,
    tree: SearchTree,
    c_param: float,
    positions: dict,
    selected_update_ids: Iterable[str],
    latest_training_ids: Iterable[str],
    elite_ids: Iterable[str],
    pending_ids: Iterable[str],
    best_node_id: Optional[str],
    style: DashboardStyle,
    bounds: Tuple[float, float, float, float],
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    if interactive:
        fig = plt.figure(figsize=(INTERACTIVE_VIEW_WIDTH / DASHBOARD_DPI, INTERACTIVE_VIEW_HEIGHT / DASHBOARD_DPI), dpi=DASHBOARD_DPI)
        tree_rect = [0.02, 0.10, 0.96, 0.78]
    else:
        x_min, x_max, y_min, y_max = bounds
        width = max(14, (x_max - x_min + 120) / DASHBOARD_DPI)
        height = max(9, (y_max - y_min + 220) / DASHBOARD_DPI)
        fig = plt.figure(figsize=(width, height), dpi=DASHBOARD_DPI)
        tree_rect = [0.03, 0.14, 0.94, 0.80]

    fig.patch.set_facecolor(THEME["page_bg"])
    if fig.canvas.manager:
        fig.canvas.manager.set_window_title(f"RF-Agent Dashboard — {task_name}")

    draw_header_panel(fig, task_name, stats)
    draw_legend_panel(fig, interactive=interactive)

    ax_tree = fig.add_axes(tree_rect)
    draw_dashboard_contents(
        ax_tree,
        tree,
        c_param,
        positions,
        selected_update_ids,
        latest_training_ids,
        elite_ids,
        pending_ids,
        best_node_id,
        FancyBboxPatch,
        style,
        bounds,
        fit_view=not interactive,
    )
    fig._rf_nav_bounds = bounds  # type: ignore[attr-defined]
    fig._rf_nav_ax = ax_tree  # type: ignore[attr-defined]
    return fig


def draw_header_panel(fig, task_name: str, stats: dict):
    from matplotlib.patches import FancyBboxPatch

    ax = fig.add_axes([0, 0.92, 1, 0.08])
    ax.set_facecolor(THEME["header_bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.025, 0.62, "RF-Agent Search Tree", fontsize=11, color=THEME["header_muted"], va="center")
    ax.text(0.025, 0.28, task_name, fontsize=16, color=THEME["header_text"], fontweight="bold", va="center")

    chips = [
        f"Nodes {stats['total']}",
        f"Trained {stats['trained']}",
        f"Pending {stats['pending']}",
        f"c = {stats['c_param']:.3f}",
    ]
    if stats["best_score"] is not None:
        chips.append(f"Best R {format_number(stats['best_score'])}")
    chip_x = 0.99
    for label in reversed(chips):
        width = 0.012 * len(label) + 0.018
        chip_x -= width + 0.012
        chip = FancyBboxPatch(
            (chip_x, 0.22),
            width,
            0.56,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="#1e293b",
            edgecolor="#334155",
            linewidth=0.8,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(chip)
        ax.text(chip_x + width / 2, 0.5, label, ha="center", va="center", fontsize=9, color="#e2e8f0", transform=ax.transAxes)


def draw_legend_panel(fig, *, interactive: bool = False):
    from matplotlib.patches import FancyBboxPatch

    width = 0.58 if interactive else 0.72
    ax = fig.add_axes([0.02, 0.035, width, 0.055])
    ax.set_facecolor(THEME["footer_bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    items = [
        ("Trained", THEME["status"]["trained"]["accent"]),
        ("Pending", THEME["status"]["pending"]["accent"]),
        ("Failed", THEME["status"]["failed"]["accent"]),
        ("Elite", THEME["highlight"]["elite"]),
        ("Selected", THEME["highlight"]["selected"]),
        ("Best", THEME["highlight"]["best"]),
    ]
    x = 0.01
    for label, color in items:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.35),
                0.018,
                0.30,
                boxstyle="round,pad=0,rounding_size=0.004",
                facecolor=color,
                edgecolor="none",
                transform=ax.transAxes,
            )
        )
        ax.text(x + 0.025, 0.5, label, fontsize=9, color=THEME["footer_text"], va="center", transform=ax.transAxes)
        x += 0.11

    if interactive:
        hint_ax = fig.add_axes([0.62, 0.035, 0.12, 0.055])
        hint_ax.set_facecolor(THEME["footer_bg"])
        hint_ax.axis("off")
        hint_ax.text(
            0.0,
            0.5,
            "Scroll/drag to navigate",
            ha="left",
            va="center",
            fontsize=8.5,
            color=THEME["footer_text"],
            transform=hint_ax.transAxes,
        )


def attach_interactive_navigation(fig, bounds: Tuple[float, float, float, float]):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    ax = fig._rf_nav_ax  # type: ignore[attr-defined]
    navigator = TreeNavigator(fig, ax, bounds)

    def make_button(label: str, left: float, callback):
        button_ax = fig.add_axes([left, 0.035, 0.045, 0.055])
        button_ax.set_facecolor("#ffffff")
        button = Button(button_ax, label, color="#ffffff", hovercolor="#e2e8f0")
        button.label.set_fontsize(11)
        button.label.set_color(THEME["text_primary"])
        button.on_clicked(callback)
        return button

    make_button("−", 0.745, lambda _event: navigator.zoom_out())
    make_button("+", 0.795, lambda _event: navigator.zoom_in())
    make_button("Fit", 0.845, lambda _event: navigator.fit_view())
    make_button("Reset", 0.905, lambda _event: navigator.reset_view())

    fig.canvas.mpl_connect("scroll_event", navigator.on_scroll)
    fig.canvas.mpl_connect("button_press_event", navigator.on_press)
    fig.canvas.mpl_connect("motion_notify_event", navigator.on_motion)
    fig.canvas.mpl_connect("button_release_event", navigator.on_release)
    fig.canvas.mpl_connect("key_press_event", navigator.on_key)
    navigator.fit_view()


class TreeNavigator:
    def __init__(self, fig, ax, bounds: Tuple[float, float, float, float]):
        self.fig = fig
        self.ax = ax
        self.bounds = bounds
        self._panning = False
        self._pan_start: Optional[Tuple[float, float, Tuple[float, float], Tuple[float, float]]] = None

    def fit_view(self, _event=None):
        x_min, x_max, y_min, y_max = self.bounds
        pad_x = max((x_max - x_min) * 0.04, 32)
        pad_y = max((y_max - y_min) * 0.06, 32)
        self.ax.set_xlim(x_min - pad_x, x_max + pad_x)
        self.ax.set_ylim(y_max + pad_y, y_min - pad_y)
        self.fig.canvas.draw_idle()

    def reset_view(self, _event=None):
        self.fit_view()

    def zoom_in(self, _event=None):
        self._zoom_at_center(ZOOM_FACTOR)

    def zoom_out(self, _event=None):
        self._zoom_at_center(1 / ZOOM_FACTOR)

    def on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        factor = ZOOM_FACTOR if event.button == "up" else 1 / ZOOM_FACTOR
        self._zoom_at(event.xdata, event.ydata, factor)

    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        self._panning = True
        self._pan_start = (event.xdata, event.ydata, self.ax.get_xlim(), self.ax.get_ylim())

    def on_motion(self, event):
        if not self._panning or self._pan_start is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x0, y0, xlim0, ylim0 = self._pan_start
        dx = event.xdata - x0
        dy = event.ydata - y0
        self.ax.set_xlim(xlim0[0] - dx, xlim0[1] - dx)
        self.ax.set_ylim(ylim0[0] - dy, ylim0[1] - dy)
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if event.button == 1:
            self._panning = False
            self._pan_start = None

    def on_key(self, event):
        if event.key in {"+", "=", "add"}:
            self.zoom_in()
        elif event.key in {"-", "subtract"}:
            self.zoom_out()
        elif event.key == "home":
            self.fit_view()
        elif event.key in {"r", "R"}:
            self.reset_view()

    def _zoom_at_center(self, factor: float):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        cx = (xlim[0] + xlim[1]) / 2
        cy = (ylim[0] + ylim[1]) / 2
        self._zoom_at(cx, cy, factor)

    def _zoom_at(self, cx: float, cy: float, factor: float):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        half_w = (xlim[1] - xlim[0]) / (2 * factor)
        half_h = (ylim[0] - ylim[1]) / (2 * factor)
        self.ax.set_xlim(cx - half_w, cx + half_w)
        self.ax.set_ylim(cy + half_h, cy - half_h)
        self.fig.canvas.draw_idle()


def compute_current_c_param(tree: SearchTree, agent_config: dict) -> float:
    max_simulations = int(agent_config.get("simulations", 80))
    progress = min(len(tree.trained_nodes()) / max(max_simulations, 1), 1.0)
    c_param_init = float(agent_config.get("c_param_init", 0.4))
    c_param_final = float(agent_config.get("c_param_final", 0.1))
    return (c_param_init - c_param_final) * (1.0 - progress) + c_param_final


def dashboard_stats(tree: SearchTree, c_param: float, best_node_id: Optional[str]) -> dict:
    best = tree.best_node()
    best_score = None
    if best and best.candidate_id != "root":
        best_score = best.q_leaf_value
    return {
        "total": len(tree.nodes),
        "trained": len(tree.trained_nodes()),
        "pending": len(tree.pending_nodes()),
        "c_param": c_param,
        "best_score": best_score,
        "best_node_id": best_node_id or (best.candidate_id if best else None),
    }


def dashboard_style(node_count: int) -> DashboardStyle:
    compact = node_count > 40
    return DashboardStyle(
        NODE_WIDTH,
        NODE_HEIGHT if not compact else 96,
        X_GAP if not compact else 36,
        Y_GAP if not compact else 52,
        MARGIN_X,
        MARGIN_Y,
        NODE_TEXT_LIMIT,
        8.5 if compact else 9.0,
        7.5 if compact else 8.0,
        6.5,
        17 if compact else 18,
        14,
        compact=compact,
    )


def dashboard_bounds(positions: dict, style: DashboardStyle) -> Tuple[float, float, float, float]:
    if not positions:
        return -style.margin_x, style.node_width + style.margin_x, -style.margin_y, style.node_height + style.margin_y
    left = min(x for x, _ in positions.values()) - style.margin_x
    right = max(x + style.node_width for x, _ in positions.values()) + style.margin_x
    top = min(y for _, y in positions.values()) - style.margin_y
    bottom = max(y + style.node_height for _, y in positions.values()) + style.margin_y
    return left, right, top, bottom


def draw_dashboard_contents(
    ax,
    tree: SearchTree,
    c_param: float,
    positions: dict,
    selected_update_ids: Iterable[str],
    latest_training_ids: Iterable[str],
    elite_ids: Iterable[str],
    pending_ids: Iterable[str],
    best_node_id: Optional[str],
    box_cls,
    style: DashboardStyle,
    bounds: Tuple[float, float, float, float],
    fit_view: bool = False,
):
    x_min, x_max, y_min, y_max = bounds
    ax.set_facecolor(THEME["canvas_bg"])
    if fit_view:
        pad_x = max((x_max - x_min) * 0.03, 24)
        pad_y = max((y_max - y_min) * 0.05, 24)
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_max + pad_y, y_min - pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    draw_grid_background(ax, bounds)

    for parent in tree.nodes.values():
        for child in parent.children:
            highlight = child.candidate_id in selected_update_ids or child.candidate_id == best_node_id
            draw_edge(ax, positions[parent.candidate_id], positions[child.candidate_id], style, highlight=highlight)

    for node in sorted(tree.nodes.values(), key=lambda item: (item.depth, item.candidate_id)):
        draw_node(
            ax,
            node,
            tree,
            c_param,
            positions[node.candidate_id],
            selected_update_ids,
            latest_training_ids,
            elite_ids,
            pending_ids,
            best_node_id,
            box_cls,
            style,
        )


def draw_grid_background(ax, bounds: Tuple[float, float, float, float]):
    x_min, x_max, y_min, y_max = bounds
    step = 40
    xs = range(int(x_min // step) * step, int(x_max // step + 1) * step, step)
    ys = range(int(y_min // step) * step, int(y_max // step + 1) * step, step)
    for x in xs:
        for y in ys:
            ax.plot(x, y, marker=".", color=THEME["grid_dot"], markersize=1.6, linestyle="None", zorder=0)


def layout_tree(root: SearchNode, style: DashboardStyle):
    leaf_index = 0
    positions = {}

    def place(node: SearchNode) -> float:
        nonlocal leaf_index
        if not node.children:
            x_center = style.margin_x + leaf_index * (style.node_width + style.x_gap) + style.node_width / 2
            leaf_index += 1
        else:
            child_centers = [place(child) for child in node.children]
            x_center = (min(child_centers) + max(child_centers)) / 2
        y = style.margin_y + node.depth * (style.node_height + style.y_gap)
        positions[node.candidate_id] = (int(x_center - style.node_width / 2), y)
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
        "score": node.q_leaf_value,
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


def draw_edge(ax, parent_pos: Tuple[int, int], child_pos: Tuple[int, int], style: DashboardStyle, *, highlight: bool = False):
    px, py = parent_pos
    cx, cy = child_pos
    x1 = px + style.node_width / 2
    y1 = py + style.node_height
    x2 = cx + style.node_width / 2
    y2 = cy
    mid_y = (y1 + y2) / 2
    xs = [x1, x1, x2, x2]
    ys = [y1, mid_y, mid_y, y2]
    ax.plot(
        xs,
        ys,
        color=THEME["edge_highlight"] if highlight else THEME["edge"],
        linewidth=2.0 if highlight else 1.3,
        solid_capstyle="round",
        zorder=1,
    )


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
    style: DashboardStyle,
):
    x, y = position
    status = node_status(node)
    palette = THEME["status"].get(status, THEME["status"]["unknown"])
    fill = palette["fill"]
    accent = palette["accent"]

    edge = "#cbd5e1"
    width = 1.0
    if node.candidate_id == best_node_id:
        edge = THEME["highlight"]["best"]
        width = 2.6
    elif node.candidate_id in selected_update_ids:
        edge = THEME["highlight"]["selected"]
        width = 2.4

    shadow = box_cls(
        (x + 2, y - 2),
        style.node_width,
        style.node_height,
        boxstyle="round,pad=0.02,rounding_size=10",
        facecolor="#e2e8f0",
        edgecolor="none",
        linewidth=0,
        zorder=1,
    )
    ax.add_patch(shadow)

    box = box_cls(
        (x, y),
        style.node_width,
        style.node_height,
        boxstyle="round,pad=0.02,rounding_size=10",
        facecolor=fill,
        edgecolor=edge,
        linewidth=width,
        zorder=2,
    )
    ax.add_patch(box)

    accent_bar = box_cls(
        (x + 3, y + 8),
        4,
        style.node_height - 16,
        boxstyle="round,pad=0,rounding_size=2",
        facecolor=accent,
        edgecolor="none",
        linewidth=0,
        zorder=3,
    )
    ax.add_patch(accent_bar)
    accent_bar.set_clip_path(box)

    uct = tree.uct_score(node, c_param) if node.candidate_id != "root" and node.is_trained else None
    score = "—" if node.candidate_id == "root" else format_number(node.q_leaf_value)
    q_value = "—" if node.candidate_id == "root" else format_number(node.q_value)
    uct_text = "—" if uct is None else format_number(uct)
    action = format_action(node.action_type, node.action_index)
    display_id = short_id(node.candidate_id)

    add_node_text(
        ax,
        box,
        x + style.pad_x + 6,
        y + 14,
        ellipsize(display_id, style.text_limit),
        fontsize=style.title_font,
        fontweight="bold",
        color=THEME["text_primary"],
    )
    add_node_text(
        ax,
        box,
        x + style.pad_x + 6,
        y + 14 + style.line_step,
        ellipsize(f"{palette['label']} · {action}", style.text_limit),
        fontsize=style.body_font,
        color=THEME["text_secondary"],
    )
    add_node_text(
        ax,
        box,
        x + style.pad_x + 6,
        y + 14 + style.line_step * 2,
        ellipsize(f"R {score}   Q {q_value}", style.text_limit),
        fontsize=style.body_font,
        color=THEME["text_primary"],
    )
    add_node_text(
        ax,
        box,
        x + style.pad_x + 6,
        y + 14 + style.line_step * 3,
        ellipsize(f"UCT {uct_text}   visits {node.visits}", style.text_limit),
        fontsize=style.body_font - 0.5,
        color=THEME["text_muted"],
    )
    if not style.compact and node.candidate_id != "root":
        add_node_text(
            ax,
            box,
            x + style.pad_x + 6,
            y + 14 + style.line_step * 4,
            ellipsize(f"verify {node.self_verify_score:.2f}", style.text_limit),
            fontsize=style.body_font - 0.5,
            color=THEME["text_muted"],
        )

    badges = []
    if node.candidate_id == best_node_id:
        badges.append(("BEST", THEME["highlight"]["best"], "#ffffff"))
    if node.candidate_id in elite_ids:
        badges.append(("ELITE", THEME["highlight"]["elite"], "#ffffff"))
    if node.candidate_id in pending_ids:
        badges.append(("TRAIN", "#fbbf24", "#78350f"))
    if node.candidate_id in latest_training_ids:
        badges.append(("NEW", "#a78bfa", "#ffffff"))

    badge_width = 44 if style.compact else 48
    badge_height = 14
    for idx, (label, color, text_color) in enumerate(badges[:3]):
        bx = x + style.node_width - badge_width - 8
        by = y + 6 + idx * (badge_height + 3)
        badge = box_cls(
            (bx, by),
            badge_width,
            badge_height,
            boxstyle="round,pad=0.02,rounding_size=4",
            facecolor=color,
            edgecolor="none",
            linewidth=0,
            zorder=4,
        )
        ax.add_patch(badge)
        badge.set_clip_path(box)
        add_node_text(
            ax,
            box,
            bx + badge_width / 2,
            by + badge_height / 2,
            label,
            fontsize=style.badge_font,
            color=text_color,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
        )


def node_status(node: SearchNode) -> str:
    if node.candidate_id == "root":
        return "root"
    if node.candidate is None:
        return "unknown"
    return node.candidate.status.get("status", node.candidate.metadata.get("status", "unknown"))


def add_node_text(
    ax,
    clip_box,
    x: float,
    y: float,
    text: str,
    *,
    fontsize: float,
    color: str = "#111827",
    fontweight: str = "normal",
    ha: str = "left",
    va: str = "top",
    zorder: int = 3,
):
    artist = ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        color=color,
        fontweight=fontweight,
        ha=ha,
        va=va,
        zorder=zorder,
        clip_on=True,
    )
    artist.set_clip_path(clip_box)
    return artist


def format_action(action_type, action_index) -> str:
    if not action_type:
        return "root"
    label = ACTION_LABELS.get(action_type, str(action_type).replace("_", " "))
    if action_index is None:
        return label
    return f"{label}[{action_index}]"


def format_number(value) -> str:
    if value == "n/a":
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    abs_number = abs(number)
    if abs_number >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    if abs_number >= 10_000:
        return f"{number / 1_000:.1f}k"
    if abs_number >= 100:
        return f"{number:.0f}"
    if abs_number >= 10:
        return f"{number:.1f}"
    return f"{number:.3f}"


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

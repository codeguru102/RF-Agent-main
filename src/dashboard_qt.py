"""Clean PyQt5 dashboard for the RF-Agent search tree.

Renders the tree.json produced by dashboard.render_dashboard inside an
interactive QGraphicsView with smooth wheel-zoom and drag-to-pan.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets


# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------
PALETTE = {
    "page": "#f1f5f9",
    "canvas": "#ffffff",
    "grid": "#e9eef5",
    "header": "#0f172a",
    "header_sub": "#94a3b8",
    "text": "#0f172a",
    "text_soft": "#475569",
    "text_muted": "#7c8aa0",
    "edge": "#cbd5e1",
    "edge_hot": "#3b82f6",
    "card_border": "#dbe3ee",
}

STATUS = {
    "trained": {"fill": "#ecfdf5", "accent": "#10b981", "label": "Trained"},
    "pending": {"fill": "#fffbeb", "accent": "#f59e0b", "label": "Pending"},
    "running": {"fill": "#fffbeb", "accent": "#f59e0b", "label": "Running"},
    "failed": {"fill": "#fef2f2", "accent": "#ef4444", "label": "Failed"},
    "root": {"fill": "#eff6ff", "accent": "#3b82f6", "label": "Root"},
    "unknown": {"fill": "#f8fafc", "accent": "#94a3b8", "label": "Unknown"},
}

HIGHLIGHT = {
    "best": "#059669",
    "selected": "#2563eb",
    "elite": "#ca8a04",
    "new": "#7c3aed",
    "multi": "#6366f1",
}

NODE_W = 232
NODE_H = 112
X_GAP = 46
Y_GAP = 78


# ----------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------
def _format_model_label(model: str) -> str:
    model = (model or "unknown").strip()
    lowered = model.lower()
    if lowered.startswith("gpt-"):
        return f"GPT{model[4:]}"
    return model


def _load_agent_model(package_root: Path) -> str:
    config_path = package_root / "configs" / "agent.json"
    if not config_path.exists():
        return "unknown"
    try:
        return json.loads(config_path.read_text(encoding="utf-8")).get("model", "unknown")
    except (OSError, json.JSONDecodeError, AttributeError):
        return "unknown"


def _parse_latest_progress(output: str) -> dict:
    for line in reversed(output.splitlines()):
        if not line.startswith("PROGRESS "):
            continue
        tokens = {}
        for part in line.split()[1:]:
            if "=" in part:
                key, value = part.split("=", 1)
                tokens[key] = value
        return tokens
    return {}


def _generation_progress_label(tokens: dict, fallback_model: str) -> str:
    from_id = tokens.get("from", "root")
    model = _format_model_label(tokens.get("model", fallback_model))
    step = int(tokens.get("step", "0"))
    total = int(tokens.get("total", "0"))
    action = tokens.get("action")
    cid = tokens.get("id")
    phase = tokens.get("phase")

    main = f"Generating using {model}"
    if phase == "generating" and action and total > 0:
        return f"{main}\n\nfrom {from_id} · {action}   ({step + 1} / {total})"
    if cid and action and total > 0:
        return f"{main}\n\nCreated {cid}   ({step} / {total} · {action})"
    if total > 0:
        return f"{main}\n\nfrom {from_id}   ({step} / {total})"
    return main


def _short_id(candidate_id: str) -> str:
    if not candidate_id:
        return "-"
    text = str(candidate_id)
    return "c" + text.rsplit("_", 1)[-1] if "_" in text else text


def _fmt(value) -> str:
    if value is None or value == "n/a":
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    a = abs(number)
    if a >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    if a >= 10_000:
        return f"{number / 1_000:.1f}k"
    if a >= 100:
        return f"{number:.0f}"
    if a >= 10:
        return f"{number:.1f}"
    return f"{number:.3f}"


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(value)


def _action(action_type, action_index) -> str:
    labels = {
        "initialize": "init",
        "mutation": "mutation",
        "mutation_mechanism": "mut-mech",
        "mutation_param": "mut-param",
        "crossover_elite": "cross-elite",
        "tree_reasoning": "tree-reason",
        "different_thought": "diff-thought",
    }
    if not action_type:
        return "root"
    label = labels.get(action_type, str(action_type).replace("_", " "))
    return label if action_index is None else f"{label}[{action_index}]"


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
def _layout(nodes: List[dict], edges: List[dict]) -> Dict[str, QtCore.QPointF]:
    children: Dict[str, List[str]] = {}
    for edge in edges:
        children.setdefault(edge["from"], []).append(edge["to"])

    depth = {node["candidate_id"]: int(node.get("depth", 0)) for node in nodes}
    known = set(depth)
    positions: Dict[str, QtCore.QPointF] = {}
    leaf = {"i": 0}

    def place(node_id: str) -> float:
        kids = [c for c in children.get(node_id, []) if c in known]
        if not kids:
            x = leaf["i"] * (NODE_W + X_GAP)
            leaf["i"] += 1
        else:
            centers = [place(child) for child in kids]
            x = (min(centers) + max(centers)) / 2
        y = depth.get(node_id, 0) * (NODE_H + Y_GAP)
        positions[node_id] = QtCore.QPointF(x, y)
        return x

    root_id = "root" if "root" in known else (nodes[0]["candidate_id"] if nodes else "root")
    if root_id in known:
        place(root_id)
    for node in nodes:  # any orphans
        if node["candidate_id"] not in positions:
            place(node["candidate_id"])
    return positions


# ----------------------------------------------------------------------------
# Disk persistence for node removal / re-parenting
# ----------------------------------------------------------------------------
def candidates_dir_for(tree_json_path) -> Optional[Path]:
    """Derive <task>/candidates from <task>/visualization/tree.json."""
    path = Path(tree_json_path).resolve()
    candidate = path.parent.parent / "candidates"
    return candidate if candidate.is_dir() else None


def _scan_candidate_folders(candidates_dir: Path) -> Dict[str, Path]:
    """Map candidate_id -> folder by reading each metadata.json."""
    mapping: Dict[str, Path] = {}
    for folder in sorted(Path(candidates_dir).glob("candidate_*")):
        meta_path = folder / "metadata.json"
        if not folder.is_dir() or not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        mapping[meta.get("candidate_id", folder.name)] = folder
    return mapping


def _read_candidate_status(folder: Path) -> dict:
    status_path = Path(folder) / "status.json"
    if not status_path.exists():
        return {}
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def remove_node_persist(candidates_dir: Path, node_id: str) -> Tuple[Optional[str], List[str]]:
    """Re-parent children to the removed node's parent and archive its folder.

    Returns (new_parent_id, child_ids). Edits the children's metadata.json on
    disk, moves the removed candidate folder into candidates/_removed/, and
    cleans up latest_generation.json references.
    """
    candidates_dir = Path(candidates_dir)
    folders = _scan_candidate_folders(candidates_dir)

    target_folder = folders.get(node_id)
    parent_id: Optional[str] = None
    if target_folder is not None:
        meta = json.loads((target_folder / "metadata.json").read_text(encoding="utf-8"))
        parent_id = meta.get("parent_id")

    child_ids: List[str] = []
    for cid, folder in folders.items():
        meta_path = folder / "metadata.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if meta.get("parent_id") == node_id:
            meta["parent_id"] = parent_id
            sources = meta.get("source_node_ids")
            if isinstance(sources, list):
                meta["source_node_ids"] = [parent_id if s == node_id else s for s in sources]
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            child_ids.append(cid)

    if target_folder is not None and target_folder.exists():
        trash = candidates_dir / "_removed"
        trash.mkdir(exist_ok=True)
        destination = trash / target_folder.name
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        shutil.move(str(target_folder), str(destination))

    _clean_reference_files(candidates_dir, node_id, parent_id)
    return parent_id, child_ids


def _result_logs_source(result_folder: Path) -> Path:
    logs_dir = Path(result_folder) / "logs"
    return logs_dir if logs_dir.is_dir() else Path(result_folder)


def scan_result_folders(root: Path) -> List[dict]:
    """Find immediate sub-folders that hold training log files."""
    results: List[dict] = []
    for folder in sorted(Path(root).iterdir()):
        if not folder.is_dir():
            continue
        source = _result_logs_source(folder)
        files = [p for p in source.rglob("*") if p.is_file()]
        if not files:
            continue
        csvs = [p for p in files if p.suffix.lower() == ".csv"]
        eval_files = [p for p in files if p.name.lower() == "eval.txt"]
        feedback_files = [p for p in files if p.name.lower() == "feedback.txt"]
        results.append({
            "name": folder.name,
            "path": folder,
            "file_count": len(files),
            "csv_count": len(csvs),
            "has_eval": bool(eval_files),
            "has_feedback": bool(feedback_files),
        })
    return results


def copy_result_into_candidate(result_folder: Path, candidate_folder: Path) -> List[str]:
    """Copy every file from a result logs folder into <candidate>/logs/."""
    logs_dir = Path(candidate_folder) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    source = _result_logs_source(result_folder)
    for item in sorted(source.rglob("*")):
        if not item.is_file():
            continue
        relative_path = item.relative_to(source)
        target_path = logs_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target_path)
        copied.append(str(relative_path))
    return copied


def _digits(text: str) -> Optional[int]:
    match = re.findall(r"\d+", str(text))
    return int(match[-1]) if match else None


def _clean_reference_files(candidates_dir: Path, node_id: str, parent_id: Optional[str]) -> None:
    latest_path = candidates_dir / "latest_generation.json"
    if latest_path.exists():
        try:
            data = json.loads(latest_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                kept = []
                for decision in data:
                    if not isinstance(decision, dict):
                        kept.append(decision)
                        continue
                    if decision.get("candidate_id") == node_id:
                        continue  # drop the removed candidate's own decision
                    for key in ("parent_id", "selected_tree_node_id"):
                        if decision.get(key) == node_id:
                            decision[key] = parent_id
                    kept.append(decision)
                latest_path.write_text(json.dumps(kept, indent=2), encoding="utf-8")
        except (ValueError, OSError):
            pass


# ----------------------------------------------------------------------------
# Graphics
# ----------------------------------------------------------------------------
class NodeCard(QtWidgets.QGraphicsItem):
    def __init__(self, node: dict):
        super().__init__()
        self.node = node
        self.on_remove = None  # callback(candidate_id) set by the window
        self.on_open_metrics = None  # callback(candidate_id) on double-click
        self.on_selection_click = None  # callback(candidate_id, ctrl_pressed)
        self.on_compare_selected = None  # callback() open comparison dialog
        self.on_toggle_good_to_train = None  # callback(candidate_id) from context menu
        self.setAcceptHoverEvents(True)
        self.setToolTip(self._tooltip())
        self._hover = False
        self._multi_selected = False

    def set_multi_selected(self, selected: bool):
        if self._multi_selected != selected:
            self._multi_selected = selected
            self.update()

    def contextMenuEvent(self, event):
        if self.node.get("candidate_id") == "root":
            return
        menu = QtWidgets.QMenu()
        toggle_action = None
        compare_action = None
        if callable(self.on_toggle_good_to_train):
            current = _as_bool(self.node.get("good_to_train"))
            label = "Mark good_to_train = False" if current else "Mark good_to_train = True"
            toggle_action = menu.addAction(label)
            menu.addSeparator()
        if callable(self.on_compare_selected):
            compare_action = menu.addAction("Compare selected nodes…")
        remove_action = menu.addAction("Remove node")
        chosen = menu.exec_(event.screenPos())
        if chosen == toggle_action and callable(self.on_toggle_good_to_train):
            self.on_toggle_good_to_train(self.node["candidate_id"])
        elif callable(self.on_compare_selected) and chosen == compare_action:
            self.on_compare_selected()
        elif chosen == remove_action and callable(self.on_remove):
            self.on_remove(self.node["candidate_id"])

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if (
            event.button() == QtCore.Qt.LeftButton
            and self.node.get("candidate_id") != "root"
            and callable(self.on_selection_click)
        ):
            ctrl = bool(event.modifiers() & QtCore.Qt.ControlModifier)
            self.on_selection_click(self.node["candidate_id"], ctrl)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent):
        if (
            event.button() == QtCore.Qt.LeftButton
            and self.node.get("candidate_id") != "root"
            and callable(self.on_open_metrics)
        ):
            self.on_open_metrics(self.node["candidate_id"])
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(-2, -2, NODE_W + 4, NODE_H + 6)

    def _tooltip(self) -> str:
        n = self.node
        rows = [
            f"<b>{n['candidate_id']}</b>",
            f"status: {n.get('status', 'unknown')}",
            f"good_to_train: {_as_bool(n.get('good_to_train'))}",
            f"action: {_action(n.get('action_type'), n.get('action_index'))}",
            f"reward: {_fmt(n.get('score'))}",
            f"q value: {_fmt(n.get('q_value'))}",
            f"uct: {_fmt(n.get('uct_score'))}",
            f"visits: {n.get('visits', 0)}",
            f"verify: {_fmt(n.get('self_verify_score'))}",
            f"depth: {n.get('depth', 0)}",
            "<i>Ctrl+click multi-select · double-click: metrics + reward code</i>",
        ]
        return "<br>".join(rows)

    def paint(self, painter: QtGui.QPainter, option, widget=None):
        n = self.node
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        status = n.get("status", "unknown")
        if n["candidate_id"] == "root":
            status = "root"
        palette = STATUS.get(status, STATUS["unknown"])

        rect = QtCore.QRectF(0, 0, NODE_W, NODE_H)

        # shadow
        shadow = QtCore.QRectF(2, 3, NODE_W, NODE_H)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(15, 23, 42, 26))
        painter.drawRoundedRect(shadow, 12, 12)

        # border highlight
        border = QtGui.QColor(PALETTE["card_border"])
        border_w = 1.2
        if self._multi_selected:
            border = QtGui.QColor(HIGHLIGHT["multi"])
            border_w = 2.8
        elif n.get("is_best"):
            # border = QtGui.QColor(HIGHLIGHT["best"])
            # border_w = 2.6
            pass
        elif n.get("selected_for_update"):
            border = QtGui.QColor(HIGHLIGHT["selected"])
            border_w = 2.4
        if self._hover and border_w < 2:
            border = QtGui.QColor(HIGHLIGHT["selected"])
            border_w = 2.0

        painter.setBrush(QtGui.QColor(palette["fill"]))
        pen = QtGui.QPen(border, border_w)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 12, 12)

        # accent bar
        accent_path = QtGui.QPainterPath()
        accent_path.addRoundedRect(QtCore.QRectF(8, 12, 4, NODE_H - 24), 2, 2)
        painter.fillPath(accent_path, QtGui.QColor(palette["accent"]))

        text_left = 22
        # title
        painter.setPen(QtGui.QColor(PALETTE["text"]))
        font = painter.font()
        font.setPointSizeF(11)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QtCore.QRectF(text_left, 10, NODE_W - text_left - 60, 20),
                         QtCore.Qt.AlignVCenter, _short_id(n["candidate_id"]))

        # subtitle
        font.setBold(False)
        font.setPointSizeF(8.5)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(PALETTE["text_soft"]))
        painter.drawText(QtCore.QRectF(text_left, 32, NODE_W - text_left - 12, 16),
                         QtCore.Qt.AlignVCenter,
                         f"{palette['label']} · {_action(n.get('action_type'), n.get('action_index'))}")

        # metrics
        painter.setPen(QtGui.QColor(PALETTE["text"]))
        font.setPointSizeF(9)
        painter.setFont(font)
        is_root = n["candidate_id"] == "root"
        r_txt = "—" if is_root else _fmt(n.get("score"))
        q_txt = "—" if is_root else _fmt(n.get("q_value"))
        painter.drawText(QtCore.QRectF(text_left, 52, NODE_W - text_left - 12, 16),
                         QtCore.Qt.AlignVCenter, f"R {r_txt}    Q {q_txt}")

        uct_txt = "—" if is_root else _fmt(n.get("uct_score"))
        self._draw_uct_line(painter, font, text_left, 72, uct_txt, n.get("visits", 0))
        if not is_root:
            painter.setPen(QtGui.QColor(PALETTE["text_muted"]))
            font.setPointSizeF(8.5)
            font.setBold(False)
            painter.setFont(font)
            train_flag = "T" if _as_bool(n.get("good_to_train")) else "F"
            painter.drawText(QtCore.QRectF(text_left, 90, NODE_W - text_left - 12, 16),
                             QtCore.Qt.AlignVCenter, f"verify {_fmt(n.get('self_verify_score'))}   train {train_flag}")

        # badges
        badges = []
        if n.get("is_best"):
            # badges.append(("BEST", HIGHLIGHT["best"]))
            pass
        if n.get("is_elite"):
            badges.append(("ELITE", HIGHLIGHT["elite"]))
        if n.get("selected_for_training"):
            badges.append(("TRAIN", "#d97706"))
        elif n.get("status") in ("pending", "running") and not _as_bool(n.get("good_to_train")):
            badges.append(("NO TRAIN", "#64748b"))
        if n.get("new_training_candidate"):
            badges.append(("NEW", HIGHLIGHT["new"]))
        bx = NODE_W - 12
        by = 10
        font.setPointSizeF(7)
        font.setBold(True)
        painter.setFont(font)
        for label, color in badges[:3]:
            w = 58 if len(label) > 5 else 44
            badge_rect = QtCore.QRectF(bx - w, by, w, 15)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(color))
            painter.drawRoundedRect(badge_rect, 4, 4)
            painter.setPen(QtGui.QColor("#ffffff"))
            painter.drawText(badge_rect, QtCore.Qt.AlignCenter, label)
            by += 18

    def _draw_uct_line(self, painter, font, left, top, uct_txt, visits):
        y_rect_h = 18
        # "UCT" label (muted)
        font.setPointSizeF(8.5)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(PALETTE["text_muted"]))
        label = "UCT "
        label_w = QtGui.QFontMetricsF(font).horizontalAdvance(label)
        painter.drawText(QtCore.QRectF(left, top, label_w, y_rect_h),
                         QtCore.Qt.AlignVCenter, label)

        # UCT value (bold, emphasized)
        font.setPointSizeF(11)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(HIGHLIGHT["selected"]))
        value_w = QtGui.QFontMetricsF(font).horizontalAdvance(uct_txt)
        painter.drawText(QtCore.QRectF(left + label_w, top - 1, value_w + 6, y_rect_h),
                         QtCore.Qt.AlignVCenter, uct_txt)

        # visits (muted)
        font.setPointSizeF(8.5)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(PALETTE["text_muted"]))
        painter.drawText(QtCore.QRectF(left + label_w + value_w + 12, top, NODE_W - left - label_w - value_w - 24, y_rect_h),
                         QtCore.Qt.AlignVCenter, f"visits {visits}")

    def hoverEnterEvent(self, event):
        self._hover = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hover = False
        self.update()
        super().hoverLeaveEvent(event)


class TreeView(QtWidgets.QGraphicsView):
    def __init__(self, scene: QtWidgets.QGraphicsScene):
        super().__init__(scene)
        self.on_clear_selection = None
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing |
                            QtGui.QPainter.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QtGui.QColor(PALETTE["canvas"]))
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.SmartViewportUpdate)
        self._zoom = 1.0
        self._min_zoom = 0.05
        self._max_zoom = 6.0

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        item = self.itemAt(event.pos())
        if (
            event.button() == QtCore.Qt.LeftButton
            and not isinstance(item, NodeCard)
            and not (event.modifiers() & QtCore.Qt.ControlModifier)
            and callable(self.on_clear_selection)
        ):
            self.on_clear_selection()
        super().mousePressEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self._apply_zoom(factor)

    def _apply_zoom(self, factor: float):
        target = self._zoom * factor
        if target < self._min_zoom or target > self._max_zoom:
            return
        self._zoom = target
        self.scale(factor, factor)

    def zoom_in(self):
        self._apply_zoom(1.25)

    def zoom_out(self):
        self._apply_zoom(1 / 1.25)

    def fit(self):
        rect = self.scene().itemsBoundingRect()
        if rect.isNull():
            return
        rect = rect.adjusted(-40, -40, 40, 40)
        self.fitInView(rect, QtCore.Qt.KeepAspectRatio)
        self._zoom = self.transform().m11()

    def reset(self):
        self.fit()

    def drawBackground(self, painter: QtGui.QPainter, rect: QtCore.QRectF):
        super().drawBackground(painter, rect)
        step = 32
        left = int(rect.left()) - (int(rect.left()) % step)
        top = int(rect.top()) - (int(rect.top()) % step)
        painter.setPen(QtGui.QPen(QtGui.QColor(PALETTE["grid"]), 0))
        x = left
        while x < rect.right():
            y = top
            while y < rect.bottom():
                painter.drawPoint(QtCore.QPointF(x, y))
                y += step
            x += step


class DashboardWindow(QtWidgets.QMainWindow):
    def __init__(self, data: dict, tree_json_path=None):
        super().__init__()
        self.data = data
        self.tree_json_path = Path(tree_json_path) if tree_json_path else None
        self.candidates_dir = candidates_dir_for(tree_json_path) if tree_json_path else None
        self._active_proc = None
        self._active_busy = None
        self._metrics_dialogs: List = []
        self._compare_dialogs: List = []
        self._selected_ids: set = set()
        self._node_cards: Dict[str, NodeCard] = {}
        task = data.get("task_name", "task")
        self.setWindowTitle(f"RF-Agent Dashboard — {task}")
        self.resize(1480, 900)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.main_layout = layout

        self.header_widget = self._build_header(data)
        layout.addWidget(self.header_widget)

        self.scene = QtWidgets.QGraphicsScene()
        self.scene.setBackgroundBrush(QtGui.QColor(PALETTE["canvas"]))
        self.view = TreeView(self.scene)
        self.view.on_clear_selection = self._clear_selection
        self._build_scene(data)
        layout.addWidget(self.view, 1)

        layout.addWidget(self._build_footer())
        self.setCentralWidget(central)
        self.setStyleSheet(f"QMainWindow {{ background: {PALETTE['page']}; }}")

        QtCore.QTimer.singleShot(0, self.view.fit)

    # -- header ----------------------------------------------------------------
    def _build_header(self, data: dict) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        bar.setFixedHeight(74)
        bar.setStyleSheet(f"background: {PALETTE['header']};")
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(22, 12, 18, 12)

        title_box = QtWidgets.QVBoxLayout()
        title_box.setSpacing(0)
        sub = QtWidgets.QLabel("RF-Agent Search Tree")
        sub.setStyleSheet(f"color: {PALETTE['header_sub']}; font-size: 11px;")
        title = QtWidgets.QLabel(data.get("task_name", "task"))
        title.setStyleSheet("color: #f8fafc; font-size: 19px; font-weight: 700;")
        title_box.addWidget(sub)
        title_box.addWidget(title)
        h.addLayout(title_box)
        h.addStretch(1)

        nodes = data.get("nodes", [])
        trained = sum(1 for n in nodes if n.get("status") == "trained")
        pending = sum(1 for n in nodes if n.get("status") in ("pending", "running"))
        good = sum(1 for n in nodes if n.get("status") in ("pending", "running") and _as_bool(n.get("good_to_train")))
        best = next((n for n in nodes if n.get("is_best")), None)
        chips = [("Nodes", str(len(nodes))), ("Trained", str(trained)),
                 ("Pending", str(pending)), ("Good", str(good)), ("c", f"{float(data.get('c_param', 0)):.3f}")]
        if best is not None:
            chips.append(("Best R", _fmt(best.get("score"))))
        for label, value in chips:
            h.addWidget(self._chip(label, value))

        h.addSpacing(12)
        generate_btn = QtWidgets.QPushButton("✦  Generate candidates")
        generate_btn.setCursor(QtCore.Qt.PointingHandCursor)
        generate_btn.setFixedHeight(32)
        generate_btn.setMinimumWidth(168)
        generate_btn.setStyleSheet(
            "QPushButton { color: #ffffff; background: #2563eb; border: 1px solid #1d4ed8;"
            "border-radius: 8px; font-size: 13px; font-weight: 600; padding: 0 14px; }"
            "QPushButton:hover { background: #3b82f6; }")
        generate_btn.clicked.connect(self._open_generate)
        h.addWidget(generate_btn)

        h.addSpacing(8)
        update_btn = QtWidgets.QPushButton("Update bad rewards")
        update_btn.setCursor(QtCore.Qt.PointingHandCursor)
        update_btn.setFixedHeight(32)
        update_btn.setMinimumWidth(154)
        update_btn.setStyleSheet(
            "QPushButton { color: #ffffff; background: #be123c; border: 1px solid #9f1239;"
            "border-radius: 8px; font-size: 13px; font-weight: 600; padding: 0 12px; }"
            "QPushButton:hover { background: #e11d48; }")
        update_btn.clicked.connect(self._open_update_bad)
        h.addWidget(update_btn)

        h.addSpacing(8)
        sync_btn = QtWidgets.QPushButton("⟳  Sync trained results")
        sync_btn.setCursor(QtCore.Qt.PointingHandCursor)
        sync_btn.setFixedHeight(32)
        sync_btn.setMinimumWidth(168)
        sync_btn.setStyleSheet(
            "QPushButton { color: #ffffff; background: #059669; border: 1px solid #047857;"
            "border-radius: 8px; font-size: 13px; font-weight: 600; padding: 0 14px; }"
            "QPushButton:hover { background: #10b981; }")
        sync_btn.clicked.connect(self._open_sync_dialog)
        h.addWidget(sync_btn)

        h.addSpacing(8)
        self._compare_btn = QtWidgets.QPushButton("Compare selected (0)")
        self._compare_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self._compare_btn.setFixedHeight(32)
        self._compare_btn.setMinimumWidth(168)
        self._compare_btn.setEnabled(False)
        self._compare_btn.setStyleSheet(
            "QPushButton { color: #ffffff; background: #6366f1; border: 1px solid #4f46e5;"
            "border-radius: 8px; font-size: 13px; font-weight: 600; padding: 0 14px; }"
            "QPushButton:hover { background: #818cf8; }"
            "QPushButton:disabled { background: #475569; border-color: #334155; color: #94a3b8; }")
        self._compare_btn.clicked.connect(self._open_compare_dialog)
        h.addWidget(self._compare_btn)
        self._sync_compare_button()

        h.addSpacing(12)
        for text, slot, primary in [("−", self._zoom_out, False), ("+", self._zoom_in, False),
                                     ("Fit", self._fit, True), ("Reset", self._reset, False)]:
            h.addWidget(self._button(text, slot, primary))
        return bar

    def _chip(self, label: str, value: str) -> QtWidgets.QWidget:
        chip = QtWidgets.QLabel(f"{label}  {value}")
        chip.setStyleSheet(
            "QLabel { color: #e2e8f0; background: #1e293b; border: 1px solid #334155;"
            "border-radius: 9px; padding: 5px 11px; font-size: 12px; }")
        return chip

    def _button(self, text: str, slot, primary: bool) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(text)
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.setFixedHeight(32)
        btn.setMinimumWidth(44)
        bg = "#2563eb" if primary else "#1e293b"
        hover = "#3b82f6" if primary else "#334155"
        btn.setStyleSheet(
            f"QPushButton {{ color: #f8fafc; background: {bg}; border: 1px solid #334155;"
            f"border-radius: 8px; font-size: 14px; padding: 0 12px; }}"
            f"QPushButton:hover {{ background: {hover}; }}")
        btn.clicked.connect(slot)
        return btn

    # -- footer / legend -------------------------------------------------------
    def _build_footer(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        bar.setFixedHeight(40)
        bar.setStyleSheet("background: #ffffff; border-top: 1px solid #e2e8f0;")
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(20, 0, 20, 0)
        legend = [("Trained", STATUS["trained"]["accent"]), ("Pending", STATUS["pending"]["accent"]),
                  ("Failed", STATUS["failed"]["accent"]), ("Elite", HIGHLIGHT["elite"]),
                  ("Selected", HIGHLIGHT["selected"]), ("Multi-select", HIGHLIGHT["multi"]),
                  ("Best", HIGHLIGHT["best"])]
        for label, color in legend:
            swatch = QtWidgets.QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(f"background: {color}; border-radius: 3px;")
            text = QtWidgets.QLabel(label)
            text.setStyleSheet(f"color: {PALETTE['text_soft']}; font-size: 12px;")
            h.addWidget(swatch)
            h.addWidget(text)
            h.addSpacing(14)
        h.addStretch(1)
        hint = QtWidgets.QLabel(
            "Ctrl+click to select multiple nodes · Compare selected · double-click for plots · "
            "right-click to toggle train status/remove")
        hint.setStyleSheet(f"color: {PALETTE['text_muted']}; font-size: 12px;")
        h.addWidget(hint)
        return bar

    # -- scene -----------------------------------------------------------------
    def _build_scene(self, data: dict):
        self._node_cards = {}
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        positions = _layout(nodes, edges)
        by_id = {n["candidate_id"]: n for n in nodes}

        for edge in edges:
            src = positions.get(edge["from"])
            dst = positions.get(edge["to"])
            if src is None or dst is None:
                continue
            child = by_id.get(edge["to"], {})
            hot = bool(child.get("is_best") or child.get("selected_for_update"))
            self._add_edge(src, dst, hot)

        for node in nodes:
            pos = positions.get(node["candidate_id"])
            if pos is None:
                continue
            card = NodeCard(node)
            cid = node["candidate_id"]
            card.on_remove = self.request_remove
            card.on_open_metrics = self._open_node_metrics
            card.on_selection_click = self._on_node_selection_click
            card.on_compare_selected = self._open_compare_dialog
            card.on_toggle_good_to_train = self._toggle_good_to_train
            card.set_multi_selected(cid in self._selected_ids)
            card.setPos(pos)
            card.setZValue(2)
            self.scene.addItem(card)
            if cid != "root":
                self._node_cards[cid] = card

        rect = self.scene.itemsBoundingRect().adjusted(-60, -60, 60, 60)
        self.scene.setSceneRect(rect)

    def _add_edge(self, src: QtCore.QPointF, dst: QtCore.QPointF, hot: bool):
        x1 = src.x() + NODE_W / 2
        y1 = src.y() + NODE_H
        x2 = dst.x() + NODE_W / 2
        y2 = dst.y()
        mid = (y1 + y2) / 2
        path = QtGui.QPainterPath(QtCore.QPointF(x1, y1))
        path.lineTo(x1, mid)
        path.lineTo(x2, mid)
        path.lineTo(x2, y2)
        color = QtGui.QColor(PALETTE["edge_hot"] if hot else PALETTE["edge"])
        pen = QtGui.QPen(color, 2.0 if hot else 1.4)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
        item = self.scene.addPath(path, pen)
        item.setZValue(1)

    # -- multi-select & comparison ---------------------------------------------
    def _on_node_selection_click(self, node_id: str, ctrl_pressed: bool):
        if node_id == "root":
            return
        if ctrl_pressed:
            if node_id in self._selected_ids:
                self._selected_ids.discard(node_id)
            else:
                self._selected_ids.add(node_id)
        else:
            if node_id in self._selected_ids and len(self._selected_ids) == 1:
                self._selected_ids.clear()
            else:
                self._selected_ids = {node_id}
        self._apply_selection_visuals()
        self._sync_compare_button()

    def _apply_selection_visuals(self):
        for cid, card in self._node_cards.items():
            card.set_multi_selected(cid in self._selected_ids)

    def _sync_compare_button(self):
        if not hasattr(self, "_compare_btn") or self._compare_btn is None:
            return
        count = len(self._selected_ids)
        self._compare_btn.setEnabled(count >= 1)
        self._compare_btn.setText(
            f"Compare selected ({count})" if count else "Compare selected (0)")

    def _clear_selection(self):
        self._selected_ids.clear()
        self._apply_selection_visuals()
        self._sync_compare_button()

    def _open_compare_dialog(self):
        if not self._selected_ids:
            QtWidgets.QMessageBox.information(
                self,
                "Compare nodes",
                "Select one or more nodes with Ctrl+click, then press Compare selected.",
            )
            return
        nodes = [
            n for n in self.data.get("nodes", [])
            if n.get("candidate_id") in self._selected_ids
        ]
        nodes.sort(key=lambda n: n.get("candidate_id", ""))
        folders = {}
        if self.candidates_dir:
            scanned = _scan_candidate_folders(self.candidates_dir)
            for cid in self._selected_ids:
                if cid in scanned:
                    folders[cid] = scanned[cid]
        from metrics_dialog import open_node_comparison

        self._compare_dialogs = [d for d in self._compare_dialogs if d.isVisible()]
        dialog = open_node_comparison(nodes, folders, parent=self)
        self._compare_dialogs.append(dialog)

    # -- manual review status reload ------------------------------------------
    def _load_statuses_from_disk(self):
        if not self.candidates_dir:
            QtWidgets.QMessageBox.warning(
                self,
                "Load unavailable",
                "Candidate folder not found for this task, so status.json files cannot be loaded.",
            )
            return
        folders = _scan_candidate_folders(self.candidates_dir)
        updated = 0
        for node in self.data.get("nodes", []):
            cid = node.get("candidate_id")
            folder = folders.get(cid)
            if folder is None:
                continue
            status = _read_candidate_status(folder)
            if not status:
                continue
            node["status"] = status.get("status", node.get("status", "unknown"))
            node["good_to_train"] = _as_bool(status.get("good_to_train", False))
            node["selected_for_training"] = (
                node.get("status") in ("pending", "running")
                and _as_bool(node.get("good_to_train"))
            )
            updated += 1
        self._persist_tree_json()
        self._refresh_view()
        QtWidgets.QMessageBox.information(
            self,
            "Statuses loaded",
            f"Loaded status.json for {updated} candidate node(s).",
        )

    def _toggle_good_to_train(self, node_id: str):
        if node_id == "root":
            return
        node = next(
            (n for n in self.data.get("nodes", []) if n.get("candidate_id") == node_id),
            None,
        )
        if node is None:
            return

        next_value = not _as_bool(node.get("good_to_train"))
        node["good_to_train"] = next_value
        node["selected_for_training"] = (
            node.get("status") in ("pending", "running") and next_value
        )

        if self.candidates_dir:
            folder = _scan_candidate_folders(self.candidates_dir).get(node_id)
            if folder is not None:
                status_path = folder / "status.json"
                status = _read_candidate_status(folder)
                if not status:
                    status = {
                        "status": node.get("status", "pending"),
                        "error_message": node.get("error_message", ""),
                    }
                status["status"] = status.get("status", node.get("status", "pending"))
                status["good_to_train"] = next_value
                status["updated_at"] = datetime.now(timezone.utc).isoformat()
                try:
                    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
                except OSError as exc:
                    node["good_to_train"] = not next_value
                    node["selected_for_training"] = (
                        node.get("status") in ("pending", "running")
                        and _as_bool(node.get("good_to_train"))
                    )
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Toggle failed",
                        f"Could not update {status_path}:\n{exc}",
                    )
                    return

        self._persist_tree_json()
        self._refresh_view()

    # -- node metrics (double-click) -------------------------------------------
    def _open_node_metrics(self, node_id: str):
        if node_id == "root":
            return
        node = next(
            (n for n in self.data.get("nodes", []) if n.get("candidate_id") == node_id),
            None,
        )
        if node is None:
            return
        if not self.candidates_dir:
            QtWidgets.QMessageBox.information(
                self,
                "Metrics unavailable",
                "Candidate folder not found for this task, so training logs cannot be loaded.",
            )
            return
        folder = _scan_candidate_folders(self.candidates_dir).get(node_id)
        if folder is None:
            QtWidgets.QMessageBox.information(
                self,
                "Metrics unavailable",
                f"No on-disk folder found for {node_id}.",
            )
            return
        from metrics_dialog import open_node_metrics

        package_root = Path(__file__).resolve().parent.parent
        task_dir = self.candidates_dir.parent if self.candidates_dir else None
        self._metrics_dialogs = [d for d in self._metrics_dialogs if d.isVisible()]
        dialog = open_node_metrics(
            node,
            folder,
            parent=self,
            task_dir=task_dir,
            package_root=package_root,
        )
        self._metrics_dialogs.append(dialog)

    # -- sync trained results --------------------------------------------------
    def _open_sync_dialog(self):
        if not self.candidates_dir:
            QtWidgets.QMessageBox.warning(
                self, "Sync unavailable",
                "Candidate folder not found for this view, so results cannot be "
                "synced to disk.")
            return

        root = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select the folder that contains the trained result folders")
        if not root:
            return

        results = scan_result_folders(Path(root))
        if not results:
            QtWidgets.QMessageBox.information(
                self, "No results found",
                "The selected folder has no sub-folders containing log files.\n\n"
                "Expected layout:\n  <selected folder>/\n    "
                "<result_1>/logs/  (any training output files)\n    <result_2>/logs/  ...")
            return

        pending = [n["candidate_id"] for n in self.data.get("nodes", [])
                   if n.get("status") in ("pending", "running") and _as_bool(n.get("good_to_train"))]
        if not pending:
            QtWidgets.QMessageBox.information(
                self, "Nothing to sync", "There are no pending nodes marked good_to_train=true.")
            return

        dialog = SyncMatchDialog(results, pending, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        mapping = dialog.mapping()  # candidate_id -> result_path
        if not mapping:
            return

        # Copy the matched result folders into each candidate's logs/ dir.
        folders = _scan_candidate_folders(self.candidates_dir)
        copied_report = []
        for candidate_id, result_path in mapping.items():
            folder = folders.get(candidate_id)
            if folder is None:
                copied_report.append(f"{candidate_id}: candidate folder not found, skipped")
                continue
            copied = copy_result_into_candidate(result_path, folder)
            copied_report.append(f"{candidate_id}: copied {len(copied)} file(s)")

        self._run_engine_action(
            action="sync",
            busy_text="Copying logs and syncing trained results…",
            title="Sync",
            extra_report=copied_report,
        )

    # -- generate candidates ---------------------------------------------------
    def _open_generate(self):
        if not self.candidates_dir:
            QtWidgets.QMessageBox.warning(
                self, "Generate unavailable",
                "Candidate folder not found for this view, so new candidates "
                "cannot be created.")
            return
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Generate candidates")
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setText(
            "Generate a new batch of reward candidates from the current tree?\n\n"
            "This calls the LLM and may take a little while. New pending nodes "
            "will appear when it finishes.")
        box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        box.setDefaultButton(QtWidgets.QMessageBox.Yes)
        if box.exec_() != QtWidgets.QMessageBox.Yes:
            return
        package_root = Path(__file__).resolve().parent.parent
        model_label = _format_model_label(_load_agent_model(package_root))
        self._run_engine_action(
            action="generate",
            busy_text=f"Generating using {model_label} (from agent.json)",
            title="Generate",
            extra_report=None,
        )

    def _open_update_bad(self):
        if not self.candidates_dir:
            QtWidgets.QMessageBox.warning(
                self, "Update unavailable",
                "Candidate folder not found for this view, so rewards cannot be updated.")
            return
        bad = [
            n["candidate_id"]
            for n in self.data.get("nodes", [])
            if n.get("status") in ("pending", "running") and not _as_bool(n.get("good_to_train"))
        ]
        if not bad:
            QtWidgets.QMessageBox.information(
                self, "Nothing to update", "No pending candidates are marked good_to_train=false.")
            return
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Update bad rewards")
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setText(
            "Regenerate reward functions in place for pending candidates marked "
            f"good_to_train=false?\n\nCandidates: {', '.join(bad)}\n\n"
            "This calls the LLM using each candidate's previous prompt plus manual feedback "
            "from logs/llm_feedback.md, logs/codex_analysis.txt, or logs/feedback.txt.")
        box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        box.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        if box.exec_() != QtWidgets.QMessageBox.Yes:
            return
        package_root = Path(__file__).resolve().parent.parent
        model_label = _format_model_label(_load_agent_model(package_root))
        self._run_engine_action(
            action="update-bad",
            busy_text=f"Updating bad rewards using {model_label}",
            title="Update bad rewards",
            extra_report=None,
        )

    # -- shared engine runner (async, non-blocking) ----------------------------
    def _run_engine_action(self, action, busy_text, title, extra_report=None):
        package_root = Path(__file__).resolve().parent.parent
        main_py = package_root / "src" / "main.py"
        if not main_py.exists():
            QtWidgets.QMessageBox.critical(
                self, f"{title} failed", "Could not locate the engine (src/main.py).")
            return
        if getattr(self, "_active_proc", None) is not None:
            QtWidgets.QMessageBox.information(
                self, "Please wait", "Another operation is still running.")
            return

        task_dir = self.candidates_dir.parent
        self._progress_fallback_model = _load_agent_model(package_root)
        proc = QtCore.QProcess(self)
        proc.setWorkingDirectory(str(package_root))
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("RF_AGENT_NO_DASHBOARD", "1")
        env.insert("PYTHONUNBUFFERED", "1")  # stream child stdout for live progress
        proc.setProcessEnvironment(env)

        busy = QtWidgets.QProgressDialog(busy_text, None, 0, 0, self)
        busy.setWindowTitle(title)
        busy.setWindowModality(QtCore.Qt.WindowModal)
        busy.setMinimumWidth(420)
        busy.setMinimumDuration(0)
        busy.setCancelButton(None)
        busy.setAutoClose(False)
        busy.setAutoReset(False)
        busy.show()

        self._active_proc = proc
        self._active_busy = busy
        self._proc_out_buffer = ""

        def on_ready():
            chunk = bytes(proc.readAllStandardOutput()).decode("utf-8", "replace")
            self._proc_out_buffer += chunk
            self._update_progress(
                self._proc_out_buffer,
                busy,
                busy_text,
                action=action,
                fallback_model=getattr(self, "_progress_fallback_model", "unknown"),
            )

        def on_finished(code, _status):
            self._proc_out_buffer += bytes(proc.readAllStandardOutput()).decode("utf-8", "replace")
            busy.close()
            out = self._proc_out_buffer
            err = bytes(proc.readAllStandardError()).decode("utf-8", "replace")
            self._active_proc = None
            self._active_busy = None
            self._progress_fallback_model = None
            if code != 0:
                QtWidgets.QMessageBox.critical(
                    self, f"{title} failed",
                    f"The {action} step returned an error:\n\n{(err or out)[-1800:]}")
                return
            self._reload_from_disk()
            self._show_action_result(title, action, out, extra_report)

        proc.readyReadStandardOutput.connect(on_ready)
        proc.finished.connect(on_finished)
        proc.start(sys.executable,
                   [str(main_py), "--internal-action", action, "--task-dir", str(task_dir)])

    @staticmethod
    def _update_progress(
        output: str,
        busy: "QtWidgets.QProgressDialog",
        busy_text: str,
        *,
        action: str = "",
        fallback_model: str = "unknown",
    ):
        tokens = _parse_latest_progress(output)
        if not tokens:
            return
        total = int(tokens.get("total", "0"))
        step = int(tokens.get("step", "0"))
        if total <= 0:
            return
        busy.setMaximum(total)
        busy.setValue(step if tokens.get("phase") != "generating" else max(step, 0))
        if action in {"generate", "update-bad"}:
            busy.setLabelText(_generation_progress_label(tokens, fallback_model))
            return
        cid = tokens.get("id")
        act = tokens.get("action")
        if cid:
            busy.setLabelText(f"{busy_text}\n\nGenerated {step} / {total}   ({cid} · {act})")
        else:
            busy.setLabelText(f"{busy_text}\n\nGenerated {step} / {total}")

    def _show_action_result(self, title, action, output, extra_report):
        if action == "sync":
            prefix = "Synced candidates"
        elif action == "update-bad":
            prefix = "Updated bad candidates"
        else:
            prefix = "Created pending candidates"
        summary_line = next((ln for ln in (output or "").splitlines()
                             if ln.startswith(prefix)), f"{title} complete.")
        message = summary_line
        if extra_report:
            message += "\n\n" + "\n".join(extra_report)
        QtWidgets.QMessageBox.information(self, f"{title} complete", message)

    def _reload_from_disk(self):
        if not self.tree_json_path or not self.tree_json_path.exists():
            return
        try:
            self.data = load_tree_data(self.tree_json_path)
        except (ValueError, OSError):
            return
        self._refresh_view()

    # -- node removal ----------------------------------------------------------
    def request_remove(self, node_id: str):
        if node_id == "root":
            return
        node = next((n for n in self.data.get("nodes", []) if n["candidate_id"] == node_id), None)
        if node is None:
            return
        parent_id = node.get("parent_id")
        children = [e["to"] for e in self.data.get("edges", []) if e["from"] == node_id]

        target = parent_id or "root"
        if children:
            detail = (f"Remove <b>{node_id}</b>?<br><br>"
                      f"Its {len(children)} child node(s) will be reconnected to "
                      f"<b>{target}</b>.")
        else:
            detail = f"Remove <b>{node_id}</b>? It has no children."

        persist_note = ("" if self.candidates_dir
                        else "<br><br><i>Note: candidate folder not found; this view "
                             "will update but the change won't be saved to disk.</i>")
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Remove node")
        box.setTextFormat(QtCore.Qt.RichText)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setText(detail + persist_note)
        box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        box.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        if box.exec_() != QtWidgets.QMessageBox.Yes:
            return

        if self.candidates_dir:
            try:
                disk_parent, disk_children = remove_node_persist(self.candidates_dir, node_id)
                parent_id = disk_parent
                if disk_children:
                    children = disk_children
            except Exception as exc:  # pragma: no cover - filesystem edge cases
                QtWidgets.QMessageBox.critical(
                    self, "Remove failed", f"Could not update files on disk:\n{exc}")
                return

        self._apply_removal_in_memory(node_id, parent_id, children)
        self._persist_tree_json()
        self._refresh_view()

    def _apply_removal_in_memory(self, node_id, parent_id, children):
        new_parent = parent_id or "root"
        child_set = set(children)
        self.data["nodes"] = [n for n in self.data.get("nodes", [])
                              if n["candidate_id"] != node_id]
        for n in self.data["nodes"]:
            if n["candidate_id"] in child_set:
                n["parent_id"] = parent_id
                n["depth"] = max(int(n.get("depth", 1)) - 1, 0)
        edges = [e for e in self.data.get("edges", [])
                 if e["from"] != node_id and e["to"] != node_id]
        for child in children:
            edges.append({"from": new_parent, "to": child})
        self.data["edges"] = edges
        if isinstance(self.data.get("elite_candidates"), list):
            self.data["elite_candidates"] = [
                c for c in self.data["elite_candidates"]
                if c.get("candidate_id") != node_id]

    def _persist_tree_json(self):
        if not self.tree_json_path:
            return
        try:
            self.tree_json_path.write_text(
                json.dumps(self.data, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _refresh_view(self):
        valid = {n.get("candidate_id") for n in self.data.get("nodes", [])}
        self._selected_ids &= valid
        transform = self.view.transform()
        self.scene.clear()
        self._build_scene(self.data)
        new_header = self._build_header(self.data)
        self.main_layout.replaceWidget(self.header_widget, new_header)
        self.header_widget.deleteLater()
        self.header_widget = new_header
        self._sync_compare_button()
        self.view.setTransform(transform)

    # -- slots -----------------------------------------------------------------
    def _zoom_in(self):
        self.view.zoom_in()

    def _zoom_out(self):
        self.view.zoom_out()

    def _fit(self):
        self.view.fit()

    def _reset(self):
        self.view.reset()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        key = event.key()
        if key == QtCore.Qt.Key_Escape:
            self._clear_selection()
        elif key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) and self._selected_ids:
            self._open_compare_dialog()
        elif key in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            self.view.zoom_in()
        elif key in (QtCore.Qt.Key_Minus, QtCore.Qt.Key_Underscore):
            self.view.zoom_out()
        elif key in (QtCore.Qt.Key_Home, QtCore.Qt.Key_R):
            self.view.fit()
        else:
            super().keyPressEvent(event)


class SyncMatchDialog(QtWidgets.QDialog):
    """Match each trained-result folder to a pending candidate node."""

    SKIP_LABEL = "— skip —"

    def __init__(self, results: List[dict], pending_ids: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Match trained results to pending nodes")
        self.resize(620, 460)
        self.results = results
        self.pending_ids = pending_ids
        self._combos: List[QtWidgets.QComboBox] = []

        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Assign each trained-result folder to the pending node it belongs to. "
            "All files from its logs folder will be copied into that node's logs folder, then synced.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #475569; font-size: 12px;")
        layout.addWidget(intro)

        table = QtWidgets.QTableWidget(len(results), 2)
        table.setHorizontalHeaderLabels(["Result folder", "Pending node"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        options = [self.SKIP_LABEL] + pending_ids
        for row, result in enumerate(results):
            extras = []
            if result["has_eval"]:
                extras.append("eval.txt")
            if result.get("has_feedback"):
                extras.append("feedback.txt")
            extra_text = ", " + ", ".join(extras) if extras else ""
            detail = f"{result['name']}   ({result['file_count']} files, {result['csv_count']} csv{extra_text})"
            item = QtWidgets.QTableWidgetItem(detail)
            item.setToolTip(str(result["path"]))
            table.setItem(row, 0, item)

            combo = QtWidgets.QComboBox()
            combo.addItems(options)
            suggested = self._suggest(result["name"], pending_ids)
            if suggested:
                combo.setCurrentText(suggested)
            combo.setMinimumWidth(180)
            self._combos.append(combo)
            table.setCellWidget(row, 1, combo)
        table.setRowHeight(0, 30)
        layout.addWidget(table, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel)
        self.sync_button = buttons.addButton("Sync", QtWidgets.QDialogButtonBox.AcceptRole)
        self.sync_button.setStyleSheet(
            "QPushButton { color: #ffffff; background: #059669; border: none;"
            "border-radius: 6px; padding: 6px 18px; font-weight: 600; }"
            "QPushButton:hover { background: #10b981; }")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _suggest(folder_name: str, pending_ids: List[str]) -> Optional[str]:
        target = _digits(folder_name)
        if target is None:
            return None
        for cid in pending_ids:
            if _digits(cid) == target:
                return cid
        return None

    def _on_accept(self):
        chosen = [c.currentText() for c in self._combos if c.currentText() != self.SKIP_LABEL]
        if not chosen:
            QtWidgets.QMessageBox.warning(
                self, "Nothing selected", "Assign at least one result to a pending node.")
            return
        duplicates = {cid for cid in chosen if chosen.count(cid) > 1}
        if duplicates:
            QtWidgets.QMessageBox.warning(
                self, "Duplicate assignment",
                "These nodes are assigned more than once:\n" + ", ".join(sorted(duplicates)))
            return
        self.accept()

    def mapping(self) -> Dict[str, Path]:
        result_map: Dict[str, Path] = {}
        for combo, result in zip(self._combos, self.results):
            cid = combo.currentText()
            if cid != self.SKIP_LABEL:
                result_map[cid] = result["path"]
        return result_map


def load_tree_data(tree_json_path) -> dict:
    return json.loads(Path(tree_json_path).read_text(encoding="utf-8"))


def show_dashboard_window(tree_json_path, block: bool = True) -> Optional[DashboardWindow]:
    data = load_tree_data(tree_json_path)
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication(sys.argv[:1])
    window = DashboardWindow(data, tree_json_path=tree_json_path)
    window.show()
    if owns_app and block:
        app.exec_()
        return None
    return window


def main():
    if len(sys.argv) < 2:
        print("usage: python dashboard_qt.py <path-to-tree.json>")
        raise SystemExit(2)
    show_dashboard_window(sys.argv[1], block=True)


if __name__ == "__main__":
    main()

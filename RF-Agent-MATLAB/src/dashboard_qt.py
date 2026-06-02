"""Clean PyQt5 dashboard for the RF-Agent search tree.

Renders the tree.json produced by dashboard.render_dashboard inside an
interactive QGraphicsView with smooth wheel-zoom and drag-to-pan.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

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

HIGHLIGHT = {"best": "#059669", "selected": "#2563eb", "elite": "#ca8a04", "new": "#7c3aed"}

NODE_W = 232
NODE_H = 112
X_GAP = 46
Y_GAP = 78


# ----------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------
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
# Graphics
# ----------------------------------------------------------------------------
class NodeCard(QtWidgets.QGraphicsItem):
    def __init__(self, node: dict):
        super().__init__()
        self.node = node
        self.setAcceptHoverEvents(True)
        self.setToolTip(self._tooltip())
        self._hover = False

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(-2, -2, NODE_W + 4, NODE_H + 6)

    def _tooltip(self) -> str:
        n = self.node
        rows = [
            f"<b>{n['candidate_id']}</b>",
            f"status: {n.get('status', 'unknown')}",
            f"action: {_action(n.get('action_type'), n.get('action_index'))}",
            f"reward: {_fmt(n.get('score'))}",
            f"q value: {_fmt(n.get('q_value'))}",
            f"uct: {_fmt(n.get('uct_score'))}",
            f"visits: {n.get('visits', 0)}",
            f"verify: {_fmt(n.get('self_verify_score'))}",
            f"depth: {n.get('depth', 0)}",
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
        if n.get("is_best"):
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
            painter.drawText(QtCore.QRectF(text_left, 90, NODE_W - text_left - 12, 16),
                             QtCore.Qt.AlignVCenter, f"verify {_fmt(n.get('self_verify_score'))}")

        # badges
        badges = []
        if n.get("is_best"):
            # badges.append(("BEST", HIGHLIGHT["best"]))
            pass
        if n.get("is_elite"):
            badges.append(("ELITE", HIGHLIGHT["elite"]))
        if n.get("selected_for_training"):
            badges.append(("TRAIN", "#d97706"))
        if n.get("new_training_candidate"):
            badges.append(("NEW", HIGHLIGHT["new"]))
        bx = NODE_W - 12
        by = 10
        font.setPointSizeF(7)
        font.setBold(True)
        painter.setFont(font)
        for label, color in badges[:3]:
            w = 44
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
    def __init__(self, data: dict):
        super().__init__()
        self.data = data
        task = data.get("task_name", "task")
        self.setWindowTitle(f"RF-Agent Dashboard — {task}")
        self.resize(1480, 900)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_header(data))

        self.scene = QtWidgets.QGraphicsScene()
        self.scene.setBackgroundBrush(QtGui.QColor(PALETTE["canvas"]))
        self.view = TreeView(self.scene)
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
        best = next((n for n in nodes if n.get("is_best")), None)
        chips = [("Nodes", str(len(nodes))), ("Trained", str(trained)),
                 ("Pending", str(pending)), ("c", f"{float(data.get('c_param', 0)):.3f}")]
        if best is not None:
            chips.append(("Best R", _fmt(best.get("score"))))
        for label, value in chips:
            h.addWidget(self._chip(label, value))

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
                  ("Selected", HIGHLIGHT["selected"]), ("Best", HIGHLIGHT["best"])]
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
        hint = QtWidgets.QLabel("Scroll to zoom · drag to pan · hover a card for details")
        hint.setStyleSheet(f"color: {PALETTE['text_muted']}; font-size: 12px;")
        h.addWidget(hint)
        return bar

    # -- scene -----------------------------------------------------------------
    def _build_scene(self, data: dict):
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
            card.setPos(pos)
            card.setZValue(2)
            self.scene.addItem(card)

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
        if key in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            self.view.zoom_in()
        elif key in (QtCore.Qt.Key_Minus, QtCore.Qt.Key_Underscore):
            self.view.zoom_out()
        elif key in (QtCore.Qt.Key_Home, QtCore.Qt.Key_R):
            self.view.fit()
        else:
            super().keyPressEvent(event)


def load_tree_data(tree_json_path) -> dict:
    return json.loads(Path(tree_json_path).read_text(encoding="utf-8"))


def show_dashboard_window(tree_json_path, block: bool = True) -> Optional[DashboardWindow]:
    data = load_tree_data(tree_json_path)
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication(sys.argv[:1])
    window = DashboardWindow(data)
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

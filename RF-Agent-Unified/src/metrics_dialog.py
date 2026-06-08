"""PyQt metrics dialog — reward performance plots on node double-click."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

# Match dashboard_qt palette
_THEME = {
    "bg": "#f8fafc",
    "card": "#ffffff",
    "text": "#0f172a",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "primary": "#3b82f6",
    "success": "#10b981",
    "warning": "#f59e0b",
    "accent": "#8b5cf6",
}

_PRIMARY_NAMES = (
    "reward",
    "rewards",
    "EpisodeReward",
    "task_score",
    "gt_reward",
    "total_reward",
    "return",
)

_COMPONENT_SKIP = frozenset(
    {"episode", "step", "time", "iteration", "index", "evaluationstatistic"}
)


class InteractiveChartTooltip:
    """Snap hover to nearest plotted point and show that point's metric values."""

    # Max normalized data-space distance (fraction of axis span) to snap.
    SNAP_THRESHOLD = 0.035
    PICK_PIXELS = 22

    def __init__(self, canvas, status_label: Optional[QtWidgets.QLabel] = None):
        self.canvas = canvas
        self.status_label = status_label
        self._series: List[dict] = []
        self._annots: Dict = {}
        self._cid = canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas._rf_hover_tooltip = self  # used by MetricsNavigationToolbar

    def clear(self):
        self._series.clear()

    def add_series(
        self,
        ax,
        xs,
        ys,
        series_name: str,
        x_label: str = "X",
        y_label: str = "Y",
    ):
        texts = []
        xs_out, ys_out = [], []
        for x, y in zip(xs, ys):
            if not (_is_finite(x) and _is_finite(y)):
                continue
            xs_out.append(float(x))
            ys_out.append(float(y))
            texts.append(
                f"{series_name}\n{x_label}: {_fmt_num(x)}\n{y_label}: {_fmt_num(y)}"
            )
        if xs_out:
            self._series.append(
                {"ax": ax, "xs": xs_out, "ys": ys_out, "texts": texts},
            )

    def add_point(self, ax, x: float, y: float, text: str):
        if not (_is_finite(x) and _is_finite(y)):
            return
        self._series.append(
            {"ax": ax, "xs": [float(x)], "ys": [float(y)], "texts": [text]},
        )

    def bind_axes(self):
        """Route matplotlib toolbar/status through data-point lookup, not raw mouse xy."""
        axes = {entry["ax"] for entry in self._series}
        for ax in axes:
            ax.format_coord = lambda x, y, axis=ax: self._format_coord_line(axis, x, y)

    def lookup(self, ax, x, y, event=None) -> Optional[str]:
        hit = self._nearest_hit(ax, x, y, event=event)
        return hit["text"] if hit else None

    def _format_coord_line(self, ax, x, y) -> str:
        text = self.lookup(ax, x, y)
        if not text:
            return "Hover closer to a point"
        return " · ".join(line.strip() for line in text.splitlines() if line.strip())

    def _nearest_hit(self, ax, x, y, event=None) -> Optional[dict]:
        if x is None or y is None:
            return None
        best = None
        best_score = float("inf")

        for entry in self._series:
            if entry["ax"] is not ax:
                continue
            xs, ys, texts = entry["xs"], entry["ys"], entry["texts"]
            if not xs:
                continue

            # Prefer pixel distance when we have the motion event (most reliable on Qt/HiDPI).
            if event is not None and event.x is not None and event.y is not None:
                for i, (px_x, px_y) in enumerate(
                    ax.transData.transform(list(zip(xs, ys)))
                ):
                    dist = ((event.x - px_x) ** 2 + (event.y - px_y) ** 2) ** 0.5
                    if dist < best_score:
                        best_score = dist
                        best = {"x": xs[i], "y": ys[i], "text": texts[i], "px_dist": dist}
                if best is not None and best.get("px_dist", 999) <= self.PICK_PIXELS:
                    return best
                best = None
                best_score = float("inf")

            # Fallback: normalized data-space distance.
            xspan = abs(ax.get_xlim()[1] - ax.get_xlim()[0]) or 1.0
            yspan = abs(ax.get_ylim()[1] - ax.get_ylim()[0]) or 1.0
            for i, (xd, yd) in enumerate(zip(xs, ys)):
                norm = ((xd - x) / xspan) ** 2 + ((yd - y) / yspan) ** 2
                if norm < best_score:
                    best_score = norm
                    best = {"x": xd, "y": yd, "text": texts[i], "norm": norm}

        if best is None:
            return None
        if "px_dist" not in best and best.get("norm", 1.0) > self.SNAP_THRESHOLD:
            return None
        return best

    def _get_annot(self, ax):
        if ax not in self._annots:
            self._annots[ax] = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(12, 12),
                textcoords="offset points",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    fc="#0f172a",
                    ec="#475569",
                    alpha=0.96,
                ),
                color="#f8fafc",
                fontsize=10,
                linespacing=1.4,
                zorder=100,
                arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=0.9),
            )
            self._annots[ax].set_visible(False)
        return self._annots[ax]

    def _set_status(self, text: str):
        if self.status_label is not None:
            self.status_label.setText(text)

    def _on_motion(self, event):
        if event.inaxes is None:
            for annot in self._annots.values():
                annot.set_visible(False)
            self._set_status("Hover over a line or point to see exact metric values")
            self.canvas.draw_idle()
            return

        ax = event.inaxes
        hit = self._nearest_hit(ax, event.xdata, event.ydata, event=event)

        for other_ax, annot in self._annots.items():
            if other_ax is not ax:
                annot.set_visible(False)

        annot = self._get_annot(ax)
        if hit is None:
            annot.set_visible(False)
            self._set_status(self._format_coord_line(ax, event.xdata, event.ydata))
        else:
            annot.xy = (hit["x"], hit["y"])
            annot.set_text(hit["text"])
            annot.set_visible(True)
            self._set_status(self._format_coord_line(ax, hit["x"], hit["y"]))
        self.canvas.draw_idle()


class MetricsNavigationToolbar:
    """Factory: toolbar that shows data metrics, not raw mouse coordinates."""

    @staticmethod
    def create(canvas, parent):
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

        tooltip = getattr(canvas, "_rf_hover_tooltip", None)

        class _Toolbar(NavigationToolbar2QT):
            def mouse_move(self, event):
                if tooltip is None or event.inaxes is None:
                    self.set_message("Hover a point for exact metric values")
                    return
                line = tooltip._format_coord_line(event.inaxes, event.xdata, event.ydata)
                self.set_message(line)

            def _update_cursor(self, event):
                super()._update_cursor(event)

        return _Toolbar(canvas, parent)


def _action_label(action_type, action_index) -> str:
    if not action_type:
        return "root"
    label = str(action_type).replace("_", " ")
    return label if action_index is None else f"{label}[{action_index}]"


def _fmt_num(value) -> str:
    if value is None:
        return "—"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x) or math.isinf(x):
        return "—"
    a = abs(x)
    if a >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    if a >= 10_000:
        return f"{x / 1_000:.1f}k"
    if a >= 100:
        return f"{x:.1f}"
    if a >= 10:
        return f"{x:.2f}"
    return f"{x:.3f}"


def _downsample(values: List[float], max_points: int = 2500) -> Tuple[List[int], List[float]]:
    n = len(values)
    if n <= max_points:
        return list(range(n)), values
    xs, ys = [], []
    step = n / max_points
    for i in range(max_points):
        idx = min(int(i * step), n - 1)
        xs.append(idx)
        ys.append(values[idx])
    if xs[-1] != n - 1:
        xs.append(n - 1)
        ys.append(values[-1])
    return xs, ys


def _read_single_column_csv(path: Path) -> Optional[List[float]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    if len(lines) < 2:
        return None
    values = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line.split(",")[0]))
        except ValueError:
            continue
    return values or None


def _read_dict_csv(path: Path) -> Dict[str, List[float]]:
    columns: Dict[str, List[float]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return {}
            for row in reader:
                for key, raw in row.items():
                    if key is None:
                        continue
                    try:
                        columns.setdefault(key, []).append(float(raw))
                    except (TypeError, ValueError):
                        continue
    except OSError:
        return {}
    return columns


def _pick_primary_metric(columns: Dict[str, List[float]]) -> Optional[str]:
    lower_map = {k.lower(): k for k in columns}
    for name in _PRIMARY_NAMES:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    for key, values in columns.items():
        if values and key.lower() not in _COMPONENT_SKIP:
            return key
    return None


def load_candidate_plot_data(folder: Path) -> dict:
    """Load time series and summary stats for plotting."""
    folder = Path(folder)
    summary_path = folder / "summary.json"
    summary: dict = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            summary = {}

    series: Dict[str, List[float]] = {}
    series_source = ""
    logs_dir = folder / "logs"
    if logs_dir.is_dir():
        for csv_path in sorted(logs_dir.glob("*.csv")):
            cols = _read_dict_csv(csv_path)
            if len(cols) >= 2:
                for key, values in cols.items():
                    if values:
                        series[key] = values
                if series:
                    series_source = csv_path.name
                    break
            single = _read_single_column_csv(csv_path)
            if single:
                header = csv_path.stem.replace("_", " ")
                if "reward" in csv_path.stem.lower():
                    header = "reward"
                series[header] = single
                series_source = csv_path.name
                break

    primary_key = _pick_primary_metric(series)
    if primary_key is None and isinstance(summary.get("metrics"), dict):
        metrics = summary["metrics"]
        for name in _PRIMARY_NAMES:
            if name in metrics and isinstance(metrics[name], dict):
                samples = metrics[name].get("samples") or []
                clean = [float(v) for v in samples if _is_finite(v)]
                if clean:
                    series[name] = clean
                    primary_key = name
                    series_source = "summary.json (samples)"
                    break
        if primary_key is None:
            for key, stats in metrics.items():
                if not isinstance(stats, dict):
                    continue
                samples = stats.get("samples") or []
                clean = [float(v) for v in samples if _is_finite(v)]
                if clean and key.lower() not in _COMPONENT_SKIP:
                    series[key] = clean
                    primary_key = key
                    series_source = "summary.json (samples)"
                    break

    components: Dict[str, dict] = {}
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    primary_lower = (primary_key or "").lower()
    for key, stats in metrics.items():
        if not isinstance(stats, dict):
            continue
        if key.lower() in _COMPONENT_SKIP or key.lower() == primary_lower:
            continue
        if key.lower() in ("episodereward",) and primary_lower in ("reward", "rewards"):
            continue
        max_v = stats.get("max")
        if not _is_finite(max_v) and not _is_finite(stats.get("last")):
            continue
        samples = stats.get("samples") or []
        clean = [float(v) for v in samples if _is_finite(v)]
        if clean:
            components[key] = {"samples": clean, "max": stats.get("max"), "final": stats.get("last")}

    eval_metrics = None
    eval_path = logs_dir / "eval.txt" if logs_dir.is_dir() else None
    if eval_path and eval_path.exists():
        import re

        lines = [ln.strip() for ln in eval_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines:
            nums = [
                float(x)
                for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", lines[-1])
            ]
            if nums:
                eval_metrics = nums

    primary_stats = None
    if primary_key and primary_key in metrics and isinstance(metrics[primary_key], dict):
        primary_stats = metrics[primary_key]
    elif primary_key:
        for prefix in ("max_", "final_", "mean_"):
            field = f"{prefix}{primary_key}"
            if field in summary:
                primary_stats = primary_stats or {}
                primary_stats[prefix.rstrip("_")] = summary[field]

    return {
        "summary": summary,
        "series": series,
        "primary_key": primary_key,
        "primary_stats": primary_stats,
        "components": components,
        "series_source": series_source,
        "eval_metrics": eval_metrics,
    }


def _is_finite(value) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x)


def _make_log_reader(task_config: dict, agent_config: dict):
    score = task_config.get("score", {})
    dummy = float(agent_config.get("dummy_failure", -10000.0))
    q_cfg = task_config.get("q_value_settings") or task_config.get("fitness_score_settings")
    from log_reader import OfflineLogReader

    return OfflineLogReader(score, dummy, q_value_config=q_cfg)


def load_q_value_history(
    task_dir: Path,
    package_root: Path,
    candidate_id: str,
) -> List[dict]:
    """Replay the search tree and return Q/visit snapshots for one node."""
    task_dir = Path(task_dir)
    package_root = Path(package_root)
    task_config_path = task_dir / "task.json"
    agent_config_path = package_root / "configs" / "agent.json"
    if not task_config_path.exists() or not agent_config_path.exists():
        return []
    try:
        from config import load_json
        from candidate_store import CandidateStore
        from task_loader import load_task_folder
        from tree import SearchTree
    except ImportError:
        return []

    try:
        task_config = load_task_folder(task_dir)
        agent_config = load_json(agent_config_path)
        store = CandidateStore(task_dir / "candidates")
        candidates = store.scan()
        log_reader = _make_log_reader(task_config, agent_config)
        tree = SearchTree(
            candidates,
            log_reader,
            float(agent_config.get("dummy_failure", -10000.0)),
            max_simulations=int(agent_config.get("simulations", 80)),
        )
        histories = tree.recompute_q_histories()
        return histories.get(candidate_id, [])
    except (ValueError, OSError, KeyError):
        return []


_CODE_THEME = {
    "bg": "#0f172a",
    "text": "#e2e8f0",
    "keyword": "#7dd3fc",
    "builtin": "#c4b5fd",
    "string": "#86efac",
    "comment": "#64748b",
    "number": "#fcd34d",
    "operator": "#f9a8d4",
}


def load_reward_source(candidate_folder: Optional[Path]) -> dict:
    """Load reward function source from a candidate folder."""
    if candidate_folder is None or not Path(candidate_folder).is_dir():
        return {"code": "", "path": "", "language": "python", "exists": False}

    folder = Path(candidate_folder)
    reward_file = "reward_fcn.py"
    language = "python"
    design_thought = ""

    meta_path = folder / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            reward_file = meta.get("reward_file", reward_file)
            language = str(meta.get("reward_language", language)).lower()
            design_thought = str(meta.get("design_thought", "") or "").strip()
        except (ValueError, OSError):
            pass

    path = folder / reward_file
    if not path.exists():
        for alt in ("reward_fcn.py", "reward_fcn.m"):
            alt_path = folder / alt
            if alt_path.exists():
                path = alt_path
                language = "matlab" if alt.endswith(".m") else "python"
                break

    if path.exists():
        try:
            code = path.read_text(encoding="utf-8")
        except OSError:
            code = ""
        return {
            "code": code,
            "path": str(path),
            "language": language,
            "exists": True,
            "design_thought": design_thought,
        }

    return {
        "code": f"# Reward file not found in\n# {folder}",
        "path": str(folder / reward_file),
        "language": language,
        "exists": False,
        "design_thought": design_thought,
    }


class CodeSyntaxHighlighter(QtGui.QSyntaxHighlighter):
    """Light Python / MATLAB syntax coloring for the reward code tab."""

    def __init__(self, document, language: str):
        super().__init__(document)
        self._rules: List[Tuple[re.Pattern, QtGui.QTextCharFormat]] = []
        self._build_rules(language)

    def _fmt(self, color: str, *, bold: bool = False, italic: bool = False) -> QtGui.QTextCharFormat:
        text_format = QtGui.QTextCharFormat()
        text_format.setForeground(QtGui.QColor(color))
        if bold:
            text_format.setFontWeight(QtGui.QFont.Bold)
        if italic:
            text_format.setFontItalic(True)
        return text_format

    def _add(self, pattern: str, fmt: QtGui.QTextCharFormat, flags: int = 0):
        self._rules.append((re.compile(pattern, flags), fmt))

    def _build_rules(self, language: str):
        lang = (language or "python").lower()
        if lang in ("matlab", "m"):
            keywords = (
                r"\b(function|end|if|elseif|else|for|while|return|switch|case|"
                r"otherwise|break|continue|persistent|global|nargin|nargout|"
                r"true|false)\b"
            )
            builtins = (
                r"\b(abs|min|max|sum|mean|exp|log|sqrt|tanh|sin|cos|isfinite|"
                r"rad2deg|deg2rad)\b"
            )
            comment = (r"%[^\n]*", _CODE_THEME["comment"])
            string_pat = r"'[^'\n]*'"
        else:
            keywords = (
                r"\b(def|class|return|import|from|as|if|elif|else|for|while|"
                r"try|except|finally|with|pass|break|continue|raise|lambda|"
                r"True|False|None|and|or|not|in|is)\b"
            )
            builtins = r"\b(self|len|range|float|int|str|list|dict|max|min|abs|sum)\b"
            comment = (r"#[^\n]*", _CODE_THEME["comment"])
            string_pat = r"(\"\"\"[\s\S]*?\"\"\"|\"[^\"\\n]*\"|'[^'\\n]*')"

        self._add(keywords, self._fmt(_CODE_THEME["keyword"], bold=True))
        self._add(builtins, self._fmt(_CODE_THEME["builtin"]))
        self._add(string_pat, self._fmt(_CODE_THEME["string"]))
        self._add(comment[0], self._fmt(comment[1], italic=True))
        self._add(
            r"\b\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\b",
            self._fmt(_CODE_THEME["number"]),
        )
        self._add(r"[+\-*/=<>!&|^~]+", self._fmt(_CODE_THEME["operator"]))

    def highlightBlock(self, text: str):
        for pattern, fmt in self._rules:
            for match in pattern.finditer(text):
                self.setFormat(match.start(), match.end() - match.start(), fmt)


class NodeMetricsDialog(QtWidgets.QDialog):
    """Tabbed node inspector: metrics plots and reward function source."""

    def __init__(
        self,
        node: dict,
        candidate_folder: Optional[Path],
        parent=None,
        *,
        q_history: Optional[List[dict]] = None,
    ):
        super().__init__(parent)
        self.node = node
        self._candidate_folder = (
            Path(candidate_folder) if candidate_folder and Path(candidate_folder).is_dir() else None
        )
        cid = node.get("candidate_id", "node")
        self.setWindowTitle(f"Node details — {cid}")
        self.resize(960, 780)
        self.setMinimumSize(760, 560)
        self._q_history = q_history or []
        self._reward_source = load_reward_source(self._candidate_folder)
        self._plot_data = (
            load_candidate_plot_data(self._candidate_folder)
            if self._candidate_folder
            else {}
        )
        self._build_ui()
        self._draw_charts()

    def _build_ui(self):
        self.setStyleSheet(
            f"QDialog {{ background: {_THEME['bg']}; }}"
            f"QLabel#title {{ color: {_THEME['text']}; font-size: 18px; font-weight: 700; }}"
            f"QLabel#sub {{ color: {_THEME['muted']}; font-size: 12px; }}"
            f"QLabel#chip {{ background: {_THEME['card']}; color: {_THEME['text']}; "
            f"border: 1px solid {_THEME['grid']}; border-radius: 8px; padding: 8px 12px; font-size: 12px; }}"
            f"QTabWidget::pane {{ border: 1px solid {_THEME['grid']}; border-radius: 8px; "
            f"background: {_THEME['card']}; top: -1px; }}"
            f"QTabBar::tab {{ background: #e2e8f0; color: {_THEME['muted']}; padding: 8px 18px; "
            f"margin-right: 3px; border-top-left-radius: 6px; border-top-right-radius: 6px; }}"
            f"QTabBar::tab:selected {{ background: {_THEME['card']}; color: {_THEME['text']}; font-weight: 600; }}"
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        header = QtWidgets.QHBoxLayout()
        title_col = QtWidgets.QVBoxLayout()
        title_col.setSpacing(2)
        title = QtWidgets.QLabel(self.node.get("candidate_id", "node"))
        title.setObjectName("title")
        status = self.node.get("status", "unknown")
        action = _action_label(self.node.get("action_type"), self.node.get("action_index"))
        sub = QtWidgets.QLabel(f"{status.capitalize()} · {action}")
        sub.setObjectName("sub")
        title_col.addWidget(title)
        title_col.addWidget(sub)
        header.addLayout(title_col)
        header.addStretch(1)
        layout.addLayout(header)

        chips_row = QtWidgets.QHBoxLayout()
        chips_row.setSpacing(10)
        for label, value in self._stat_chips():
            chip = QtWidgets.QLabel(f"<b>{label}</b><br>{value}")
            chip.setObjectName("chip")
            chip.setTextFormat(QtCore.Qt.RichText)
            chips_row.addWidget(chip)
        chips_row.addStretch(1)
        layout.addLayout(chips_row)

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self._build_metrics_tab(), "Metrics & plots")
        self._tabs.addTab(self._build_reward_tab(), "Reward function")
        layout.addWidget(self._tabs, 1)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet(
            f"QPushButton {{ background: {_THEME['primary']}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600; }}"
            f"QPushButton:hover {{ background: #2563eb; }}"
        )
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _build_metrics_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self._message = QtWidgets.QLabel()
        self._message.setWordWrap(True)
        self._message.setStyleSheet(f"color: {_THEME['muted']}; font-size: 12px; padding: 4px 0;")
        layout.addWidget(self._message)

        self._hover_status = QtWidgets.QLabel(
            "Hover over a line or point to see exact metric values")
        self._hover_status.setWordWrap(True)
        self._hover_status.setStyleSheet(
            f"color: {_THEME['text']}; font-size: 12px; font-weight: 600; "
            f"background: {_THEME['card']}; border: 1px solid {_THEME['grid']}; "
            f"border-radius: 6px; padding: 8px 10px;"
        )
        layout.addWidget(self._hover_status)

        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except ImportError:
            err = QtWidgets.QLabel(
                "Matplotlib is required for metrics plots.\n"
                "Install with: pip install matplotlib")
            err.setStyleSheet(f"color: {_THEME['text']}; padding: 24px;")
            layout.addWidget(err, 1)
            self._canvas = None
            self._hover = None
            self._toolbar = None
            return tab

        self._figure = Figure(figsize=(8.5, 5.5), dpi=100, facecolor=_THEME["bg"])
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setStyleSheet(f"background: {_THEME['bg']};")
        layout.addWidget(self._canvas, 1)
        self._hover = InteractiveChartTooltip(self._canvas, self._hover_status)
        self._toolbar = MetricsNavigationToolbar.create(self._canvas, self)
        self._toolbar.setStyleSheet(
            f"background: {_THEME['card']}; border-top: 1px solid {_THEME['grid']};"
        )
        layout.addWidget(self._toolbar)
        return tab

    def _build_reward_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        src = self._reward_source
        lang = src.get("language", "python")
        path = src.get("path", "")
        path_label = QtWidgets.QLabel(
            f"<b>File:</b> {path or '—'} &nbsp;·&nbsp; <b>Language:</b> {lang}"
        )
        path_label.setTextFormat(QtCore.Qt.RichText)
        path_label.setStyleSheet(f"color: {_THEME['muted']}; font-size: 12px;")
        layout.addWidget(path_label)

        thought = src.get("design_thought", "")
        if thought:
            thought_box = QtWidgets.QLabel(thought)
            thought_box.setWordWrap(True)
            thought_box.setStyleSheet(
                f"color: {_THEME['text']}; font-size: 12px; background: #eff6ff; "
                f"border: 1px solid #bfdbfe; border-radius: 8px; padding: 10px;"
            )
            layout.addWidget(thought_box)

        editor = QtWidgets.QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        editor.setPlainText(src.get("code", ""))
        editor.setStyleSheet(
            f"QPlainTextEdit {{ background: {_CODE_THEME['bg']}; color: {_CODE_THEME['text']}; "
            f"font-family: Consolas, 'Cascadia Code', 'Courier New', monospace; "
            f"font-size: 12px; border: 1px solid #334155; border-radius: 8px; padding: 10px; }}"
        )
        CodeSyntaxHighlighter(editor.document(), lang)
        layout.addWidget(editor, 1)

        if not src.get("exists"):
            note = QtWidgets.QLabel("Reward source file was not found on disk for this candidate.")
            note.setStyleSheet(f"color: #b45309; font-size: 12px;")
            layout.addWidget(note)

        return tab

    def _stat_chips(self) -> List[Tuple[str, str]]:
        n = self.node
        chips = [
            ("Tree reward", _fmt_num(n.get("score"))),
            ("Q value", _fmt_num(n.get("q_value"))),
            ("UCT", _fmt_num(n.get("uct_score"))),
            ("Visits", str(n.get("visits", 0))),
        ]
        pdata = self._plot_data
        key = pdata.get("primary_key")
        stats = pdata.get("primary_stats") or {}
        if key:
            chips.append((f"Max {key}", _fmt_num(stats.get("max"))))
            chips.append((f"Final {key}", _fmt_num(stats.get("last") or stats.get("final"))))
        if self._q_history:
            chips.append(("Q snapshots", str(len(self._q_history))))
        return chips

    def _plot_q_history(self, ax) -> bool:
        history = self._q_history
        if not history:
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "No Q history (node not trained or no replay yet)",
                ha="center",
                va="center",
                fontsize=11,
                color=_THEME["muted"],
                transform=ax.transAxes,
            )
            return False

        cid = self.node.get("candidate_id", "")
        sims = [int(h["sim"]) for h in history]
        qs = [float(h["q"]) for h in history]
        visits = [int(h.get("visits", 0)) for h in history]

        ax.plot(sims, qs, color=_THEME["accent"], linewidth=1.5, alpha=0.75, zorder=1)
        own_x, own_y = [], []
        anc_x, anc_y = [], []
        for h in history:
            sim = int(h["sim"])
            q = float(h["q"])
            if h.get("role") == "leaf" and h.get("trained_leaf") == cid:
                own_x.append(sim)
                own_y.append(q)
            else:
                anc_x.append(sim)
                anc_y.append(q)
        if anc_x:
            ax.scatter(
                anc_x, anc_y, s=36, color=_THEME["primary"], alpha=0.85,
                label="Q update (ancestor)", zorder=3, edgecolors="white", linewidths=0.6,
            )
        if own_x:
            ax.scatter(
                own_x, own_y, s=64, color=_THEME["success"], alpha=0.95,
                label="Own training replay", zorder=4, edgecolors="white", linewidths=0.8,
            )

        final_q = self.node.get("q_value")
        if _is_finite(final_q):
            ax.axhline(float(final_q), color=_THEME["success"], linewidth=1.0,
                       linestyle=":", alpha=0.55, label="final Q (tree)")

        ax.set_title(
            "Q value vs simulation replay (each trained node = 1 visit step)",
            fontsize=11,
            fontweight=600,
            pad=8,
        )
        ax.set_xlabel("Simulation index (trained nodes replayed in time order)")
        ax.set_ylabel("Q value")
        ax.grid(True, linestyle="-", linewidth=0.6)
        ax.legend(loc="best", fontsize=8, framealpha=0.95)

        ax2 = ax.twinx()
        ax2.plot(sims, visits, color=_THEME["warning"], linewidth=1.2, linestyle="--", alpha=0.7)
        ax2.set_ylabel("Visit count", color=_THEME["warning"])
        ax2.tick_params(axis="y", labelcolor=_THEME["warning"])

        if self._hover is not None:
            for h in history:
                sim = int(h["sim"])
                q = float(h["q"])
                role = h.get("role", "")
                leaf = h.get("trained_leaf", "—")
                if role == "leaf" and leaf == cid:
                    role_txt = "own training replay"
                else:
                    role_txt = "ancestor update"
                self._hover.add_point(
                    ax,
                    sim,
                    q,
                    (
                        f"Q — {role_txt}\n"
                        f"Simulation: {sim}\n"
                        f"Q: {_fmt_num(q)}\n"
                        f"Visits: {h.get('visits', 0)}\n"
                        f"Triggered by: {leaf}"
                    ),
                )
            self._hover.add_series(
                ax2, sims, visits, "Visit count", "Simulation", "Visits",
            )
        return True

    def _draw_charts(self):
        if self._canvas is None:
            return
        import matplotlib.pyplot as plt

        if self._hover is not None:
            self._hover.clear()
        self._figure.clear()
        plt.rcParams.update({
            "font.family": "Segoe UI",
            "font.size": 10,
            "axes.facecolor": _THEME["card"],
            "figure.facecolor": _THEME["bg"],
            "axes.edgecolor": _THEME["grid"],
            "axes.labelcolor": _THEME["text"],
            "xtick.color": _THEME["muted"],
            "ytick.color": _THEME["muted"],
            "grid.color": _THEME["grid"],
            "grid.alpha": 0.85,
        })

        pdata = self._plot_data
        series = pdata.get("series") or {}
        primary = pdata.get("primary_key")
        components = pdata.get("components") or {}
        has_reward = bool(series and primary and primary in series)
        has_q = bool(self._q_history) or self.node.get("status") == "trained"
        has_comp = len(components) > 0

        msg_parts = []
        if has_reward:
            msg_parts.append(f"Reward logs: {pdata.get('series_source') or 'logs'}")
        if self._q_history:
            msg_parts.append(f"Q history: {len(self._q_history)} snapshots from tree replay")
        elif has_q:
            msg_parts.append("Q history: unavailable")
        hint = "Hover near points for exact values · use toolbar to zoom/pan"
        base = " · ".join(msg_parts) if msg_parts else ""
        self._message.setText(f"{base} · {hint}" if base else hint)

        panel_count = sum([has_reward, has_q, has_comp])
        if panel_count == 0:
            ax = self._figure.add_subplot(111)
            ax.set_axis_off()
            ax.text(
                0.5, 0.5, "No metrics available",
                ha="center", va="center", fontsize=14, color=_THEME["muted"],
                transform=ax.transAxes,
            )
            self._canvas.draw()
            return

        row = 1
        if has_reward:
            ax_main = self._figure.add_subplot(panel_count, 1, row)
            row += 1
            values = [float(v) for v in series[primary] if _is_finite(v)]
            xs, ys = _downsample(values)
            ax_main.plot(xs, ys, color=_THEME["primary"], linewidth=1.8, label=primary)
            if len(values) > 1:
                running_max = []
                best = float("-inf")
                for v in values:
                    best = max(best, v)
                    running_max.append(best)
                rx, ry = _downsample(running_max)
                ax_main.plot(
                    rx, ry, color=_THEME["success"], linewidth=1.2, linestyle="--",
                    alpha=0.85, label="running max",
                )
            stats = pdata.get("primary_stats") or {}
            if _is_finite(stats.get("max")):
                ax_main.axhline(
                    float(stats["max"]), color=_THEME["success"], linewidth=0.9,
                    alpha=0.45, linestyle=":",
                )
            ax_main.set_title(f"{primary} over training", fontsize=12, fontweight=600, pad=10)
            ax_main.set_xlabel("Step / episode index")
            ax_main.set_ylabel(primary)
            ax_main.grid(True, linestyle="-", linewidth=0.6)
            ax_main.legend(loc="upper right", framealpha=0.95, fontsize=9)
            if self._hover is not None:
                self._hover.add_series(
                    ax_main, xs, ys, primary, "Step / episode", primary,
                )
                if len(values) > 1:
                    self._hover.add_series(
                        ax_main, rx, ry, "Running max", "Step / episode", primary,
                    )

        if has_q:
            ax_q = self._figure.add_subplot(panel_count, 1, row)
            row += 1
            self._plot_q_history(ax_q)

        if has_comp:
            ax_comp = self._figure.add_subplot(panel_count, 1, row)
            shown = 0
            colors = [_THEME["accent"], _THEME["warning"], "#06b6d4", "#ec4899", "#84cc16"]
            for idx, (name, info) in enumerate(sorted(components.items())):
                if shown >= 5:
                    break
                samples = info.get("samples") or []
                if len(samples) < 2:
                    continue
                cx, cy = _downsample(samples)
                ax_comp.plot(
                    cx, cy, color=colors[shown % len(colors)], linewidth=1.4,
                    alpha=0.9, label=name,
                )
                if self._hover is not None:
                    self._hover.add_series(
                        ax_comp, cx, cy, name, "Sample index", name,
                    )
                shown += 1
            if shown:
                ax_comp.set_title("Reward components (downsampled)", fontsize=11, fontweight=600, pad=8)
                ax_comp.set_xlabel("Sample index")
                ax_comp.grid(True, linestyle="-", linewidth=0.6)
                ax_comp.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.95)

        eval_metrics = pdata.get("eval_metrics")
        if eval_metrics and msg_parts:
            note = " · eval: " + ", ".join(_fmt_num(v) for v in eval_metrics[:6])
            if len(eval_metrics) > 6:
                note += " …"
            self._message.setText(self._message.text() + note)

        self._figure.tight_layout(pad=2.2)
        if self._hover is not None:
            self._hover.bind_axes()
        self._canvas.draw()


def open_node_metrics(
    node: dict,
    candidate_folder: Optional[Path],
    parent=None,
    *,
    task_dir: Optional[Path] = None,
    package_root: Optional[Path] = None,
) -> NodeMetricsDialog:
    q_history: List[dict] = []
    cid = node.get("candidate_id", "")
    if task_dir and package_root and cid and cid != "root":
        q_history = load_q_value_history(Path(task_dir), Path(package_root), cid)
    dialog = NodeMetricsDialog(
        node, candidate_folder, parent=parent, q_history=q_history,
    )
    dialog.setWindowModality(QtCore.Qt.NonModal)
    dialog.show()
    return dialog


def _fitness_from_summary(folder: Optional[Path]) -> Optional[float]:
    """Training fitness from summary.json (max task score / primary metric)."""
    if folder is None:
        return None
    path = Path(folder) / "summary.json"
    if not path.exists():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    for key in (
        "max_task_score",
        "max_reward",
        "max_rewards",
        "max_EpisodeReward",
        "final_task_score",
    ):
        if key in summary and _is_finite(summary[key]):
            return float(summary[key])
    metrics = summary.get("metrics")
    if isinstance(metrics, dict):
        for name in _PRIMARY_NAMES:
            stats = metrics.get(name)
            if isinstance(stats, dict) and _is_finite(stats.get("max")):
                return float(stats["max"])
    return None


def _row_for_node(node: dict, folder: Optional[Path]) -> dict:
    cid = node.get("candidate_id", "")
    fitness = _fitness_from_summary(folder)
    return {
        "candidate_id": cid,
        "status": node.get("status", "—"),
        "action": _action_label(node.get("action_type"), node.get("action_index")),
        "reward": node.get("score"),
        "q_value": node.get("q_value"),
        "fitness": fitness,
        "uct": node.get("uct_score"),
        "visits": node.get("visits", 0),
        "verify": node.get("self_verify_score"),
        "is_best": bool(node.get("is_best")),
    }


class NodeComparisonDialog(QtWidgets.QDialog):
    """Table + bar chart comparing reward, Q, fitness, and UCT for selected nodes."""

    _COLUMNS = [
        ("Candidate", "candidate_id"),
        ("Status", "status"),
        ("Action", "action"),
        ("Reward (tree)", "reward"),
        ("Q value", "q_value"),
        ("Fitness (logs)", "fitness"),
        ("UCT", "uct"),
        ("Visits", "visits"),
        ("Verify", "verify"),
    ]

    def __init__(
        self,
        nodes: List[dict],
        folders: Optional[Dict[str, Path]] = None,
        parent=None,
    ):
        super().__init__(parent)
        folders = folders or {}
        self.setWindowTitle(f"Compare nodes ({len(nodes)})")
        self.resize(max(720, 120 * len(nodes)), min(640, 120 + 48 * len(nodes)))
        self.setMinimumSize(640, 320)
        self._rows = [_row_for_node(n, folders.get(n.get("candidate_id", ""))) for n in nodes]
        self._build_ui()
        self._draw_chart()

    def _build_ui(self):
        self.setStyleSheet(
            f"QDialog {{ background: {_THEME['bg']}; }}"
            f"QLabel#title {{ color: {_THEME['text']}; font-size: 17px; font-weight: 700; }}"
            f"QLabel#sub {{ color: {_THEME['muted']}; font-size: 12px; }}"
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("Selected nodes — metrics comparison")
        title.setObjectName("title")
        sub = QtWidgets.QLabel(
            "Reward and Q come from the search tree. Fitness is max training score from "
            "summary.json when synced.")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(sub)

        table = QtWidgets.QTableWidget(len(self._rows), len(self._COLUMNS))
        table.setHorizontalHeaderLabels([c[0] for c in self._COLUMNS])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setAlternatingRowColors(True)
        table.setStyleSheet(
            f"QTableWidget {{ background: {_THEME['card']}; gridline-color: {_THEME['grid']}; "
            f"font-size: 12px; }}"
            f"QHeaderView::section {{ background: #f1f5f9; color: {_THEME['text']}; "
            f"padding: 6px; font-weight: 600; border: none; }}"
        )
        numeric_cols = {3, 4, 5, 6, 7, 8}
        for row, data in enumerate(self._rows):
            for col, (_, key) in enumerate(self._COLUMNS):
                raw = data.get(key)
                if key == "visits":
                    text = str(raw)
                elif key in ("reward", "q_value", "fitness", "uct", "verify"):
                    text = _fmt_num(raw)
                else:
                    text = str(raw) if raw is not None else "—"
                item = QtWidgets.QTableWidgetItem(text)
                if col in numeric_cols and raw is not None:
                    try:
                        item.setData(QtCore.Qt.UserRole, float(raw))
                    except (TypeError, ValueError):
                        pass
                if data.get("is_best") and col == 0:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    item.setForeground(QtGui.QColor(_THEME["success"]))
                table.setItem(row, col, item)
            table.setRowHeight(row, 32)
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        for c in range(1, table.columnCount()):
            table.horizontalHeader().setSectionResizeMode(c, QtWidgets.QHeaderView.Stretch)
        table.setSortingEnabled(True)
        layout.addWidget(table, 2)

        self._hover_status = QtWidgets.QLabel(
            "Hover a bar or UCT point for exact metric values")
        self._hover_status.setWordWrap(True)
        self._hover_status.setStyleSheet(
            f"color: {_THEME['text']}; font-size: 12px; font-weight: 600; "
            f"background: {_THEME['card']}; border: 1px solid {_THEME['grid']}; "
            f"border-radius: 6px; padding: 8px 10px;"
        )
        layout.addWidget(self._hover_status)

        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._figure = Figure(figsize=(7.5, 2.8), dpi=100, facecolor=_THEME["bg"])
            self._canvas = FigureCanvasQTAgg(self._figure)
            layout.addWidget(self._canvas, 1)
            self._hover = InteractiveChartTooltip(self._canvas, self._hover_status)
            self._toolbar = MetricsNavigationToolbar.create(self._canvas, self)
            self._toolbar.setStyleSheet(
                f"background: {_THEME['card']}; border-top: 1px solid {_THEME['grid']};"
            )
            layout.addWidget(self._toolbar)
        except ImportError:
            self._figure = None
            self._canvas = None
            self._hover = None
            self._toolbar = None

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setFixedWidth(96)
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet(
            f"QPushButton {{ background: {_THEME['primary']}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 14px; font-weight: 600; }}"
        )
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close_btn)
        layout.addLayout(row)

    def _draw_chart(self):
        if self._canvas is None or not self._rows:
            return
        import matplotlib.pyplot as plt
        import numpy as np

        if self._hover is not None:
            self._hover.clear()
        self._figure.clear()
        labels = [_short_label(r["candidate_id"]) for r in self._rows]
        x = np.arange(len(labels))
        width = 0.25
        metrics = [
            ("Reward", "reward", _THEME["primary"]),
            ("Q", "q_value", _THEME["success"]),
            ("Fitness", "fitness", _THEME["accent"]),
        ]
        ax = self._figure.add_subplot(111)
        for i, (title, key, color) in enumerate(metrics):
            vals = []
            for row in self._rows:
                v = row.get(key)
                try:
                    vals.append(float(v) if v is not None and _is_finite(v) else 0.0)
                except (TypeError, ValueError):
                    vals.append(0.0)
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals, width, label=title, color=color, alpha=0.88)
            if self._hover is not None:
                for bar, row, val in zip(bars, self._rows, vals):
                    cx = bar.get_x() + bar.get_width() / 2
                    cy = bar.get_height()
                    self._hover.add_point(
                        ax,
                        cx,
                        cy,
                        (
                            f"{row['candidate_id']}\n{title}: {_fmt_num(val)}\n"
                            f"Reward: {_fmt_num(row.get('reward'))}\n"
                            f"Q: {_fmt_num(row.get('q_value'))}\n"
                            f"Fitness: {_fmt_num(row.get('fitness'))}\n"
                            f"UCT: {_fmt_num(row.get('uct'))}"
                        ),
                    )

        uct_vals = []
        for row in self._rows:
            v = row.get("uct")
            try:
                uct_vals.append(float(v) if v is not None and _is_finite(v) else 0.0)
            except (TypeError, ValueError):
                uct_vals.append(0.0)
        ax2 = ax.twinx()
        ax2.plot(x, uct_vals, color=_THEME["warning"], marker="o", linewidth=2,
                 markersize=7, label="UCT")
        if self._hover is not None:
            for xi, row, uct in zip(x, self._rows, uct_vals):
                self._hover.add_point(
                    ax2,
                    float(xi),
                    uct,
                    f"{row['candidate_id']}\nUCT: {_fmt_num(uct)}",
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title("Reward · Q · Fitness (bars) and UCT (line)", fontsize=11, fontweight=600)
        ax.grid(True, axis="y", alpha=0.35)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
        self._figure.tight_layout(pad=1.2)
        if self._hover is not None:
            self._hover.bind_axes()
        self._canvas.draw()


def _short_label(candidate_id: str) -> str:
    if not candidate_id:
        return "?"
    if "_" in candidate_id:
        return "c" + candidate_id.rsplit("_", 1)[-1]
    return candidate_id[:10]


def open_node_comparison(
    nodes: List[dict],
    folders: Optional[Dict[str, Path]] = None,
    parent=None,
) -> NodeComparisonDialog:
    dialog = NodeComparisonDialog(nodes, folders, parent=parent)
    dialog.setWindowModality(QtCore.Qt.NonModal)
    dialog.show()
    return dialog

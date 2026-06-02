from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def _resolve_dynamics_data(dynamics: Any) -> Optional[Dict[str, Any]]:
    """Accept either a ParamDynamics object or a plain dict (e.g. loaded from JSON)."""
    if dynamics is None:
        return None
    if hasattr(dynamics, "to_dict"):
        return dynamics.to_dict()
    if isinstance(dynamics, dict):
        return dynamics
    return None


def _smooth(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def _resolve_module(name: str) -> str:
    return name.split(".")[0]


def _module_color_map(modules: Sequence[str]):
    cmap = plt.get_cmap("tab10")
    unique = sorted(set(modules))
    return {m: cmap(i % 10) for i, m in enumerate(unique)}


def visualize_param_dynamics(
    dynamics: Any,
    output_dir: str,
    config: Optional[Any] = None,
) -> None:
    """Entry point: generate all parameter dynamics visualizations."""
    data = _resolve_dynamics_data(dynamics)
    if data is None or not data.get("layer_names"):
        return

    out = Path(output_dir) / "param_dynamics"
    out.mkdir(parents=True, exist_ok=True)

    top_n = 30
    smooth_window = 3
    if config is not None:
        top_n = int(getattr(config, "top_n_layers", getattr(config, "get", lambda *a: 30)("param_tracking.visualization.top_n_layers", 30)))
        smooth_window = int(getattr(config, "alignment_smooth_window", getattr(config, "get", lambda *a: 3)("param_tracking.visualization.alignment_smooth_window", 3)))

    _plot_update_magnitude_heatmap(data, out, top_n)
    _plot_direction_alignment(data, out, smooth_window)
    _plot_cumulative_progress(data, out, smooth_window)
    _plot_layer_convergence_radar(data, out, top_n)
    _plot_layer_group_summary(data, out)


def _select_top_layers(data: Dict[str, Any], top_n: int) -> List[str]:
    layer_names = data.get("layer_names", [])
    magnitude = data.get("update_magnitude", {})
    totals = {name: sum(magnitude.get(name, [0.0])) for name in layer_names}
    ranked = sorted(layer_names, key=lambda n: totals.get(n, 0.0), reverse=True)
    return ranked[:top_n]


def _stage_boundaries(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    epochs = data.get("epochs", [])
    stages = data.get("stage_names", [])
    if not epochs or not stages or len(epochs) != len(stages):
        return []
    boundaries: List[Dict[str, Any]] = []
    prev = ""
    for i, s in enumerate(stages):
        if s != prev and epochs[i] > 0:
            boundaries.append({"epoch": epochs[i], "name": s})
        prev = s
    return boundaries


def _plot_update_magnitude_heatmap(data: Dict[str, Any], out: Path, top_n: int):
    layer_names = _select_top_layers(data, top_n)
    epochs = data.get("epochs", [])
    magnitude = data.get("update_magnitude", {})
    if not layer_names or not epochs:
        return

    mat = np.zeros((len(layer_names), len(epochs)), dtype=np.float64)
    for i, name in enumerate(layer_names):
        vals = magnitude.get(name, [0.0] * len(epochs))
        mat[i, :] = np.asarray(vals[:len(epochs)], dtype=np.float64)

    mat_log = np.log1p(mat)
    boundaries = _stage_boundaries(data)

    fig, ax = plt.subplots(figsize=(max(10, len(epochs) * 0.35), max(5, len(layer_names) * 0.22)))
    im = ax.imshow(mat_log, cmap="viridis", aspect="auto")

    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels([str(e) for e in epochs], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("layer")
    ax.set_title("layer-wise update magnitude (log1p)")

    for b in boundaries:
        ax.axvline(x=epochs.index(b["epoch"]) if b["epoch"] in epochs else -1, color="red", linestyle="--", linewidth=1.2, alpha=0.6)

    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="log(1 + ||step||)")
    fig.tight_layout()
    fig.savefig(out / "update_magnitude_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_direction_alignment(data: Dict[str, Any], out: Path, smooth_window: int):
    layer_names = data.get("layer_names", [])
    epochs = data.get("epochs", [])
    cosine = data.get("direction_cosine", {})
    if not layer_names or not epochs:
        return

    modules = [_resolve_module(n) for n in layer_names]
    colors = _module_color_map(modules)
    boundaries = _stage_boundaries(data)

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(epochs) * 0.3), 9))

    # Top: all layers (faded)
    ax = axes[0]
    for name in layer_names:
        vals = np.asarray(cosine.get(name, [0.0] * len(epochs)), dtype=np.float64)
        vals = _smooth(vals[:len(epochs)], smooth_window)
        ax.plot(epochs[:len(vals)], vals, color=colors[_resolve_module(name)], alpha=0.22, linewidth=0.6)
    ax.axhline(y=0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=1.0, color="green", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("cosine similarity")
    ax.set_title("per-layer update direction alignment (cosine with global direction)")
    ax.grid(alpha=0.25)

    # Bottom: module-grouped averages
    ax = axes[1]
    module_groups: Dict[str, List[str]] = {}
    for name in layer_names:
        module_groups.setdefault(_resolve_module(name), []).append(name)

    for module, names in sorted(module_groups.items()):
        curves = []
        for name in names:
            vals = np.asarray(cosine.get(name, [0.0] * len(epochs)), dtype=np.float64)
            curves.append(vals[:len(epochs)])
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        stacked = np.stack([c[:min_len] for c in curves], axis=0)
        mean_curve = np.mean(stacked, axis=0)
        mean_curve = _smooth(mean_curve, smooth_window)
        ax.plot(epochs[:len(mean_curve)], mean_curve, color=colors[module], linewidth=1.8, label=module)
    ax.axhline(y=0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cosine similarity")
    ax.set_title("module-grouped average direction alignment")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    for ax in axes:
        for b in boundaries:
            ax.axvline(x=b["epoch"], color="red", linestyle="--", linewidth=1.0, alpha=0.35)
            if b["epoch"] <= (epochs[-1] if epochs else 0):
                ax.text(b["epoch"], ax.get_ylim()[1] * 0.95, b["name"], fontsize=7,
                        color="red", ha="left", va="top", rotation=90, alpha=0.7)

    fig.tight_layout()
    fig.savefig(out / "direction_alignment.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_cumulative_progress(data: Dict[str, Any], out: Path, smooth_window: int):
    layer_names = data.get("layer_names", [])
    epochs = data.get("epochs", [])
    progress = data.get("cumulative_progress", {})
    if not layer_names or not epochs:
        return

    modules = [_resolve_module(n) for n in layer_names]
    colors = _module_color_map(modules)
    boundaries = _stage_boundaries(data)

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(epochs) * 0.3), 9))

    ax = axes[0]
    for name in layer_names:
        vals = np.asarray(progress.get(name, [0.0] * len(epochs)), dtype=np.float64)
        vals = _smooth(vals[:len(epochs)], smooth_window)
        ax.plot(epochs[:len(vals)], vals, color=colors[_resolve_module(name)], alpha=0.22, linewidth=0.6)
    ax.axhline(y=1.0, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("cosine similarity")
    ax.set_title("per-layer cumulative progress (cosine with global direction)")
    ax.grid(alpha=0.25)

    ax = axes[1]
    module_groups: Dict[str, List[str]] = {}
    for name in layer_names:
        module_groups.setdefault(_resolve_module(name), []).append(name)

    for module, names in sorted(module_groups.items()):
        curves = []
        for name in names:
            vals = np.asarray(progress.get(name, [0.0] * len(epochs)), dtype=np.float64)
            curves.append(vals[:len(epochs)])
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        stacked = np.stack([c[:min_len] for c in curves], axis=0)
        mean_curve = np.mean(stacked, axis=0)
        mean_curve = _smooth(mean_curve, smooth_window)
        ax.plot(epochs[:len(mean_curve)], mean_curve, color=colors[module], linewidth=1.8, label=module)
    ax.axhline(y=1.0, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cosine similarity")
    ax.set_title("module-grouped average cumulative progress")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    for ax in axes:
        for b in boundaries:
            ax.axvline(x=b["epoch"], color="red", linestyle="--", linewidth=1.0, alpha=0.35)

    fig.tight_layout()
    fig.savefig(out / "cumulative_progress.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_layer_convergence_radar(data: Dict[str, Any], out: Path, top_n: int):
    layer_names = _select_top_layers(data, top_n)
    progress = data.get("cumulative_progress", {})
    if not layer_names:
        return

    values = [progress.get(n, [0.0])[-1] for n in layer_names]
    values = [max(-1.0, min(1.0, v)) for v in values]

    n = len(layer_names)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    ax.fill(angles, values, color="#4e79a7", alpha=0.25)
    ax.plot(angles, values, color="#4e79a7", linewidth=1.8, marker="o", markersize=4)
    ax.set_xticks(angles[:-1])
    short_names = [n.split(".")[-1][:20] for n in layer_names]
    ax.set_xticklabels(short_names, fontsize=6)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels(["-1.0", "-0.5", "0.0", "0.5", "1.0"], fontsize=7)
    ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axhline(y=1.0, color="green", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_title("layer convergence radar (final cumulative progress)", pad=20)
    fig.tight_layout()
    fig.savefig(out / "layer_convergence_radar.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_layer_group_summary(data: Dict[str, Any], out: Path):
    layer_names = data.get("layer_names", [])
    progress = data.get("cumulative_progress", {})
    magnitude = data.get("update_magnitude", {})
    if not layer_names:
        return

    module_groups: Dict[str, Dict[str, List[float]]] = {}
    for name in layer_names:
        m = _resolve_module(name)
        module_groups.setdefault(m, {"progress": [], "magnitude": []})
        p_vals = progress.get(name, [])
        m_vals = magnitude.get(name, [])
        if p_vals:
            module_groups[m]["progress"].append(p_vals[-1])
        if m_vals:
            module_groups[m]["magnitude"].append(sum(m_vals))

    modules_sorted = sorted(module_groups.keys())
    n = len(modules_sorted)
    if n == 0:
        return

    x = np.arange(n)
    width = 0.35

    avg_progress = [np.mean(module_groups[m]["progress"]) if module_groups[m]["progress"] else 0.0 for m in modules_sorted]
    avg_magnitude = [np.mean(module_groups[m]["magnitude"]) if module_groups[m]["magnitude"] else 0.0 for m in modules_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(max(8, n * 1.2), 5))

    ax = axes[0]
    bars = ax.bar(x, avg_progress, width * 2, color="#4e79a7", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(modules_sorted, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("avg final cumulative progress")
    ax.set_title("module-wise final convergence")
    ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(y=1.0, color="green", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, avg_progress):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03 * np.sign(bar.get_height() or 1),
                f"{val:.2f}", ha="center", fontsize=7)

    ax = axes[1]
    bars = ax.bar(x, avg_magnitude, width * 2, color="#e15759", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(modules_sorted, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("avg total update magnitude")
    ax.set_title("module-wise total update magnitude")
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, avg_magnitude):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(avg_magnitude) * 0.02,
                f"{val:.3f}", ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out / "layer_group_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

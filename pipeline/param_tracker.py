from __future__ import annotations

import re
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from munch import Munch

from pipeline.monitor import tprint


def _cfg_get(config: Any, path: str, default: Any = None) -> Any:
    current = config
    for key in path.split("."):
        if current is None:
            return default
        if isinstance(current, Munch):
            current = getattr(current, key, None)
        elif isinstance(current, dict):
            current = current.get(key, None)
        else:
            current = getattr(current, key, None)
    return default if current is None else current


@dataclass
class ParamDynamics:
    layer_names: List[str]
    epochs: List[int]
    stage_names: List[str]
    global_direction: Dict[str, np.ndarray]
    update_magnitude: Dict[str, List[float]]
    direction_cosine: Dict[str, List[float]]
    cumulative_progress: Dict[str, List[float]]

    def module_summary(self) -> Dict[str, Dict[str, float]]:
        groups: Dict[str, List[str]] = {}
        for name in self.layer_names:
            module = name.split(".")[0]
            groups.setdefault(module, []).append(name)

        summary: Dict[str, Dict[str, float]] = {}
        for module, names in groups.items():
            progress_vals = [self.cumulative_progress[n][-1] for n in names if self.cumulative_progress.get(n)]
            magnitude_vals = [sum(self.update_magnitude.get(n, [0.0])) for n in names]
            summary[module] = {
                "avg_final_progress": float(np.mean(progress_vals)) if progress_vals else 0.0,
                "avg_total_update": float(np.mean(magnitude_vals)) if magnitude_vals else 0.0,
            }
        return summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_names": self.layer_names,
            "num_layers": len(self.layer_names),
            "num_epochs": len(self.epochs),
            "epochs": self.epochs,
            "stage_names": self.stage_names,
            "global_direction_norm": {k: float(np.linalg.norm(v)) for k, v in self.global_direction.items()},
            "update_magnitude": self.update_magnitude,
            "direction_cosine": self.direction_cosine,
            "cumulative_progress": self.cumulative_progress,
            "module_summary": self.module_summary(),
        }


class ParamTracker:
    def __init__(self, model: nn.Module, config: Munch):
        self.model = model
        self.config = config
        self._pt_cfg = _cfg_get(config, "param_tracking", Munch())

        self.save_full_snapshots = bool(_cfg_get(config, "param_tracking.save_full_snapshots", False))
        self.snapshot_interval = max(1, int(_cfg_get(config, "param_tracking.snapshot_interval", 1)))

        self.mode = str(_cfg_get(config, "param_tracking.layers.mode", "all"))
        self.include_keywords = list(_cfg_get(config, "param_tracking.layers.include_keywords", []))
        self.include_patterns = list(_cfg_get(config, "param_tracking.layers.include_patterns", []))
        self.exclude_keywords = list(_cfg_get(config, "param_tracking.layers.exclude_keywords", [
            "norm", "bn", "gn", "running_mean", "running_var", "num_batches_tracked",
        ]))

        self.layer_names = self._resolve_tracked_layers()
        tprint(f"param_tracker: tracking {len(self.layer_names)} layers "
               f"(mode={self.mode}, exclude={self.exclude_keywords})")

        self._snapshots: List[Dict[str, np.ndarray]] = []
        self._recorded_epochs: List[int] = []
        self._recorded_stages: List[str] = []

    def _resolve_tracked_layers(self) -> List[str]:
        layers: List[str] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name for kw in self.exclude_keywords):
                continue
            if self.mode == "all":
                layers.append(name)
            elif self.mode == "named":
                if any(kw in name for kw in self.include_keywords):
                    layers.append(name)
            elif self.mode == "regex":
                if self.include_patterns:
                    if any(re.search(pat, name) for pat in self.include_patterns):
                        layers.append(name)
                else:
                    layers.append(name)
        return sorted(layers)

    def capture_snapshot(self) -> Dict[str, np.ndarray]:
        state = self.model.state_dict()
        snap: Dict[str, np.ndarray] = {}
        for name in self.layer_names:
            if name in state:
                tensor = state[name].detach().cpu()
                snap[name] = tensor.numpy().astype(np.float32).ravel()
        return snap

    def record_epoch(self, epoch: int, stage_name: str):
        if self.snapshot_interval > 1 and epoch % self.snapshot_interval != 0:
            return
        snap = self.capture_snapshot()
        self._snapshots.append(snap)
        self._recorded_epochs.append(epoch)
        self._recorded_stages.append(stage_name)

    def finalize(self) -> ParamDynamics:
        tprint("param_tracker: finalizing dynamics metrics...")
        if len(self._snapshots) < 2:
            tprint("param_tracker: insufficient snapshots (<2), returning empty dynamics")
            return ParamDynamics(
                layer_names=list(self.layer_names),
                epochs=self._recorded_epochs,
                stage_names=self._recorded_stages,
                global_direction={},
                update_magnitude={},
                direction_cosine={},
                cumulative_progress={},
            )

        snap0 = self._snapshots[0]
        snapT = self._snapshots[-1]

        global_direction: Dict[str, np.ndarray] = {}
        for name in self.layer_names:
            if name in snap0 and name in snapT:
                global_direction[name] = snapT[name] - snap0[name]

        epochs = self._recorded_epochs
        stages = self._recorded_stages

        update_magnitude: Dict[str, List[float]] = {name: [] for name in self.layer_names}
        direction_cosine: Dict[str, List[float]] = {name: [] for name in self.layer_names}
        cumulative_progress: Dict[str, List[float]] = {name: [] for name in self.layer_names}

        for i, (name) in enumerate(self.layer_names):
            if name not in global_direction or name not in snap0:
                for _ in range(len(epochs)):
                    update_magnitude[name].append(0.0)
                    direction_cosine[name].append(0.0)
                    cumulative_progress[name].append(0.0)
                continue

            g_dir = global_direction[name]
            g_norm = float(np.linalg.norm(g_dir))
            init_vec = snap0[name]

            for t in range(len(epochs)):
                cur = self._snapshots[t][name]
                if t == 0:
                    update_magnitude[name].append(0.0)
                    direction_cosine[name].append(0.0)
                    cumulative_progress[name].append(0.0)
                else:
                    prev = self._snapshots[t - 1][name]
                    step = cur - prev
                    step_norm = float(np.linalg.norm(step))
                    update_magnitude[name].append(step_norm)

                    if step_norm > 1e-12 and g_norm > 1e-12:
                        cos_step = float(np.dot(step, g_dir) / (step_norm * g_norm))
                    else:
                        cos_step = 0.0
                    direction_cosine[name].append(cos_step)

                    cum = cur - init_vec
                    cum_norm = float(np.linalg.norm(cum))
                    if cum_norm > 1e-12 and g_norm > 1e-12:
                        cos_cum = float(np.dot(cum, g_dir) / (cum_norm * g_norm))
                    else:
                        cos_cum = 0.0
                    cumulative_progress[name].append(cos_cum)

        tprint("param_tracker: dynamics finalized.")
        return ParamDynamics(
            layer_names=list(self.layer_names),
            epochs=epochs,
            stage_names=stages,
            global_direction=global_direction,
            update_magnitude=update_magnitude,
            direction_cosine=direction_cosine,
            cumulative_progress=cumulative_progress,
        )

    def save(self, output_dir: Path) -> None:
        dynamics = self.finalize()
        json_path = output_dir / "param_dynamics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(dynamics.to_dict(), f, indent=2, ensure_ascii=False)
        tprint(f"param_tracker: dynamics saved to {json_path}")

        if self.save_full_snapshots and self._snapshots:
            npz_path = output_dir / "param_snapshots.npz"
            payload: Dict[str, np.ndarray] = {
                "epochs": np.array(self._recorded_epochs, dtype=np.int32),
                "layer_names": np.array(self.layer_names, dtype=object),
            }
            for i, (name) in enumerate(self.layer_names):
                key = f"snap_{i}"
                data = np.stack([s[name] for s in self._snapshots], axis=0)
                payload[key] = data.astype(np.float32)
            np.savez_compressed(str(npz_path), **payload)
            tprint(f"param_tracker: full snapshots saved to {npz_path}")

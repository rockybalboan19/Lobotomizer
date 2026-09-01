from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch


def choose_experts_to_keep(activation_stats: Dict[str, Any], keep_fraction: float, adapter: Optional[Dict[str, Any]] = None) -> Dict[str, List[int]]:
    """Rank experts by the router's actual measurable signal: total gate weight assigned to each expert.

    The full EASY-EP formula requires the expert output norm and the residual-stream shift, which cannot be
    computed from the router hook alone. This function therefore uses the observable gating-score baseline and
    does not claim to implement the complete paper formula.
    """
    keep_map: Dict[str, List[int]] = {}
    for layer_idx, stats in sorted(activation_stats.items(), key=lambda kv: int(kv[0])):
        counts = {int(k): v for k, v in stats["expert_counts"].items()}
        if not counts:
            keep_map[layer_idx] = []
            continue

        gating_scores = {}
        for expert_id, activation_count in counts.items():
            avg_weight = stats.get("average_weights", {}).get(str(expert_id), 0.0)
            score_weight = stats.get("gating_score", {}).get(str(expert_id), 0.0)
            gating_scores[expert_id] = float(score_weight if score_weight else (avg_weight * activation_count))

        target = max(1, int(round(len(counts) * keep_fraction)))
        target = min(target, len(counts))
        ranked = sorted(gating_scores.items(), key=lambda kv: kv[1], reverse=True)
        keep = [expert_id for expert_id, _ in ranked[:target]]
        keep_map[layer_idx] = sorted(keep)
    return keep_map


def prune_layer_experts(layer: torch.nn.Module, keep_indices: Sequence[int]) -> None:
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return
    experts = getattr(mlp, "experts", None)
    if experts is None:
        return

    keep_indices = sorted(set(int(i) for i in keep_indices))
    if not keep_indices:
        return

    if hasattr(experts, "gate_up_proj"):
        gate_up = experts.gate_up_proj.detach().clone()
        down = experts.down_proj.detach().clone()
        experts.gate_up_proj = torch.nn.Parameter(gate_up[keep_indices])
        experts.down_proj = torch.nn.Parameter(down[keep_indices])
        try:
            experts.num_experts = len(keep_indices)
        except Exception:
            pass

    if hasattr(mlp, "gate"):
        gate = mlp.gate
        if hasattr(gate, "weight"):
            gate.weight = torch.nn.Parameter(gate.weight[keep_indices].detach().clone())
        if hasattr(gate, "bias") and gate.bias is not None:
            gate.bias = torch.nn.Parameter(gate.bias[keep_indices].detach().clone())
        try:
            gate.num_experts = len(keep_indices)
        except Exception:
            pass


def build_daughter_model(parent_model: torch.nn.Module, keep_map: Dict[str, List[int]], output_dir: str) -> torch.nn.Module:
    daughter = copy.deepcopy(parent_model)
    daughter.eval()

    for layer_idx, layer in enumerate(daughter.model.layers):
        key = str(layer_idx)
        if key not in keep_map:
            continue
        keep = keep_map[key]
        if not keep:
            continue
        prune_layer_experts(layer, keep)

    return daughter

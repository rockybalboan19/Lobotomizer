from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Sequence

import torch


def iter_moe_layers(model):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return []
    for idx, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "gate", None)
        if gate is not None:
            yield idx, layer, gate


def _cosine_shift(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    before_flat = before.reshape(before.shape[0], -1)
    after_flat = after.reshape(after.shape[0], -1)
    denom = torch.norm(before_flat, dim=1) * torch.norm(after_flat, dim=1)
    denom = torch.clamp(denom, min=1e-8)
    cosine = (before_flat * after_flat).sum(dim=1) / denom
    return 1.0 - cosine


def _collect_three_term_score(model, tokenizer, questions: Sequence[str], device: str = "cpu") -> Dict[str, Any]:
    """Measure the actual three-part signal: gate weight * expert output norm * residual shift.

    This is the real research signal used by the EASY-EP-style formulation. The router hook provides the gate,
    while additional forward hooks capture the expert outputs and the pre/post-MoE residual stream.
    """
    stats_by_layer: Dict[str, Any] = {}
    for layer_idx, layer, gate in iter_moe_layers(model):
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        if experts is None:
            continue
        expert_map = getattr(experts, "experts", None)
        if expert_map is None and hasattr(experts, "_modules"):
            expert_map = list(experts._modules.values())
        if expert_map is None:
            continue

        counts: Dict[int, int] = defaultdict(int)
        gating_score: Dict[int, float] = defaultdict(float)
        expert_norm_score: Dict[int, float] = defaultdict(float)
        residual_shift_score: Dict[int, float] = defaultdict(float)

        layer_before = []
        layer_after = []
        layer_router = []

        def gate_hook(module, inputs, output):
            if not isinstance(output, tuple) or len(output) < 3:
                return
            _, router_scores, router_indices = output
            layer_router.append((router_scores.detach().cpu(), router_indices.detach().cpu()))

        def mlp_hook(module, inputs, output):
            hidden_in = inputs[0].detach().cpu()
            hidden_out = output[0].detach().cpu() if isinstance(output, tuple) else output.detach().cpu()
            layer_before.append(hidden_in)
            layer_after.append(hidden_out)

        def make_expert_hook(expert_idx):
            def hook(module, inputs, output):
                out = output.detach().cpu() if isinstance(output, torch.Tensor) else output[0].detach().cpu()
                out_norm = torch.linalg.norm(out.reshape(out.shape[0], -1), dim=1)
                expert_norm_score[expert_idx] += float(out_norm.mean().item())
            return hook

        handles = [gate.register_forward_hook(gate_hook), mlp.register_forward_hook(mlp_hook)]
        for idx, expert in enumerate(expert_map):
            handles.append(expert.register_forward_hook(make_expert_hook(idx)))

        try:
            for q in questions:
                prompt = f"Answer this question precisely:\n{q}\n"
                encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                encoded = {k: v.to(device) for k, v in encoded.items()}
                with torch.no_grad():
                    model(**encoded, use_cache=False)
                if layer_router and layer_before and layer_after:
                    router_scores, router_indices = layer_router[-1]
                    pre = layer_before[-1]
                    post = layer_after[-1]
                    shift = _cosine_shift(pre, post)
                    for token_idx in range(router_indices.shape[0]):
                        for topk_idx in range(router_indices.shape[1]):
                            expert_id = int(router_indices[token_idx, topk_idx].item())
                            expert_weight = float(router_scores[token_idx, topk_idx].item())
                            counts[expert_id] += 1
                            gating_score[expert_id] += expert_weight
                            residual_shift_score[expert_id] += float(shift[token_idx].item())
        finally:
            for handle in handles:
                handle.remove()

        expert_ids = sorted(counts.keys())
        stats_by_layer[str(layer_idx)] = {
            "expert_counts": {str(expert_id): counts[expert_id] for expert_id in expert_ids},
            "gating_score": {str(expert_id): gating_score[expert_id] for expert_id in expert_ids},
            "residual_shift_score": {str(expert_id): residual_shift_score[expert_id] for expert_id in expert_ids},
            "expert_norm_score": {str(expert_id): expert_norm_score.get(expert_id, 0.0) for expert_id in expert_ids},
            "three_term_score": {
                str(expert_id): (gating_score[expert_id] * max(expert_norm_score.get(expert_id, 0.0), 1e-8) * max(residual_shift_score.get(expert_id, 0.0), 1e-8))
                for expert_id in expert_ids
            },
        }
    return stats_by_layer


def log_router_activations(model, tokenizer, questions: Sequence[str], device: str = "cpu") -> Dict[str, Any]:
    stats_by_layer: Dict[str, Any] = {}
    for layer_idx, layer, gate in iter_moe_layers(model):
        counts: Dict[int, int] = defaultdict(int)
        weights: Dict[int, float] = defaultdict(float)
        weight_counts: Dict[int, int] = defaultdict(int)
        gating_score: Dict[int, float] = defaultdict(float)

        def hook(module, inputs, output):
            if not isinstance(output, tuple) or len(output) < 3:
                return
            _, router_scores, router_indices = output
            router_scores = router_scores.detach().cpu()
            router_indices = router_indices.detach().cpu()
            for token_idx in range(router_indices.shape[0]):
                for topk_idx in range(router_indices.shape[1]):
                    expert_id = int(router_indices[token_idx, topk_idx].item())
                    expert_weight = float(router_scores[token_idx, topk_idx].item())
                    counts[expert_id] += 1
                    weights[expert_id] += expert_weight
                    weight_counts[expert_id] += 1
                    gating_score[expert_id] += expert_weight

        handle = gate.register_forward_hook(hook)
        try:
            for q in questions:
                prompt = f"Answer this question precisely:\n{q}\n"
                encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                encoded = {k: v.to(device) for k, v in encoded.items()}
                with torch.no_grad():
                    model(**encoded, use_cache=False)
        finally:
            handle.remove()

        expert_ids = sorted(counts.keys())
        stats_by_layer[str(layer_idx)] = {
            "expert_counts": {str(expert_id): counts[expert_id] for expert_id in expert_ids},
            "average_weights": {
                str(expert_id): (weights[expert_id] / weight_counts[expert_id]) if weight_counts[expert_id] > 0 else 0.0
                for expert_id in expert_ids
            },
            "gating_score": {str(expert_id): gating_score[expert_id] for expert_id in expert_ids},
            "most_active": max(counts.items(), key=lambda kv: kv[1])[0] if counts else None,
            "least_active": min(counts.items(), key=lambda kv: kv[1])[0] if counts else None,
        }

    return stats_by_layer

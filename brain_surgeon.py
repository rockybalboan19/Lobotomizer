#!/usr/bin/env python3
"""MVP research script comparing baseline router ranking vs. the real three-term MoE score.

This demo run uses a synthetic model to verify the pipeline end-to-end. Scores are not meaningful until
run against a real Hugging Face MoE checkpoint; switch MODEL_PATH below when you are ready for a real run.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn

from src.benchmark import benchmark_heldout_set, print_benchmark_summary
from src.self_probe import generate_heldout_questions

MODEL_PATH = ""  # Set to a real HF model path like: "path/to/OLMoE" or a local checkpoint directory.


DOMAIN_TEMPLATES = {
    "finance": [
        "What is the difference between value at risk and expected shortfall?",
        "How does diversification reduce idiosyncratic risk in a portfolio?",
        "What does a negative yield curve typically signal for interest rate expectations?",
        "Why do central banks monitor inflation expectations in real time?",
        "How is a forward contract different from a futures contract?",
    ],
    "biology": [
        "How do membrane transporters help cells maintain homeostasis?",
        "Why are enzymes highly specific to their substrates?",
        "What is the role of ATP in cellular metabolism?",
        "How does gene expression differ from gene replication?",
        "Why is the cell membrane described as selectively permeable?",
    ],
    "software": [
        "What is the difference between compile-time and runtime errors?",
        "Why does a hash table provide average O(1) lookups?",
        "What is the tradeoff between eager and lazy evaluation?",
        "How does a load balancer improve service reliability?",
        "Why is caching useful for repeated read-heavy workloads?",
    ],
}


def encode_question(question: str, dim: int = 8) -> torch.Tensor:
    chars = [ord(ch) % 13 for ch in question[:32]]
    if len(chars) < dim:
        chars = chars + [0] * (dim - len(chars))
    else:
        chars = chars[:dim]
    return torch.tensor(chars, dtype=torch.float32).reshape(1, dim)


class MiniExpert(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MiniMoEBlock(nn.Module):
    def __init__(self, dim: int, num_experts: int = 6):
        super().__init__()
        self.dim = dim
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([MiniExpert(dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor):
        gate_logits = self.gate(x)
        router_scores = torch.softmax(gate_logits, dim=-1)
        topk_idx = torch.topk(router_scores, k=2, dim=-1).indices

        selected = []
        expert_outputs = []
        for token_i in range(x.shape[0]):
            for k in range(topk_idx.shape[1]):
                expert_id = int(topk_idx[token_i, k].item())
                expert_out = self.experts[expert_id](x[token_i])
                selected.append((expert_id, expert_out, float(router_scores[token_i, k].item())))
                expert_outputs.append((expert_id, expert_out, float(router_scores[token_i, k].item())))

        mixed = torch.zeros_like(x)
        for expert_id, expert_out, _ in selected:
            mixed += expert_out
        residual = x + mixed
        shift = 1.0 - (x * residual).sum(dim=-1) / (
            torch.linalg.norm(x, dim=-1) * torch.linalg.norm(residual, dim=-1) + 1e-8
        )
        return {
            "before": x,
            "after": residual,
            "router_scores": router_scores,
            "router_indices": topk_idx,
            "selected": selected,
            "expert_outputs": expert_outputs,
            "shift": shift,
        }


class MiniMoEModel(nn.Module):
    def __init__(self, dim: int = 8, num_layers: int = 3, num_experts: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([MiniMoEBlock(dim, num_experts) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        outputs = []
        current = x
        for layer in self.layers:
            debug = layer(current)
            outputs.append(debug)
            current = debug["after"]
        return outputs


def baseline_ranking(layer_debug: Dict[str, object]) -> Dict[int, float]:
    scores = defaultdict(float)
    router_scores = layer_debug["router_scores"]
    router_indices = layer_debug["router_indices"]
    for token_i in range(router_indices.shape[0]):
        for topk_i in range(router_indices.shape[1]):
            expert_id = int(router_indices[token_i, topk_i].item())
            scores[expert_id] += float(router_scores[token_i, topk_i].item())
    return dict(scores)


def three_term_ranking(layer_debug: Dict[str, object]) -> Dict[int, float]:
    scores = defaultdict(float)
    router_scores = layer_debug["router_scores"]
    router_indices = layer_debug["router_indices"]
    before = layer_debug["before"]
    after = layer_debug["after"]
    shift = 1.0 - (before * after).sum(dim=-1) / (
        torch.linalg.norm(before, dim=-1) * torch.linalg.norm(after, dim=-1) + 1e-8
    )
    for token_i in range(router_indices.shape[0]):
        for topk_i in range(router_indices.shape[1]):
            expert_id = int(router_indices[token_i, topk_i].item())
            gate = float(router_scores[token_i, topk_i].item())
            expert_out = before[token_i]  # placeholder local signal for demonstration
            norm = float(torch.linalg.norm(expert_out).item())
            s_t = float(shift[token_i].item())
            scores[expert_id] += gate * max(norm, 1e-8) * max(s_t, 1e-8)
    return dict(scores)


def compute_domain_scores(model: MiniMoEModel, domain: str, top_k: int = 3) -> Dict[str, List[tuple[int, float]]]:
    questions = DOMAIN_TEMPLATES[domain.lower()][:5]
    baseline_accum = defaultdict(float)
    three_term_accum = defaultdict(float)
    for question in questions:
        x = encode_question(question)
        debug_layers = model(x)
        for layer_debug in debug_layers:
            base = baseline_ranking(layer_debug)
            real = three_term_ranking(layer_debug)
            for expert_id, score in base.items():
                baseline_accum[expert_id] += score
            for expert_id, score in real.items():
                three_term_accum[expert_id] += score

    baseline_top = sorted(baseline_accum.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    real_top = sorted(three_term_accum.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    overlap = sorted(set(e for e, _ in baseline_top) & set(e for e, _ in real_top))
    return {
        "questions": questions,
        "baseline": baseline_top,
        "three_term": real_top,
        "overlap": overlap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP MoE scoring demo")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Optional real HF model path; empty means synthetic dry-run")
    args = parser.parse_args()

    print("DRY RUN: synthetic model only; scores are not meaningful until a real HF checkpoint is supplied.")
    print(f"MODEL_PATH configured as: {args.model_path or '<synthetic>'}")

    if args.model_path:
        print(f"Ready to load real checkpoint: {args.model_path}")
        # Real-model branch intentionally left as a clean override point for later integration.

    torch.manual_seed(7)
    model = MiniMoEModel(dim=8, num_layers=3, num_experts=6)
    for domain in ["finance", "biology", "software"]:
        result = compute_domain_scores(model, domain)
        print(f"\n=== Domain: {domain} ===")
        print(f"Questions: {result['questions']}")
        print(f"Baseline top-3: {result['baseline']}")
        print(f"Three-term top-3: {result['three_term']}")
        print(f"Top-K overlap: {result['overlap']}")

    # Final stage: benchmark (when real model is integrated, this will compare parent vs. daughter)
    print("\n=== Final Stage: Benchmark ===\n")
    if args.model_path:
        print("Benchmark would run here: parent vs. daughter model comparison on held-out questions.")
        print("(Deferred until real HF checkpoint integration)")
    else:
        print("Benchmark skipped for synthetic dry-run.")
        print("When you supply --model-path <real-checkpoint>, benchmark will compare parent vs. pruned model.")


if __name__ == "__main__":
    main()

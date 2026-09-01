from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


def infer_layer_index_from_path(path: str) -> Optional[int]:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", path)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:^|\.)layer\.(\d+)(?:\.|$)", path)
    if match:
        return int(match.group(1))
    return None


def generate_moe_adapter(model) -> Dict[str, Any]:
    """Inspect the runtime model structure to build a per-run MoE adapter."""
    router_candidates: List[str] = []
    expert_candidates: List[str] = []
    shared_candidates: List[str] = []
    layer_map: Dict[int, Dict[str, Any]] = {}

    for name, module in model.named_modules():
        lower = name.lower()
        is_router_like = any(token in lower for token in ["gate", "router"]) and "shared" not in lower
        is_expert_like = "expert" in lower and not any(token in lower for token in ["shared", "shared_expert"])
        is_shared_expert_like = "shared" in lower and "expert" in lower

        if is_router_like:
            router_candidates.append(name)
        if is_expert_like:
            expert_candidates.append(name)
        if is_shared_expert_like:
            shared_candidates.append(name)

        if hasattr(module, "num_experts") or hasattr(module, "gate_up_proj") or hasattr(module, "down_proj"):
            if "expert" in lower or "moe" in lower:
                expert_candidates.append(name)

    if not router_candidates:
        raise ValueError("MoE router detection failed: no router/gate modules were found in the loaded model tree.")
    if not expert_candidates:
        raise ValueError("MoE expert detection failed: no expert modules were found in the loaded model tree.")

    for path in sorted(set(router_candidates + expert_candidates + shared_candidates)):
        layer_idx = infer_layer_index_from_path(path)
        if layer_idx is None:
            continue
        info = layer_map.setdefault(layer_idx, {"router_paths": [], "expert_paths": [], "shared_expert_paths": []})
        if path in router_candidates:
            info["router_paths"].append(path)
        if path in expert_candidates:
            info["expert_paths"].append(path)
        if path in shared_candidates:
            info["shared_expert_paths"].append(path)

    if not layer_map:
        raise ValueError(
            "MoE adapter detection is ambiguous: router and expert modules were found, but no layer index could be inferred from their paths."
        )

    ambiguous_layers = []
    for layer_idx, info in sorted(layer_map.items()):
        if not info["router_paths"] or not info["expert_paths"]:
            ambiguous_layers.append(layer_idx)
    if ambiguous_layers:
        raise ValueError(
            "MoE adapter detection is ambiguous: some layers have router paths without expert paths or vice versa. "
            f"Ambiguous layer ids: {ambiguous_layers}. Details: {json.dumps(layer_map, default=str)}"
        )

    return {
        "router_candidates": sorted(router_candidates),
        "expert_candidates": sorted(expert_candidates),
        "shared_expert_candidates": sorted(shared_candidates),
        "layer_map": {
            str(layer_idx): {
                "router_paths": sorted(info["router_paths"]),
                "expert_paths": sorted(info["expert_paths"]),
                "shared_expert_paths": sorted(info["shared_expert_paths"]),
            }
            for layer_idx, info in sorted(layer_map.items())
        },
        "router_path_pattern": "*gate* or *router*",
        "expert_path_pattern": "*expert* and/or modules with num_experts / gate_up_proj / down_proj",
        "shared_expert_policy": "shared experts are never pruned; they are excluded from keep/discard selection",
    }


def print_moe_adapter(adapter: Dict[str, Any]) -> None:
    print("\n=== MoE adapter auto-detection ===")
    print(f"Router path pattern: {adapter['router_path_pattern']}")
    print(f"Expert path pattern: {adapter['expert_path_pattern']}")
    print(f"Shared-expert policy: {adapter['shared_expert_policy']}")
    print("Router candidates:")
    for item in adapter["router_candidates"][:20]:
        print(f"  - {item}")
    if not adapter["router_candidates"]:
        print("  - none found")
    print("Expert candidates:")
    for item in adapter["expert_candidates"][:20]:
        print(f"  - {item}")
    if not adapter["expert_candidates"]:
        print("  - none found")
    print("Shared experts found:")
    if adapter["shared_expert_candidates"]:
        for item in adapter["shared_expert_candidates"][:20]:
            print(f"  - {item}")
    else:
        print("  - none")
    print("Layer breakdown:")
    for layer_idx, info in sorted(adapter["layer_map"].items(), key=lambda kv: int(kv[0])):
        print(f"  Layer {layer_idx}:")
        print(f"    routers: {info['router_paths'][:5]}" + (" ..." if len(info['router_paths']) > 5 else ""))
        print(f"    experts: {info['expert_paths'][:5]}" + (" ..." if len(info['expert_paths']) > 5 else ""))
        print(f"    shared_experts: {info['shared_expert_paths'][:5]}" + (" ..." if len(info['shared_expert_paths']) > 5 else ""))

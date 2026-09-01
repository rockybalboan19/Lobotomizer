from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_text(s: str) -> str:
    return " ".join(s.lower().replace("\n", " ").split())


def similarity_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def is_similar(a: str, b: str, threshold: float = 0.82) -> bool:
    return similarity_ratio(a, b) >= threshold


def dedupe_questions(items: Sequence[str], threshold: float = 0.82) -> List[str]:
    kept: List[str] = []
    for item in items:
        cleaned = item.strip()
        if not cleaned:
            continue
        if any(is_similar(cleaned, existing, threshold) for existing in kept):
            continue
        kept.append(cleaned)
    return kept


def parse_question_lines(raw_text: str) -> List[str]:
    lines = []
    for line in raw_text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = cleaned.strip("-\t *#0123456789.:\"'` ")
        if len(cleaned) > 12:
            lines.append(cleaned)
    return lines


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def try_load_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 180, temperature: float = 0.8, top_p: float = 0.95) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()

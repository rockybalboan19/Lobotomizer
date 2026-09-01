from __future__ import annotations

import os
import random
import re
import subprocess
import sys
import tempfile
import textwrap
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .utils import generate_text


def answer_question(model, tokenizer, question: str) -> str:
    prompt = f"Answer the following question as accurately as possible.\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def extract_multiple_choice_answer(response: str) -> Optional[str]:
    text = response.strip().upper()
    for pattern in [r"\bANSWER\s*[:\-]?\s*([A-D])\b", r"\b([A-D])\b"]:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def looks_like_code_question(question: str) -> bool:
    q = question.lower()
    markers = ["write code", "python", "function", "implement", "script", "code", "debug", "programming", "return a function"]
    return any(marker in q for marker in markers)


def extract_python_block(text: str) -> Optional[str]:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[0].strip()
    return None


def run_python_in_subprocess(code: str, timeout_sec: float = 5.0) -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Execution timed out"
    except Exception as e:
        return False, f"Execution error: {str(e)}"


def benchmark_heldout_set(
    parent_model, parent_tokenizer, daughter_model, daughter_tokenizer, heldout_questions: Sequence[str]
) -> Dict[str, Any]:
    parent_correct = 0
    daughter_correct = 0
    parent_regressions = []
    daughter_improvements = []

    for question in heldout_questions:
        parent_answer = answer_question(parent_model, parent_tokenizer, question)
        daughter_answer = answer_question(daughter_model, daughter_tokenizer, question)

        if looks_like_code_question(question):
            parent_code = extract_python_block(parent_answer)
            daughter_code = extract_python_block(daughter_answer)

            parent_ok, _ = run_python_in_subprocess(parent_code if parent_code else "pass")
            daughter_ok, _ = run_python_in_subprocess(daughter_code if daughter_code else "pass")

            if parent_ok:
                parent_correct += 1
            if daughter_ok:
                daughter_correct += 1

            if parent_ok and not daughter_ok:
                parent_regressions.append((question, parent_answer, daughter_answer))
            elif not parent_ok and daughter_ok:
                daughter_improvements.append((question, parent_answer, daughter_answer))
        else:
            parent_choice = extract_multiple_choice_answer(parent_answer)
            daughter_choice = extract_multiple_choice_answer(daughter_answer)
            if parent_choice:
                parent_correct += 1
            if daughter_choice:
                daughter_correct += 1

            if parent_choice and not daughter_choice:
                parent_regressions.append((question, parent_answer, daughter_answer))
            elif not parent_choice and daughter_choice:
                daughter_improvements.append((question, parent_answer, daughter_answer))

    return {
        "parent_correct": parent_correct,
        "daughter_correct": daughter_correct,
        "total_questions": len(heldout_questions),
        "parent_regressions": parent_regressions,
        "daughter_improvements": daughter_improvements,
    }


def print_benchmark_summary(summary: Dict[str, Any]) -> None:
    print("\n=== Benchmark Summary ===")
    print(f"Parent correct: {summary['parent_correct']} / {summary['total_questions']}")
    print(f"Daughter correct: {summary['daughter_correct']} / {summary['total_questions']}")
    print(f"Daughter gain: {summary['daughter_correct'] - summary['parent_correct']:+d}")

    if summary["parent_regressions"]:
        print(f"\nParent regressions (daughter could not answer): {len(summary['parent_regressions'])}")
        for i, (q, parent_ans, daughter_ans) in enumerate(summary["parent_regressions"][:3], start=1):
            print(f"  {i}. {q[:60]}...")
            print(f"     Parent: {parent_ans[:80]}")
            print(f"     Daughter: {daughter_ans[:80]}")

    if summary["daughter_improvements"]:
        print(f"\nDaughter improvements (parent could not answer): {len(summary['daughter_improvements'])}")
        for i, (q, parent_ans, daughter_ans) in enumerate(summary["daughter_improvements"][:3], start=1):
            print(f"  {i}. {q[:60]}...")
            print(f"     Parent: {parent_ans[:80]}")
            print(f"     Daughter: {daughter_ans[:80]}")

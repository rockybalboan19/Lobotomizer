from __future__ import annotations

import textwrap
from typing import List, Sequence

from .utils import dedupe_questions, generate_text, is_similar, parse_question_lines


def make_self_probe_prompt(domain: str, target_count: int, heldout: bool = False) -> str:
    if heldout:
        return textwrap.dedent(f"""
            You are generating a held-out evaluation set for a benchmark.
            Domain: {domain}
            Generate exactly {target_count} short, distinct, domain-specific questions.
            Make them clearly different from the calibration set and cover different subareas.
            Output only the list, one question per line.
        """).strip()
    return textwrap.dedent(f"""
        You are generating a calibration set of domain-specific knowledge probes.
        Domain: {domain}
        Generate exactly {target_count} short, concrete, high-signal questions.
        Cover breadth and edge cases without repeating similar wording.
        Output only the list, one question per line.
    """).strip()


def generate_question_set(model, tokenizer, domain: str, desired_count: int, heldout: bool = False) -> List[str]:
    prompt = make_self_probe_prompt(domain, desired_count, heldout=heldout)
    raw = generate_text(model, tokenizer, prompt, max_new_tokens=380, temperature=0.9)
    questions = dedupe_questions(parse_question_lines(raw), threshold=0.85)
    if not questions:
        raise RuntimeError("Model generated no valid domain questions.")
    return questions[:desired_count]


def generate_calibration_questions(model, tokenizer, domain: str, desired_count: int = 150) -> List[str]:
    return generate_question_set(model, tokenizer, domain, desired_count, heldout=False)


def generate_heldout_questions(model, tokenizer, domain: str, calibration_questions: Sequence[str], desired_count: int = 30) -> List[str]:
    candidates = generate_question_set(model, tokenizer, domain, desired_count, heldout=True)
    filtered = []
    for q in dedupe_questions(candidates, threshold=0.82):
        if all(not is_similar(q, c, 0.82) for c in calibration_questions):
            filtered.append(q)
    if len(filtered) < desired_count:
        extra = generate_question_set(model, tokenizer, domain, desired_count, heldout=True)
        for q in extra:
            if all(not is_similar(q, c, 0.82) for c in calibration_questions):
                filtered.append(q)
    return dedupe_questions(filtered, threshold=0.82)[:desired_count]


def build_probe_records(model, tokenizer, questions: Sequence[str], subarea: str = "calibration"):
    """Generate a real answer for each probe and keep the pair as calibration metadata."""
    records = []
    for idx, question in enumerate(questions):
        prompt = f"Answer the following question as accurately as possible.\nQuestion: {question}\nAnswer:"
        answer = generate_text(model, tokenizer, prompt, max_new_tokens=180, temperature=0.7)
        records.append({"id": idx, "question": question, "answer": answer, "subarea": subarea})
    return records

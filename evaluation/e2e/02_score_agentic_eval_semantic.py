from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field


load_dotenv()

SCORER_VERSION = "agentic_eval_semantic_v2"

SHAPE_FILES: dict[str, tuple[str, str]] = {
    "single_focused": ("single_focused_raw.jsonl", "single_focused_result.jsonl"),
    "broad_coverage": ("broad_coverage_raw.jsonl", "broad_coverage_result.jsonl"),
    "multi_part": ("multi_part_raw.jsonl", "multi_part_result.jsonl"),
    "comparison": ("comparison_raw.jsonl", "comparison_result.jsonl"),
    "context_dependent": ("context_dependent_raw.jsonl", "context_dependent_result.jsonl"),
}

NORMAL_MODEL_CALLS: dict[str, int] = {
    "single_focused": 2,
    "broad_coverage": 2,
    "multi_part": 4,
    "comparison": 4,
    "context_dependent": 3,
}

EXPECTED_ROUTES: dict[str, str] = {
    "single_focused": "single_fast",
    "broad_coverage": "broad_adaptive",
    "multi_part": "multi_batch",
    "comparison": "comparison_batch",
}

EXPECTED_NORMAL_TERMINATIONS: dict[str, str] = {
    "single_focused": "AUTO_APPROVED",
    "broad_coverage": "AUTO_APPROVED",
    "multi_part": "APPROVED_REVIEW",
    "comparison": "APPROVED_REVIEW",
    "context_dependent": "AUTO_APPROVED",
}


class SemanticClaimJudgment(BaseModel):
    claim_id: str
    answer_status: Literal["correct", "partial", "missing", "incorrect"]
    retrieval_supported: bool
    citation_supported: bool
    rationale: str = ""


class SemanticCaseJudgment(BaseModel):
    claims: list[SemanticClaimJudgment] = Field(default_factory=list)
    unsupported_material_claims: list[str] = Field(default_factory=list)


JUDGE_SYSTEM_PROMPT = """
You are evaluating a California DMV RAG system using SEMANTIC evidence, not
chunk-ID equality.

The evaluator gives you:
- the user question,
- a reference answer,
- required claims and their gold supporting spans,
- ACTUAL retrieved passages from the live corpus,
- ACTUAL citation excerpts emitted with the answer,
- the assistant's RAG answer.

IMPORTANT RUBRIC
1) Never require exact wording or exact chunk identity. Paraphrases and an
   equivalent passage count as support.
2) Judge the CENTRAL MATERIAL MEANING of each required claim.
   - correct: the answer clearly conveys the claim's central rule/fact and does
     not materially change it. Do NOT downgrade merely for omitting examples,
     optional elaboration, redundant detail, or wording that is not necessary
     to preserve the rule's meaning.
   - partial: the central idea is present, but a MATERIAL condition/exception/
     qualifier stated in the required claim is missing or weakened enough to
     change when/how the rule applies.
   - missing: the claim is not meaningfully addressed.
   - incorrect: the answer materially contradicts or misstates the claim.
3) retrieval_supported=true when at least one ACTUAL retrieved passage
   semantically supports the required claim or a materially equivalent rule.
   The live passage may have a different chunk ID, heading granularity, or
   wording from the gold supporting span.
4) citation_supported=true when at least one ACTUAL citation excerpt
   semantically supports the required claim. Exact source IDs are irrelevant.
5) unsupported_material_claims should contain only MATERIAL factual claims in
   the assistant answer that are unsupported by ALL available evidence:
   - gold supporting spans/reference answer,
   - actual retrieved passages,
   - actual citation excerpts,
   or that contradict that evidence.
   Do NOT flag harmless explanation, conversational filler, reasonable
   paraphrases, or extra facts that are actually supported by live retrieved or
   cited evidence merely because they are absent from the synthetic gold subset.
6) Use only the supplied evidence. Do not use outside knowledge.

Return every required claim exactly once and preserve the supplied claim_id.
Be concise and consistent.
""".strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path} line {line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def load_by_id(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        item_id = str(row.get("id", "")).strip()
        if not item_id:
            raise ValueError(f"Row without id in {path}")
        if item_id in out:
            raise ValueError(f"Duplicate id {item_id} in {path}")
        out[item_id] = row
    return out


def unique_strings(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def normalize_retrieval_groups(result_row: dict[str, Any]) -> list[dict[str, Any]]:
    raw_groups = result_row.get("retrieval_groups", [])
    if not isinstance(raw_groups, list):
        return []
    groups: list[dict[str, Any]] = []
    for raw in raw_groups:
        if not isinstance(raw, dict):
            continue
        groups.append(
            {
                "sub_question_id": str(raw.get("sub_question_id", "")).strip(),
                "query": str(raw.get("query", "")).strip(),
                "chunk_ids": unique_strings(raw.get("chunk_ids", [])),
            }
        )
    return groups


def flatten_top_k(groups: list[dict[str, Any]], k: int) -> list[str]:
    flattened: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for chunk_id in group.get("chunk_ids", [])[:k]:
            chunk_id = str(chunk_id).strip()
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                flattened.append(chunk_id)
    return flattened


def load_live_corpus(path: Path) -> dict[str, dict[str, str]]:
    corpus: dict[str, dict[str, str]] = {}
    for row in read_jsonl(path):
        chunk_id = str(row.get("stable_chunk_id", row.get("chunk_id", ""))).strip()
        if not chunk_id:
            continue
        corpus[chunk_id] = {
            "text": str(row.get("text", "")).strip(),
            "heading_path": str(row.get("heading_path", "")).strip(),
            "section_id": str(row.get("section_id", "")).strip(),
            "file_name": str(row.get("file_name", "")).strip(),
        }
    return corpus


def citations_from_result(result_row: dict[str, Any]) -> list[dict[str, str]]:
    raw = result_row.get("citations", [])
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for citation in raw:
        if not isinstance(citation, dict):
            continue
        out.append(
            {
                "source_id": str(citation.get("source_id", "")).strip(),
                "heading_path": str(citation.get("heading_path", "")).strip(),
                "section_id": str(citation.get("section_id", "")).strip(),
                "excerpt": str(citation.get("excerpt", "")).strip(),
            }
        )
    return out


def build_actual_retrieved_passages(
    result_row: dict[str, Any],
    corpus: dict[str, dict[str, str]],
    top_k: int,
    max_chars_per_passage: int,
) -> tuple[list[dict[str, Any]], float]:
    groups = normalize_retrieval_groups(result_row)
    citations = citations_from_result(result_row)
    citation_by_id = {
        c["source_id"]: c for c in citations if c.get("source_id")
    }

    passages: list[dict[str, Any]] = []
    seen: set[str] = set()
    requested = 0
    resolved = 0

    for group in groups:
        group_id = group.get("sub_question_id", "")
        query = group.get("query", "")
        for rank, chunk_id in enumerate(group.get("chunk_ids", [])[:top_k], start=1):
            requested += 1
            if chunk_id in seen:
                if chunk_id in corpus or citation_by_id.get(chunk_id, {}).get("excerpt"):
                    resolved += 1
                continue
            seen.add(chunk_id)

            corpus_row = corpus.get(chunk_id, {})
            text = str(corpus_row.get("text", "")).strip()
            heading = str(corpus_row.get("heading_path", "")).strip()
            section = str(corpus_row.get("section_id", "")).strip()
            source = "live_corpus"

            if not text:
                citation = citation_by_id.get(chunk_id, {})
                text = str(citation.get("excerpt", "")).strip()
                heading = heading or str(citation.get("heading_path", "")).strip()
                section = section or str(citation.get("section_id", "")).strip()
                source = "citation_fallback" if text else "unresolved"

            if text:
                resolved += 1

            passages.append(
                {
                    "group_id": group_id,
                    "group_query": query,
                    "rank_in_group": rank,
                    "chunk_id": chunk_id, 
                    "heading_path": heading,
                    "section_id": section,
                    "text": text[:max_chars_per_passage],
                    "text_source": source,
                }
            )

    resolution_rate = (resolved / requested) if requested else 1.0
    return passages, resolution_rate


def build_actual_citation_passages(
    result_row: dict[str, Any],
    corpus: dict[str, dict[str, str]],
    max_chars_per_passage: int,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for citation in citations_from_result(result_row):
        source_id = citation.get("source_id", "")
        corpus_row = corpus.get(source_id, {})
        excerpt = citation.get("excerpt", "") or str(corpus_row.get("text", ""))
        out.append(
            {
                "source_id": source_id,
                "heading_path": citation.get("heading_path", "") or str(corpus_row.get("heading_path", "")),
                "section_id": citation.get("section_id", "") or str(corpus_row.get("section_id", "")),
                "excerpt": str(excerpt).strip()[:max_chars_per_passage],
            }
        )
    return out


def binary_match(expected: Any, actual: Any) -> float | None:
    e = str(expected or "").strip().lower()
    a = str(actual or "").strip().lower()
    if not e or not a:
        return None
    return float(e == a)


def id_recall_diagnostic(dataset_row: dict[str, Any], result_row: dict[str, Any], top_k: int) -> float:
    expectations = dataset_row.get("expectations", {}) or {}
    gold = set(unique_strings(expectations.get("gold_chunk_ids", [])))
    if not gold:
        return 0.0
    retrieved = set(flatten_top_k(normalize_retrieval_groups(result_row), top_k))
    return len(gold & retrieved) / len(gold)


def build_deterministic_metrics(
    dataset_row: dict[str, Any],
    result_row: dict[str, Any],
    top_k: int,
    retrieval_text_resolution_rate: float,
) -> dict[str, float | int | None]:
    expectations = dataset_row.get("expectations", {}) or {}
    expected_shape = str(expectations.get("question_shape", "")).strip()
    expected_mode = str(expectations.get("expected_retrieval_mode", "")).strip()
    meta = result_row.get("rag_meta", {}) or {}
    groups = normalize_retrieval_groups(result_row)

    shape_match = None if expected_shape == "context_dependent" else binary_match(
        expected_shape, meta.get("actual_question_shape")
    )
    expected_route = EXPECTED_ROUTES.get(expected_shape)
    model_calls = int(meta.get("model_calls_used", 0) or 0)
    retrieval_rounds = int(meta.get("retrieval_rounds_used", 0) or 0)
    normal_calls = NORMAL_MODEL_CALLS.get(expected_shape)

    return {
        "shape_match": shape_match,
        "route_match": binary_match(expected_route, meta.get("actual_route")) if expected_route else None,
        "retrieval_mode_match": binary_match(expected_mode, meta.get("actual_retrieval_mode")),
        "context_history_seeded": (
            float(int(meta.get("eval_seeded_history_pairs", 0) or 0) > 0)
            if expected_shape == "context_dependent"
            else None
        ),
        "retrieval_group_count": len(groups),
        "retrieval_group_count_match": float(
            len(groups) >= 2 if expected_mode == "decomposed" else len(groups) == 1
        ),
        "model_calls_used": model_calls,
        "retrieval_rounds_used": retrieval_rounds,
        "model_call_overhead": max(0, model_calls - normal_calls) if normal_calls is not None else None,
        "retrieval_round_overhead": max(0, retrieval_rounds - 1),
        "normal_termination_match": binary_match(
            EXPECTED_NORMAL_TERMINATIONS.get(expected_shape), meta.get("termination_reason")
        ),
        "budget_exhausted": float(bool(meta.get("budget_exhausted", False))),
        "insufficient_evidence": float(bool(meta.get("insufficient_evidence", False))),
        "blocked": float(bool(meta.get("blocked", False))),
        "request_error": float(bool(meta.get("error", False))),
        "latency_ms": float(result_row.get("latency_ms", 0.0) or 0.0),
        "retrieval_text_resolution_rate": retrieval_text_resolution_rate,
        f"id_exact_recall_diagnostic@{top_k}": id_recall_diagnostic(dataset_row, result_row, top_k),
    }


def build_judge_messages(
    dataset_row: dict[str, Any],
    result_row: dict[str, Any],
    retrieved_passages: list[dict[str, Any]],
    citation_passages: list[dict[str, str]],
) -> list[dict[str, str]]:
    expectations = dataset_row.get("expectations", {}) or {}
    required_claims: list[dict[str, Any]] = []
    for claim in expectations.get("required_claims", []):
        if not isinstance(claim, dict):
            continue
        required_claims.append(
            {
                "claim_id": str(claim.get("claim_id", "")).strip(),
                "claim": str(claim.get("claim", "")).strip(),
                "supporting_spans": unique_strings(claim.get("supporting_spans", [])),
                "coverage_part": str(claim.get("coverage_part", "")).strip() or None,
                "comparison_side": str(claim.get("comparison_side", "")).strip() or None,
            }
        )

    payload = {
        "user_question": (dataset_row.get("inputs", {}) or {}).get("user_query", ""),
        "reference_answer": expectations.get("reference_answer", ""),
        "required_claims": required_claims,
        "actual_retrieved_passages": retrieved_passages,
        "actual_citation_passages": citation_passages,
        "assistant_answer": result_row.get("rag_answer", ""),
    }
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def normalize_judgment(
    dataset_row: dict[str, Any], judgment: SemanticCaseJudgment
) -> SemanticCaseJudgment:
    expectations = dataset_row.get("expectations", {}) or {}
    expected_ids = [
        str(claim.get("claim_id", "")).strip()
        for claim in expectations.get("required_claims", [])
        if isinstance(claim, dict) and str(claim.get("claim_id", "")).strip()
    ]
    by_id: dict[str, SemanticClaimJudgment] = {}
    for claim in judgment.claims:
        claim_id = claim.claim_id.strip()
        if claim_id in expected_ids and claim_id not in by_id:
            by_id[claim_id] = claim

    claims = [
        by_id.get(
            claim_id,
            SemanticClaimJudgment(
                claim_id=claim_id,
                answer_status="missing",
                retrieval_supported=False,
                citation_supported=False,
                rationale="Judge omitted this required claim.",
            ),
        )
        for claim_id in expected_ids
    ]
    return SemanticCaseJudgment(
        claims=claims,
        unsupported_material_claims=[
            str(x).strip() for x in judgment.unsupported_material_claims if str(x).strip()
        ],
    )


async def judge_case(
    *,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    dataset_row: dict[str, Any],
    result_row: dict[str, Any],
    retrieved_passages: list[dict[str, Any]],
    citation_passages: list[dict[str, str]],
    model: str,
) -> SemanticCaseJudgment:
    async with semaphore:
        completion = await client.beta.chat.completions.parse(
            model=model,
            messages=build_judge_messages(
                dataset_row, result_row, retrieved_passages, citation_passages
            ),
            response_format=SemanticCaseJudgment,
        )
        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise RuntimeError(f"Judge refused: {message.refusal}")
        parsed = message.parsed
        if parsed is None:
            raise RuntimeError("Judge returned empty parsed output")
        return normalize_judgment(dataset_row, parsed)


def metrics_from_judgment(judgment: SemanticCaseJudgment) -> dict[str, float | int]:
    claims = judgment.claims
    total = len(claims)
    if total == 0:
        return {
            "semantic_claim_retrieval_recall": 0.0,
            "semantic_claim_citation_recall": 0.0,
            "claim_answer_correctness": 0.0,
            "claim_answer_completeness": 0.0,
            "fully_correct_claim_rate": 0.0,
            "incorrect_claim_rate": 0.0,
            "correct_claim_grounded_rate": 0.0,
            "unsupported_material_claim_count": len(judgment.unsupported_material_claims),
            "unsupported_material_claim_pass": float(not judgment.unsupported_material_claims),
            "semantic_quality_pass": 0.0,
        }

    status_counts = Counter(c.answer_status for c in claims)
    weighted_correct = status_counts["correct"] + 0.5 * status_counts["partial"]
    completeness = (total - status_counts["missing"]) / total
    retrieval_recall = sum(1 for c in claims if c.retrieval_supported) / total
    citation_recall = sum(1 for c in claims if c.citation_supported) / total

    correct_claims = [c for c in claims if c.answer_status == "correct"]
    correct_grounded = (
        sum(1 for c in correct_claims if c.citation_supported) / len(correct_claims)
        if correct_claims
        else 0.0
    )

    unsupported_count = len(judgment.unsupported_material_claims)
    correctness = weighted_correct / total
    incorrect_rate = status_counts["incorrect"] / total
    
    semantic_quality_pass = float(
        correctness >= 0.75
        and completeness >= 0.80
        and incorrect_rate == 0.0
        and unsupported_count == 0
    )

    return {
        "semantic_claim_retrieval_recall": retrieval_recall,
        "semantic_claim_citation_recall": citation_recall,
        "claim_answer_correctness": correctness,
        "claim_answer_completeness": completeness,
        "fully_correct_claim_rate": status_counts["correct"] / total,
        "incorrect_claim_rate": incorrect_rate,
        "correct_claim_grounded_rate": correct_grounded,
        "unsupported_material_claim_count": unsupported_count,
        "unsupported_material_claim_pass": float(unsupported_count == 0),
        "semantic_quality_pass": semantic_quality_pass,
    }


def metric_average(rows: list[dict[str, Any]], metric_name: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = (row.get("metrics", {}) or {}).get(metric_name)
        if isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    rank = max(1, math.ceil(p * len(values)))
    return round(values[min(rank - 1, len(values) - 1)], 2)


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "avg": round(statistics.fmean(values), 2),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names: set[str] = set()
    for row in rows:
        metrics = row.get("metrics", {}) or {}
        if isinstance(metrics, dict):
            metric_names.update(metrics.keys())

    excluded = {
        "latency_ms",
        "model_calls_used",
        "retrieval_rounds_used",
        "retrieval_group_count",
        "unsupported_material_claim_count",
    }
    quality = {
        name: metric_average(rows, name)
        for name in sorted(metric_names)
        if name not in excluded
    }

    def values(name: str) -> list[float]:
        out: list[float] = []
        for row in rows:
            value = (row.get("metrics", {}) or {}).get(name)
            if isinstance(value, (int, float)):
                out.append(float(value))
        return out

    judge_errors = sum(1 for row in rows if (row.get("judge", {}) or {}).get("error"))
    return {
        "count": len(rows),
        "quality": quality,
        "runtime": {
            "latency_ms": distribution(values("latency_ms")),
            "model_calls_used": distribution(values("model_calls_used")),
            "retrieval_rounds_used": distribution(values("retrieval_rounds_used")),
            "unsupported_material_claim_count": distribution(values("unsupported_material_claim_count")),
        },
        "judge_errors": judge_errors,
    }


def build_summary(
    rows: list[dict[str, Any]], *, judge_enabled: bool, judge_model: str, top_k: int, corpus_path: str
) -> dict[str, Any]:
    by_shape: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_shape.setdefault(str(row.get("question_shape", "UNKNOWN")), []).append(row)
    return {
        "scorer_version": SCORER_VERSION,
        "judge_enabled": judge_enabled,
        "judge_model": judge_model if judge_enabled else None,
        "semantic_retrieval_top_k_per_group": top_k,
        "live_corpus_path": corpus_path,
        "overall": summarize_rows(rows),
        "by_shape": {shape: summarize_rows(items) for shape, items in sorted(by_shape.items())},
        "notes": {
            "primary_retrieval_metric": (
                "semantic_claim_retrieval_recall: LLM checks whether actual retrieved TEXT semantically supports each required claim. Chunk-ID equality is not used."
            ),
            "primary_citation_metric": (
                "semantic_claim_citation_recall: LLM checks whether actual citation EXCERPTS semantically support each required claim."
            ),
            "id_metric": (
                "id_exact_recall_diagnostic is diagnostic only and may fall when corpus chunk IDs drift."
            ),
            "claim_answer_correctness": "correct=1.0, partial=0.5, missing/incorrect=0.0",
            "semantic_quality_pass": (
                "Pass when claim correctness >= 0.75, completeness >= 0.80, incorrect_claim_rate == 0, and unsupported material claim count == 0. This is a thresholded quality gate, not a perfect-answer metric."
            ),
            "unsupported_definition": (
                "Extra facts are penalized only when unsupported by BOTH gold evidence and actual live retrieved/cited evidence, or when contradictory."
            ),
        },
    }


def fmt_pct(value: Any) -> str:
    return "N/A" if not isinstance(value, (int, float)) else f"{float(value) * 100:.1f}%"


def fmt_num(value: Any, digits: int = 0) -> str:
    return "N/A" if not isinstance(value, (int, float)) else f"{float(value):.{digits}f}"


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Semantic Agentic RAG Evaluation Summary",
        "",
        f"Scorer version: `{summary['scorer_version']}`  ",
        f"LLM judge enabled: `{summary['judge_enabled']}`  ",
        f"Judge model: `{summary.get('judge_model')}`  ",
        f"Semantic retrieval top-k per group: `{summary.get('semantic_retrieval_top_k_per_group')}`  ",
        "",
        "## Per-shape quality",
        "",
        "| Shape | N | Route | Mode | Semantic Retrieval | Semantic Citation | Answer Correct | Complete | Unsupported-pass | Quality-pass | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for shape, item in summary.get("by_shape", {}).items():
        q = item.get("quality", {}) or {}
        lat = (item.get("runtime", {}) or {}).get("latency_ms", {}) or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    shape,
                    str(item.get("count", 0)),
                    fmt_pct(q.get("route_match")),
                    fmt_pct(q.get("retrieval_mode_match")),
                    fmt_pct(q.get("semantic_claim_retrieval_recall")),
                    fmt_pct(q.get("semantic_claim_citation_recall")),
                    fmt_pct(q.get("claim_answer_correctness")),
                    fmt_pct(q.get("claim_answer_completeness")),
                    fmt_pct(q.get("unsupported_material_claim_pass")),
                    fmt_pct(q.get("semantic_quality_pass")),
                    fmt_num(lat.get("p50")),
                    fmt_num(lat.get("p95")),
                ]
            )
            + " |"
        )

    overall = summary.get("overall", {}) or {}
    q = overall.get("quality", {}) or {}
    runtime = overall.get("runtime", {}) or {}
    lat = runtime.get("latency_ms", {}) or {}
    lines += [
        "",
        "## Overall",
        "",
        f"- Cases: **{overall.get('count', 0)}**",
        f"- Semantic claim retrieval recall: **{fmt_pct(q.get('semantic_claim_retrieval_recall'))}**",
        f"- Semantic claim citation recall: **{fmt_pct(q.get('semantic_claim_citation_recall'))}**",
        f"- Claim answer correctness: **{fmt_pct(q.get('claim_answer_correctness'))}**",
        f"- Claim answer completeness: **{fmt_pct(q.get('claim_answer_completeness'))}**",
        f"- Unsupported-material pass: **{fmt_pct(q.get('unsupported_material_claim_pass'))}**",
        f"- Semantic quality pass: **{fmt_pct(q.get('semantic_quality_pass'))}**",
        f"- Retrieval text resolution: **{fmt_pct(q.get('retrieval_text_resolution_rate'))}**",
        f"- ID exact recall diagnostic: **{fmt_pct(next((v for k, v in q.items() if k.startswith('id_exact_recall_diagnostic@')), None))}**",
        f"- Error rate: **{fmt_pct(q.get('request_error'))}**",
        f"- Latency p50: **{fmt_num(lat.get('p50'))} ms**",
        f"- Latency p95: **{fmt_num(lat.get('p95'))} ms**",
        f"- Judge errors: **{overall.get('judge_errors', 0)}**",
        "",
        "## Interpretation",
        "",
        "Primary quality uses semantic support from actual retrieved/cited text. Exact chunk-ID matching is retained only as a drift diagnostic.",
        "",
    ]
    return "\n".join(lines)


async def score_case(
    *,
    shape: str,
    dataset_row: dict[str, Any],
    result_row: dict[str, Any],
    corpus: dict[str, dict[str, str]],
    top_k: int,
    max_chars_per_passage: int,
    judge_enabled: bool,
    client: AsyncOpenAI | None,
    semaphore: asyncio.Semaphore | None,
    judge_model: str,
) -> dict[str, Any]:
    expectations = dataset_row.get("expectations", {}) or {}
    retrieved_passages, resolution_rate = build_actual_retrieved_passages(
        result_row, corpus, top_k, max_chars_per_passage
    )
    citation_passages = build_actual_citation_passages(
        result_row, corpus, max_chars_per_passage
    )

    metrics = build_deterministic_metrics(
        dataset_row, result_row, top_k, resolution_rate
    )

    judge_payload: dict[str, Any] = {
        "enabled": judge_enabled,
        "model": judge_model if judge_enabled else None,
        "error": False,
    }

    if judge_enabled:
        if client is None or semaphore is None:
            raise RuntimeError("Judge client not initialized")
        try:
            judgment = await judge_case(
                client=client,
                semaphore=semaphore,
                dataset_row=dataset_row,
                result_row=result_row,
                retrieved_passages=retrieved_passages,
                citation_passages=citation_passages,
                model=judge_model,
            )
            judge_payload["judgment"] = judgment.model_dump(mode="json")
            metrics.update(metrics_from_judgment(judgment))
        except Exception as exc:
            judge_payload.update(
                {
                    "error": True,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    meta = result_row.get("rag_meta", {}) or {}
    return {
        "scorer_version": SCORER_VERSION,
        "id": str(dataset_row.get("id", "")),
        "question_shape": shape,
        "query": (dataset_row.get("inputs", {}) or {}).get("user_query", ""),
        "reference_answer": expectations.get("reference_answer", ""),
        "rag_answer": result_row.get("rag_answer", ""),
        "expected": {
            "question_shape": expectations.get("question_shape"),
            "retrieval_mode": expectations.get("expected_retrieval_mode"),
            "required_claims": expectations.get("required_claims", []),
            "gold_chunk_ids_diagnostic_only": expectations.get("gold_chunk_ids", []),
        },
        "observed": {
            "actual_question_shape": meta.get("actual_question_shape"),
            "actual_route": meta.get("actual_route"),
            "actual_retrieval_mode": meta.get("actual_retrieval_mode"),
            "retrieval_groups": normalize_retrieval_groups(result_row),
            "retrieved_passages_used_by_judge": retrieved_passages,
            "citation_passages_used_by_judge": citation_passages,
            "termination_reason": meta.get("termination_reason"),
        },
        "metrics": metrics,
        "judge": judge_payload,
    }


async def run(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    corpus_path = Path(args.corpus_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Live corpus not found: {corpus_path}. Point --corpus-path to the corpus JSONL whose stable_chunk_id values match the live Qdrant ingestion."
        )
    corpus = load_live_corpus(corpus_path)
    if not corpus:
        raise RuntimeError(f"No chunks loaded from {corpus_path}")

    selected_shapes = [args.shape] if args.shape else list(SHAPE_FILES)
    judge_enabled = not args.skip_judge

    if judge_enabled and AsyncOpenAI is None:
        raise SystemExit(
            "The openai package is required for semantic LLM judging. Install evaluation/requirements-eval.txt or use --skip-judge."
        )

    if judge_enabled and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is required unless --skip-judge is used. Put it in evaluation/.env or export it in the shell."
        )

    client: AsyncOpenAI | None = None
    semaphore: asyncio.Semaphore | None = None
    if judge_enabled:
        client = AsyncOpenAI(timeout=args.judge_timeout_sec, max_retries=2)
        semaphore = asyncio.Semaphore(args.workers)

    all_scored: list[dict[str, Any]] = []
    for shape in selected_shapes:
        dataset_name, result_name = SHAPE_FILES[shape]
        dataset_path = dataset_dir / dataset_name
        result_path = results_dir / result_name
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)
        if not result_path.exists():
            raise FileNotFoundError(result_path)

        dataset_by_id = load_by_id(dataset_path)
        result_by_id = load_by_id(result_path)
        missing = sorted(set(dataset_by_id) - set(result_by_id))
        if missing:
            raise RuntimeError(
                f"{shape}: {len(missing)} dataset cases have no live result. First: {missing[0]}"
            )

        tasks = [
            score_case(
                shape=shape,
                dataset_row=dataset_by_id[item_id],
                result_row=result_by_id[item_id],
                corpus=corpus,
                top_k=args.top_k,
                max_chars_per_passage=args.max_chars_per_passage,
                judge_enabled=judge_enabled,
                client=client,
                semaphore=semaphore,
                judge_model=args.judge_model,
            )
            for item_id in dataset_by_id
        ]
        shape_rows = await asyncio.gather(*tasks)
        all_scored.extend(shape_rows)
        shape_out = out_dir / f"{shape}_semantic_scored.jsonl"
        write_jsonl(shape_out, shape_rows)
        print(json.dumps({"shape": shape, "cases": len(shape_rows), "out": str(shape_out)}))

    all_out = out_dir / "all_semantic_scored.jsonl"
    write_jsonl(all_out, all_scored)

    summary = build_summary(
        all_scored,
        judge_enabled=judge_enabled,
        judge_model=args.judge_model,
        top_k=args.top_k,
        corpus_path=str(corpus_path),
    )
    summary_json = out_dir / "summary_semantic.json"
    summary_md = out_dir / "summary_semantic.md"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_md.write_text(render_summary_markdown(summary), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "scorer_version": SCORER_VERSION,
                "cases": len(all_scored),
                "all_scored": str(all_out),
                "summary_json": str(summary_json),
                "summary_md": str(summary_md),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Semantic-text Agentic RAG evaluator. Primary retrieval/citation quality is judged from actual evidence text, not chunk-ID equality."
        )
    )
    parser.add_argument("--dataset-dir", default="evaluation/datasets/dmv_agentic_eval_v2")
    parser.add_argument("--results-dir", default="evaluation/results")
    parser.add_argument("--out-dir", default="evaluation/reports/agentic_eval_semantic")
    parser.add_argument(
        "--corpus-path",
        default="evaluation/datasets/dmv_retrieval_gold_v1/corpus.jsonl",
        help="Live/current corpus JSONL used to resolve retrieved chunk IDs into full text.",
    )
    parser.add_argument("--shape", choices=list(SHAPE_FILES), default=None)
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Max retrieved passages per retrieval group exposed to the semantic judge. Default 8 so broad_coverage can use its full live retrieval budget.",
    )
    parser.add_argument(
        "--max-chars-per-passage",
        type=int,
        default=1800,
        help="Cap each retrieved/cited passage sent to the judge to control token cost.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("EVAL_JUDGE_MODEL", "gpt-5.6-sol"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--judge-timeout-sec", type=float, default=180.0)
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Validate loading/routing/runtime/corpus text resolution without LLM semantic scoring.",
    )

    args = parser.parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")
    if args.max_chars_per_passage < 200:
        raise SystemExit("--max-chars-per-passage must be >= 200")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
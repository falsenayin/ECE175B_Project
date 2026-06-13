#!/usr/bin/env python3
"""Evaluate grounding in batches using a ChatGPT-managed Codex mini model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from codex_subscription import CodexBatchClient, grounding_batch_schema
from evaluate_grounding import (
    build_judge_prompt,
    evidence_cache_key,
    load_articles,
    load_manifest,
    normalize_top_k,
    quote_matches_context,
    result_key,
    supplied_context,
    write_audit_template,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--results-file", default="results_main.csv")
    parser.add_argument(
        "--method-filter",
        help="Only judge rows with this method value, for example structured_retrieval.",
    )
    parser.add_argument(
        "--top-k-filter",
        help="Only judge rows with this normalized top_k value, for example 5.",
    )
    parser.add_argument("--judge-model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--output-subdir", default="grounding_gpt54mini")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-new-batches", type=int)
    parser.add_argument("--audit-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=175)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/QuALITY.v1.0.1.htmlstripped.dev"),
    )
    return parser.parse_args()


def batch_prompt(tasks: list[tuple[str, str]]) -> str:
    parts = [
        "Evaluate each grounding task independently and strictly.",
        "Do not use tools or outside knowledge.",
        "Return one result for every task ID using the required JSON schema.",
        "The supporting quote must be a short contiguous quote copied from that task's supplied context.",
        "An empty quote is allowed for unsupported answers and truly absence-based answers.",
        "",
    ]
    for task_id, prompt in tasks:
        parts.extend([f'<task id="{task_id}">', prompt, "</task>", ""])
    return "\n".join(parts)


def judgment_row(
    result: pd.Series,
    context: str,
    cache_key: str,
    payload: dict,
    judge_model: str,
    reasoning_effort: str,
    latency: float,
    batch_result: object,
) -> dict:
    supported = bool(payload["supported"])
    quote = str(payload["supporting_quote"])
    confidence = str(payload["confidence"])
    reason = str(payload["reason"])
    if supported and quote.strip() and not quote_matches_context(quote, context):
        supported = False
        quote = ""
        confidence = "low"
        reason = "The claimed supporting quote was not a contiguous passage in the supplied context."
    return {
        "result_key": result_key(result),
        "cache_key": cache_key,
        "reused_from_result_key": "",
        "example_id": result["example_id"],
        "article_id": result["article_id"],
        "method": result["method"],
        "top_k": normalize_top_k(result.get("top_k", "")),
        "question": result["question"],
        "prediction": result["prediction"],
        "is_correct": result["is_correct"],
        "judge_supported": int(supported),
        "confidence": confidence,
        "supporting_quote": quote,
        "reason": reason,
        "raw_judge_output": json.dumps(payload, ensure_ascii=True),
        "judge_latency_seconds": round(latency, 6),
        "judge_prompt_tokens": "",
        "judge_output_tokens": "",
        "judge_model": judge_model,
        "supplied_context": context,
        "codex_reasoning_effort": reasoning_effort,
        "codex_batch_input_tokens": batch_result.input_tokens or "",
        "codex_batch_cached_input_tokens": batch_result.cached_input_tokens or "",
        "codex_batch_output_tokens": batch_result.output_tokens or "",
    }


def reused_row(result: pd.Series, cached: dict) -> dict:
    row = dict(cached)
    row.update(
        {
            "result_key": result_key(result),
            "method": result["method"],
            "top_k": normalize_top_k(result.get("top_k", "")),
            "reused_from_result_key": cached["result_key"],
        }
    )
    return row


def process_directory(
    directory: Path,
    args: argparse.Namespace,
    client: CodexBatchClient,
    articles: dict[str, str],
    batch_budget: list[int],
) -> None:
    results_path = directory / args.results_file
    results = pd.read_csv(results_path, keep_default_na=False, dtype={"article_id": str})
    if args.method_filter:
        results = results[results["method"].astype(str) == args.method_filter].copy()
    if args.top_k_filter:
        results = results[
            results["top_k"].map(normalize_top_k) == str(args.top_k_filter)
        ].copy()
    manifest = load_manifest(results_path)
    chunk_size = int(manifest["chunk_size_words"])
    overlap = int(manifest["overlap_words"])
    output_dir = directory / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "grounding_judgments.csv"
    rows = (
        pd.read_csv(output_path, keep_default_na=False).to_dict("records")
        if output_path.exists()
        else []
    )
    by_result = {row["result_key"]: row for row in rows}
    by_cache = {row["cache_key"]: row for row in rows}

    pending: list[tuple[pd.Series, str, str]] = []
    for _, result in results.iterrows():
        key = result_key(result)
        if key in by_result:
            continue
        context = supplied_context(result, articles, chunk_size, overlap)
        cache_key = evidence_cache_key(result, context)
        cached = by_cache.get(cache_key)
        if cached:
            row = reused_row(result, cached)
            rows.append(row)
            by_result[key] = row
            pd.DataFrame(rows).to_csv(output_path, index=False)
            continue
        pending.append((result, context, cache_key))

    for start in range(0, len(pending), args.batch_size):
        if args.max_new_batches is not None and batch_budget[0] >= args.max_new_batches:
            break
        raw_batch = []
        for result, context, cache_key in pending[start : start + args.batch_size]:
            if cache_key in by_cache:
                copied = reused_row(result, by_cache[cache_key])
                rows.append(copied)
                by_result[copied["result_key"]] = copied
            else:
                raw_batch.append((result, context, cache_key))
        if not raw_batch:
            pd.DataFrame(rows).to_csv(output_path, index=False)
            continue
        unique: dict[str, tuple[pd.Series, str, str]] = {}
        duplicates: dict[str, list[pd.Series]] = {}
        for result, context, cache_key in raw_batch:
            if cache_key in unique:
                duplicates.setdefault(cache_key, []).append(result)
            else:
                unique[cache_key] = (result, context, cache_key)
        tasks = [
            (cache_key, build_judge_prompt(result, context))
            for cache_key, (result, context, _) in unique.items()
        ]
        print(
            f"{directory.name}: grounding batch {batch_budget[0] + 1}, "
            f"{len(rows)}/{len(results)} rows complete",
            flush=True,
        )
        response = client.ask(batch_prompt(tasks), grounding_batch_schema())
        payloads = {str(item["id"]): item for item in response.payload["results"]}
        expected = set(unique)
        extra = set(payloads) - expected
        if extra:
            print(
                f"Warning: judge returned {len(extra)} unexpected extra ID(s); "
                "they will be ignored.",
                flush=True,
            )
            payloads = {key: value for key, value in payloads.items() if key in expected}
        missing = expected - set(payloads)
        if missing:
            print(
                f"Warning: judge omitted {len(missing)} task ID(s); "
                "they will be retried later.",
                flush=True,
            )
            pending.extend(unique[cache_key] for cache_key in missing)
        per_item_latency = response.latency_seconds / len(unique)
        for cache_key, (result, context, _) in unique.items():
            if cache_key not in payloads:
                continue
            row = judgment_row(
                result,
                context,
                cache_key,
                payloads[cache_key],
                args.judge_model,
                args.reasoning_effort,
                per_item_latency,
                response,
            )
            rows.append(row)
            by_result[row["result_key"]] = row
            by_cache[cache_key] = row
            for duplicate in duplicates.get(cache_key, []):
                copied = reused_row(duplicate, row)
                rows.append(copied)
                by_result[copied["result_key"]] = copied
        pd.DataFrame(rows).to_csv(output_path, index=False)
        batch_budget[0] += 1

    judgments = pd.DataFrame(rows)
    if not judgments.empty:
        write_audit_template(judgments, output_dir, args.audit_size, args.seed)
    print(f"{directory.name}: {len(rows)}/{len(results)} grounding rows complete")


def main() -> None:
    args = parse_args()
    articles = load_articles(args.dataset_path)
    client = CodexBatchClient(args.judge_model, args.reasoning_effort)
    budget = [0]
    for directory in args.model_dirs:
        process_directory(directory, args, client, articles, budget)
        if args.max_new_batches is not None and budget[0] >= args.max_new_batches:
            break


if __name__ == "__main__":
    main()

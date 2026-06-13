#!/usr/bin/env python3
"""Judge whether each prediction is supported by the context supplied to the generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from pilot_experiment import LlamaClient, chunk_words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/QuALITY.v1.0.1.htmlstripped.dev"),
    )
    parser.add_argument("--judge-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--base-url", default="https://api.groq.com/openai/v1")
    parser.add_argument("--api-key-env", default="GROQ_API_KEY")
    parser.add_argument("--api-style", choices=("meta", "openai"), default="openai")
    parser.add_argument("--max-new-judgments", type=int)
    parser.add_argument("--audit-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=175)
    parser.add_argument("--max-parse-retries", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def normalize_top_k(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return str(int(float(text)))


def result_key(row: pd.Series | dict) -> str:
    return (
        f"{row['example_id']}|{row['method']}|{normalize_top_k(row.get('top_k', ''))}"
    )


def load_articles(path: Path) -> dict[str, str]:
    articles: dict[str, str] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            articles.setdefault(str(item["article_id"]), item["article"])
    return articles


def load_manifest(results_path: Path) -> dict:
    path = results_path.parent / "experiment_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Grounding evaluation requires the experiment manifest."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def parse_indices(value: object) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    text = str(value).strip()
    return [int(item) for item in json.loads(text)] if text else []


def supplied_context(
    row: pd.Series, articles: dict[str, str], chunk_size: int, overlap: int
) -> str:
    article = articles[str(row["article_id"])]
    if row["method"] == "full_context":
        return article
    chunks = chunk_words(article, chunk_size, overlap)
    return "\n\n".join(
        chunks[index] for index in parse_indices(row["retrieved_chunk_indices"])
    )


def evidence_cache_key(row: pd.Series, context: str) -> str:
    material = "\n".join(
        [
            str(row["article_id"]),
            str(row["question"]),
            str(row["prediction"]),
            context,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def build_judge_prompt(row: pd.Series, context: str) -> str:
    choices = json.loads(row["choices_json"])
    predicted_index = "ABCD".find(str(row["prediction"]))
    predicted_text = choices[predicted_index] if predicted_index >= 0 else "INVALID"
    choices_text = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip("ABCD", choices)
    )
    return f"""You are a strict independent grounding evaluator.

Decide whether the supplied context alone contains enough evidence to justify
the exact predicted answer to the multiple-choice question.

Rules:
- Do not use outside knowledge or the dataset's gold answer.
- Judge the exact predicted option, not a nearby fact or a different option.
- Evidence about a different person, object, event, time, or ship does not count.
- If the predicted option combines multiple claims, every claim must be supported.
- For an inference, the context must make that inference clearly justified.
- Options such as "unknown", "we never learn", or "not stated" are supported only
  when the supplied context does not resolve the question. For these absence-based
  answers, supporting_quote may be an empty string.
- Otherwise, supported answers require one short exact contiguous quote copied
  from the supplied context.
- When uncertain, mark supported false.

Question:
{row['question']}

Choices:
{choices_text}

Predicted answer:
{row['prediction']}. {predicted_text}

Supplied context:
{context}

Return only one JSON object with exactly these fields:
{{
  "supported": true or false,
  "confidence": "high", "medium", or "low",
  "supporting_quote": "an exact quote from the supplied context, or an empty string when unsupported",
  "reason": "one short sentence"
}}
"""


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
    payload = json.loads(cleaned)
    required = {"supported", "confidence", "supporting_quote", "reason"}
    if set(payload) != required:
        raise ValueError(f"Judge JSON fields must be exactly {sorted(required)}")
    if not isinstance(payload["supported"], bool):
        raise ValueError("supported must be a JSON boolean")
    if payload["confidence"] not in {"high", "medium", "low"}:
        raise ValueError("confidence must be high, medium, or low")
    if not isinstance(payload["supporting_quote"], str) or not isinstance(
        payload["reason"], str
    ):
        raise ValueError("supporting_quote and reason must be strings")
    return payload


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_matches_context(quote: str, context: str) -> bool:
    """Accept exact text or a contiguous token sequence with punctuation changes."""
    if normalized(quote) in normalized(context):
        return True

    def tokens(text: str) -> list[str]:
        canonical = unicodedata.normalize("NFKC", text).lower()
        return re.findall(r"[a-z0-9]+", canonical)

    quote_tokens = tokens(quote)
    if len(quote_tokens) < 4:
        return False
    quote_sequence = f" {' '.join(quote_tokens)} "
    context_sequence = f" {' '.join(tokens(context))} "
    return quote_sequence in context_sequence


def judge_one(
    client: LlamaClient, row: pd.Series, context: str, max_parse_retries: int
) -> tuple[dict, str, float, int | None, int | None]:
    prompt = build_judge_prompt(row, context)
    last_error: Exception | None = None
    for _ in range(max_parse_retries + 1):
        generation = client.ask(prompt)
        try:
            payload = extract_json(generation.text)
            if (
                payload["supported"]
                and payload["supporting_quote"].strip()
                and not quote_matches_context(payload["supporting_quote"], context)
            ):
                raise ValueError("supporting_quote is not an exact context quote")
            return (
                payload,
                generation.text,
                generation.latency_seconds,
                generation.provider_prompt_tokens,
                generation.provider_output_tokens,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            invalid_quote = ""
            if "payload" in locals() and isinstance(payload, dict):
                invalid_quote = str(payload.get("supporting_quote", ""))[:300]
            prompt += (
                "\nYour previous response was invalid because the supporting quote was "
                "not found as a contiguous passage in the supplied context. Do not "
                "paraphrase or combine separate passages. For ordinary supported "
                "answers, copy a short contiguous quote directly from the context. "
                "For absence-based answers such as unknown or not stated, an empty "
                "quote is allowed when the context truly does not resolve the question. "
                "Otherwise, if no valid quote exists, set supported to false. "
                f"Invalid previous quote: {json.dumps(invalid_quote)}. Return only JSON."
            )
    print(
        "Warning: recording a conservative unsupported judgment after repeated "
        f"validation failures for {result_key(row)}: {last_error}"
    )
    fallback = {
        "supported": False,
        "confidence": "low",
        "supporting_quote": "",
        "reason": (
            "The judge claimed support but did not provide a verifiable contiguous "
            "quote after repeated attempts."
        ),
    }
    return (
        fallback,
        generation.text,
        generation.latency_seconds,
        generation.provider_prompt_tokens,
        generation.provider_output_tokens,
    )


def parse_human_label(value: object) -> int | None:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "supported"}:
        return 1
    if text in {"0", "false", "no", "unsupported"}:
        return 0
    return None


def write_audit_template(
    judgments: pd.DataFrame, output_dir: Path, audit_size: int, seed: int
) -> None:
    path = output_dir / "manual_audit.csv"
    existing_labels: dict[str, tuple[object, object]] = {}
    if path.exists():
        existing = pd.read_csv(path, keep_default_na=False)
        existing_labels = {
            row["result_key"]: (
                row.get("human_supported", ""),
                row.get("human_notes", ""),
            )
            for _, row in existing.iterrows()
        }

    rng = random.Random(seed)
    audit_rows = []
    selected_ids: set[str] = set()
    per_label_target = audit_size // 2
    for label in (0, 1):
        candidates = judgments[judgments["judge_supported"] == label].sort_values(
            "result_key"
        )
        candidate_rows = [
            group.iloc[rng.randrange(len(group))]
            for _, group in candidates.groupby("example_id")
        ]
        rng.shuffle(candidate_rows)
        for row in candidate_rows:
            if row["example_id"] in selected_ids:
                continue
            audit_rows.append(row)
            selected_ids.add(row["example_id"])
            if (
                sum(int(item["judge_supported"]) == label for item in audit_rows)
                >= per_label_target
            ):
                break
    remaining_rows = [
        group.iloc[rng.randrange(len(group))]
        for example_id, group in judgments.groupby("example_id")
        if example_id not in selected_ids
    ]
    rng.shuffle(remaining_rows)
    audit_rows.extend(remaining_rows[: max(0, audit_size - len(audit_rows))])
    audit = pd.DataFrame(audit_rows).copy()
    audit["human_supported"] = [
        existing_labels.get(key, ("", ""))[0] for key in audit["result_key"]
    ]
    audit["human_notes"] = [
        existing_labels.get(key, ("", ""))[1] for key in audit["result_key"]
    ]
    columns = [
        "result_key",
        "example_id",
        "method",
        "top_k",
        "question",
        "prediction",
        "judge_supported",
        "confidence",
        "supporting_quote",
        "reason",
        "supplied_context",
        "human_supported",
        "human_notes",
    ]
    audit[columns].to_csv(path, index=False)

    human = [parse_human_label(value) for value in audit["human_supported"]]
    valid = [index for index, value in enumerate(human) if value is not None]
    summary = {
        "n_audit_rows": len(audit),
        "n_human_labeled": len(valid),
        "agreement": "",
        "cohen_kappa": "",
        "passes_80_percent_agreement": "",
    }
    if valid:
        judge_labels = [int(audit.iloc[index]["judge_supported"]) for index in valid]
        human_labels = [int(human[index]) for index in valid]
        agreement = sum(a == b for a, b in zip(judge_labels, human_labels)) / len(valid)
        summary["agreement"] = agreement
        summary["cohen_kappa"] = (
            cohen_kappa_score(human_labels, judge_labels)
            if len(set(human_labels)) > 1
            else ""
        )
        summary["passes_80_percent_agreement"] = agreement >= 0.8
    pd.DataFrame([summary]).to_csv(
        output_dir / "grounding_validation_summary.csv", index=False
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} before running grounding evaluation.")

    results = pd.read_csv(
        args.results, keep_default_na=False, dtype={"article_id": str}
    )
    manifest = load_manifest(args.results)
    articles = load_articles(args.dataset_path)
    chunk_size = int(manifest["chunk_size_words"])
    overlap = int(manifest["overlap_words"])
    output_path = args.output_dir / "grounding_judgments.csv"
    existing = (
        pd.read_csv(output_path, keep_default_na=False).to_dict("records")
        if output_path.exists() and not args.no_resume
        else []
    )
    existing_by_result = {row["result_key"]: row for row in existing}
    existing_by_cache = {row["cache_key"]: row for row in existing}
    rows = list(existing)
    new_api_judgments = 0
    client = LlamaClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.judge_model,
        api_style=args.api_style,
        timeout=180,
        max_retries=3,
    )

    for _, result in results.iterrows():
        key = result_key(result)
        if key in existing_by_result:
            continue
        context = supplied_context(result, articles, chunk_size, overlap)
        cache_key = evidence_cache_key(result, context)
        cached = existing_by_cache.get(cache_key)
        if cached:
            row = dict(cached)
            row.update(
                {
                    "result_key": key,
                    "method": result["method"],
                    "top_k": normalize_top_k(result.get("top_k", "")),
                    "reused_from_result_key": cached["result_key"],
                }
            )
        else:
            if (
                args.max_new_judgments is not None
                and new_api_judgments >= args.max_new_judgments
            ):
                print(
                    f"Reached --max-new-judgments={args.max_new_judgments}; checkpoint saved."
                )
                break
            payload, raw, latency, prompt_tokens, output_tokens = judge_one(
                client, result, context, args.max_parse_retries
            )
            new_api_judgments += 1
            row = {
                "result_key": key,
                "cache_key": cache_key,
                "reused_from_result_key": "",
                "example_id": result["example_id"],
                "article_id": result["article_id"],
                "method": result["method"],
                "top_k": normalize_top_k(result.get("top_k", "")),
                "question": result["question"],
                "prediction": result["prediction"],
                "is_correct": result["is_correct"],
                "judge_supported": int(payload["supported"]),
                "confidence": payload["confidence"],
                "supporting_quote": payload["supporting_quote"],
                "reason": payload["reason"],
                "raw_judge_output": raw,
                "judge_latency_seconds": latency,
                "judge_prompt_tokens": (
                    prompt_tokens if prompt_tokens is not None else ""
                ),
                "judge_output_tokens": (
                    output_tokens if output_tokens is not None else ""
                ),
                "judge_model": args.judge_model,
                "supplied_context": context,
            }
            existing_by_cache[cache_key] = row
        rows.append(row)
        existing_by_result[key] = row
        pd.DataFrame(rows).to_csv(output_path, index=False)

    judgments = pd.DataFrame(rows)
    if not judgments.empty:
        write_audit_template(judgments, args.output_dir, args.audit_size, args.seed)
        print(
            judgments.groupby("method", as_index=False)
            .agg(
                grounding_support_rate=("judge_supported", "mean"),
                n=("result_key", "count"),
            )
            .to_markdown(index=False)
        )


if __name__ == "__main__":
    main()

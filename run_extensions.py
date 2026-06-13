#!/usr/bin/env python3
"""Run targeted prompt, retrieval, and long-context stress extensions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from pilot_experiment import (
    Example,
    GenerationResult,
    LlamaClient,
    Retriever,
    ask_valid_answer,
    build_plain_prompt,
    build_structured_prompt,
    chunk_words,
    format_choices,
    load_examples,
    load_excluded_article_ids,
    make_token_counter,
    query_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("prompt", "retrieval", "stress"),
        required=True,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/QuALITY.v1.0.1.htmlstripped.dev"),
    )
    parser.add_argument(
        "--exclude-pilot-articles",
        type=Path,
        default=Path("outputs/multi_model/all_results_main.csv"),
    )
    parser.add_argument(
        "--primary-results",
        type=Path,
        default=Path("outputs/final_openrouter_llama/results_main.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--stress-limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=175)
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument(
        "--sentence-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    parser.add_argument("--model", default="meta-llama/llama-4-scout")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--api-style", choices=("openai", "meta"), default="openai")
    parser.add_argument("--max-new-calls", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    return int(digest[:16], 16)


def load_articles(path: Path) -> dict[str, dict]:
    articles: dict[str, dict] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            articles[str(item["article_id"])] = item
    return articles


def build_structured_neutral_prompt(example: Example, evidence: list[str]) -> str:
    evidence_text = "\n\n".join(
        f"Evidence {index}\n{chunk}" for index, chunk in enumerate(evidence, start=1)
    )
    return (
        f"Question\n{example.question}\n\n{evidence_text}\n\n"
        f"Choices\n{format_choices(example.choices)}\n\n"
        "Instruction: Answer the multiple-choice question using the evidence. "
        "Choose A, B, C, or D. Respond with only the letter."
    )


def build_structured_synthesis_prompt(example: Example, evidence: list[str]) -> str:
    evidence_text = "\n\n".join(
        f"Evidence {index}\n{chunk}" for index, chunk in enumerate(evidence, start=1)
    )
    return (
        f"Question\n{example.question}\n\n{evidence_text}\n\n"
        f"Choices\n{format_choices(example.choices)}\n\n"
        "Instruction: Compare every answer choice against the supplied evidence, "
        "select the best-supported choice, and then respond with only A, B, C, or D."
    )


def build_multi_document_prompt(
    example: Example, documents: list[tuple[str, str]]
) -> str:
    document_text = "\n\n".join(
        f"Document {index}\n{text}"
        for index, (_, text) in enumerate(documents, start=1)
    )
    return (
        f"{document_text}\n\nQuestion:\n{example.question}\n\n"
        f"Choices:\n{format_choices(example.choices)}\n\n"
        "Instruction: Answer the multiple-choice question using the documents. "
        "Respond with only A, B, C, or D."
    )


def arrange_documents(
    target: tuple[str, str], distractors: list[tuple[str, str]], position: str
) -> list[tuple[str, str]]:
    if position == "begin":
        return [target, *distractors]
    if position == "end":
        return [*distractors, target]
    middle = len(distractors) // 2
    return [*distractors[:middle], target, *distractors[middle:]]


def parse_indices(value: object) -> list[int]:
    return [int(item) for item in json.loads(str(value))]


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return pd.read_csv(path, keep_default_na=False).to_dict("records")


def save_rows(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def row_key(row: dict) -> str:
    return f"{row['example_id']}|{row['variant']}"


def result_row(
    example: Example,
    variant: str,
    prompt: str,
    generation: GenerationResult,
    count_tokens: Callable[[str], int],
    metadata: dict | None = None,
) -> dict:
    from pilot_experiment import parse_answer

    prediction = parse_answer(generation.text, example.choices)
    if prediction not in "ABCD":
        raise ValueError(
            f"Invalid prediction for {example.example_id}: {generation.text!r}"
        )
    provider_tokens = generation.provider_prompt_tokens
    row = {
        "example_id": example.example_id,
        "article_id": example.article_id,
        "question": example.question,
        "correct_answer": example.correct_answer,
        "difficult": example.difficult,
        "context_requirement": example.context_requirement,
        "context_scope": "local" if example.context_requirement <= 2 else "global",
        "variant": variant,
        "prediction": prediction,
        "is_correct": int(prediction == example.correct_answer),
        "raw_output": generation.text,
        "input_tokens": (
            provider_tokens if provider_tokens is not None else count_tokens(prompt)
        ),
        "estimated_input_tokens": count_tokens(prompt),
        "provider_prompt_tokens": (
            provider_tokens if provider_tokens is not None else ""
        ),
        "provider_output_tokens": generation.provider_output_tokens or "",
        "response_latency_seconds": generation.latency_seconds,
        "model": "",
    }
    row.update(metadata or {})
    return row


def copied_primary_row(
    source: pd.Series, variant: str, metadata: dict | None = None
) -> dict:
    row = {
        "example_id": source["example_id"],
        "article_id": source["article_id"],
        "question": source["question"],
        "correct_answer": source["correct_answer"],
        "difficult": source["difficult"],
        "context_requirement": source["context_requirement"],
        "context_scope": source["context_scope"],
        "variant": variant,
        "prediction": source["prediction"],
        "is_correct": source["is_correct"],
        "raw_output": source["raw_output"],
        "input_tokens": source["input_tokens"],
        "estimated_input_tokens": source["estimated_input_tokens"],
        "provider_prompt_tokens": source["provider_prompt_tokens"],
        "provider_output_tokens": source["provider_output_tokens"],
        "response_latency_seconds": source["response_latency_seconds"],
        "model": source["model"],
        "source": "reused_primary",
    }
    row.update(metadata or {})
    return row


def normalized_scores(scores: np.ndarray) -> np.ndarray:
    low, high = float(scores.min()), float(scores.max())
    if math.isclose(low, high):
        return np.zeros_like(scores)
    return (scores - low) / (high - low)


def bm25_scores(chunks: list[str], query: str) -> np.ndarray:
    documents = [re.findall(r"\w+", chunk.lower()) for chunk in chunks]
    query_terms = re.findall(r"\w+", query.lower())
    n_documents = len(documents)
    document_frequency = Counter(
        term for document in documents for term in set(document)
    )
    avg_length = sum(len(document) for document in documents) / n_documents
    scores = []
    for document in documents:
        counts = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = counts[term]
            if not frequency:
                continue
            idf = math.log(
                1
                + (n_documents - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            denominator = frequency + 1.5 * (
                1 - 0.75 + 0.75 * len(document) / avg_length
            )
            score += idf * frequency * 2.5 / denominator
        scores.append(score)
    return np.asarray(scores)


def rank_retrieval_variants(
    chunks: list[str],
    query: str,
    sentence_model,
    cross_encoder,
) -> dict[str, tuple[list[int], list[float]]]:
    embeddings = np.asarray(
        sentence_model.encode(chunks + [query], normalize_embeddings=True)
    )
    chunk_embeddings, query_embedding = embeddings[:-1], embeddings[-1]
    dense_scores = chunk_embeddings @ query_embedding
    dense_rank = np.argsort(dense_scores)[::-1].tolist()

    lexical = normalized_scores(bm25_scores(chunks, query))
    hybrid_scores = 0.5 * normalized_scores(dense_scores) + 0.5 * lexical
    hybrid_rank = np.argsort(hybrid_scores)[::-1].tolist()

    candidates = dense_rank[: min(10, len(dense_rank))]
    rerank_scores = np.asarray(
        cross_encoder.predict([(query, chunks[index]) for index in candidates])
    ).reshape(-1)
    rerank_order = np.argsort(rerank_scores)[::-1].tolist()
    rerank_rank = [candidates[index] for index in rerank_order]

    mmr_rank: list[int] = []
    remaining = set(range(len(chunks)))
    while remaining and len(mmr_rank) < min(5, len(chunks)):
        best_index = max(
            remaining,
            key=lambda index: (
                0.7 * dense_scores[index]
                - 0.3
                * (
                    max(
                        float(chunk_embeddings[index] @ chunk_embeddings[selected])
                        for selected in mmr_rank
                    )
                    if mmr_rank
                    else 0
                )
            ),
        )
        mmr_rank.append(best_index)
        remaining.remove(best_index)

    return {
        "dense": (dense_rank, [float(dense_scores[index]) for index in dense_rank]),
        "hybrid": (
            hybrid_rank,
            [float(hybrid_scores[index]) for index in hybrid_rank],
        ),
        "rerank": (
            rerank_rank,
            [float(rerank_scores[index]) for index in rerank_order],
        ),
        "mmr": (mmr_rank, [float(dense_scores[index]) for index in mmr_rank]),
    }


def setup(
    args: argparse.Namespace,
) -> tuple[list[Example], pd.DataFrame, Callable[[str], int]]:
    excluded = load_excluded_article_ids(args.exclude_pilot_articles)
    examples = load_examples(
        args.dataset_path,
        args.limit,
        args.seed,
        sampling="paired-article-difficulty",
        excluded_article_ids=excluded,
    )
    primary = pd.read_csv(
        args.primary_results, keep_default_na=False, dtype={"article_id": str}
    )
    count_tokens, _ = make_token_counter()
    return examples, primary, count_tokens


def make_client(args: argparse.Namespace) -> LlamaClient:
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} before running.")
    return LlamaClient(api_key, args.base_url, args.model, args.api_style, 180, 3)


def can_call(args: argparse.Namespace, new_calls: int) -> bool:
    return args.max_new_calls is None or new_calls < args.max_new_calls


def run_prompt(args: argparse.Namespace) -> None:
    examples, primary, count_tokens = setup(args)
    output_path = args.output_dir / "results_prompt_ablation.csv"
    rows = read_rows(output_path)
    existing = {row_key(row) for row in rows}
    client = None if args.dry_run else make_client(args)
    new_calls = 0

    for example in examples:
        chunks = chunk_words(example.article, args.chunk_size, args.overlap)
        source_rows = primary[primary["example_id"] == example.example_id]
        plain = source_rows[source_rows["method"] == "plain_retrieval"].iloc[0]
        strict = source_rows[source_rows["method"] == "structured_retrieval"].iloc[0]
        indices = parse_indices(plain["retrieved_chunk_indices"])
        evidence = [chunks[index] for index in indices]
        for variant, source in (("plain", plain), ("structured_strict", strict)):
            key = f"{example.example_id}|{variant}"
            if key not in existing:
                rows.append(
                    copied_primary_row(
                        source,
                        variant,
                        {"retrieved_chunk_indices": json.dumps(indices)},
                    )
                )
                existing.add(key)
        prompts = {
            "structured_neutral": build_structured_neutral_prompt(example, evidence),
            "structured_synthesis": build_structured_synthesis_prompt(
                example, evidence
            ),
        }
        for variant, prompt in prompts.items():
            key = f"{example.example_id}|{variant}"
            if key in existing:
                continue
            if args.dry_run:
                rows.append(
                    {
                        "example_id": example.example_id,
                        "article_id": example.article_id,
                        "variant": variant,
                        "estimated_input_tokens": count_tokens(prompt),
                    }
                )
            elif can_call(args, new_calls):
                generation = ask_valid_answer(client, prompt, example.choices)
                rows.append(
                    result_row(
                        example,
                        variant,
                        prompt,
                        generation,
                        count_tokens,
                        {
                            "retrieved_chunk_indices": json.dumps(indices),
                            "source": "new_call",
                            "model": args.model,
                        },
                    )
                )
                new_calls += 1
            else:
                save_rows(rows, output_path)
                return
            existing.add(key)
            save_rows(rows, output_path)
    save_rows(rows, output_path)


def run_retrieval(args: argparse.Namespace) -> None:
    examples, primary, count_tokens = setup(args)
    output_path = args.output_dir / "results_retrieval_ablation.csv"
    rows = read_rows(output_path)
    existing = {row_key(row) for row in rows}
    retriever = Retriever("sentence-transformer", args.sentence_model)
    from sentence_transformers import CrossEncoder

    cross_encoder = CrossEncoder(args.cross_encoder_model)
    client = None if args.dry_run else make_client(args)
    new_calls = 0

    for example in examples:
        chunks = chunk_words(example.article, args.chunk_size, args.overlap)
        source = primary[
            (primary["example_id"] == example.example_id)
            & (primary["method"] == "plain_retrieval")
        ].iloc[0]
        key = f"{example.example_id}|dense"
        if key not in existing:
            rows.append(
                copied_primary_row(
                    source,
                    "dense",
                    {
                        "retrieved_chunk_indices": source["retrieved_chunk_indices"],
                        "retrieval_scores": source["retrieval_scores"],
                    },
                )
            )
            existing.add(key)
        rankings = rank_retrieval_variants(
            chunks, query_text(example), retriever.sentence_model, cross_encoder
        )
        for variant in ("hybrid", "rerank", "mmr"):
            key = f"{example.example_id}|{variant}"
            if key in existing:
                continue
            ranked, scores = rankings[variant]
            indices = ranked[:3]
            prompt = build_plain_prompt(example, [chunks[index] for index in indices])
            if args.dry_run:
                rows.append(
                    {
                        "example_id": example.example_id,
                        "article_id": example.article_id,
                        "variant": variant,
                        "estimated_input_tokens": count_tokens(prompt),
                        "retrieved_chunk_indices": json.dumps(indices),
                    }
                )
            elif can_call(args, new_calls):
                generation = ask_valid_answer(client, prompt, example.choices)
                rows.append(
                    result_row(
                        example,
                        variant,
                        prompt,
                        generation,
                        count_tokens,
                        {
                            "retrieved_chunk_indices": json.dumps(indices),
                            "retrieval_scores": json.dumps(scores[:3]),
                            "source": "new_call",
                            "model": args.model,
                        },
                    )
                )
                new_calls += 1
            else:
                save_rows(rows, output_path)
                return
            existing.add(key)
            save_rows(rows, output_path)
    save_rows(rows, output_path)


def stress_corpus(
    example: Example,
    articles: dict[str, dict],
    count: int,
    seed: int,
) -> tuple[tuple[str, str], list[tuple[str, str]]]:
    target = (example.article_id, example.article)
    candidates = [
        (article_id, article["article"])
        for article_id, article in articles.items()
        if article_id != example.article_id
    ]
    rng = random.Random(stable_seed(seed, example.article_id))
    return target, rng.sample(candidates, count)


def chunk_documents(
    documents: list[tuple[str, str]], chunk_size: int, overlap: int
) -> tuple[list[str], list[str]]:
    chunks, article_ids = [], []
    for article_id, document in documents:
        document_chunks = chunk_words(document, chunk_size, overlap)
        chunks.extend(document_chunks)
        article_ids.extend([article_id] * len(document_chunks))
    return chunks, article_ids


def run_stress(args: argparse.Namespace) -> None:
    examples, primary, count_tokens = setup(args)
    examples = examples[: args.stress_limit]
    articles = load_articles(args.dataset_path)
    output_path = args.output_dir / "results_stress_test.csv"
    rows = read_rows(output_path)
    existing = {row_key(row) for row in rows}
    retriever = Retriever("sentence-transformer", args.sentence_model)
    client = None if args.dry_run else make_client(args)
    new_calls = 0

    for example in examples:
        source = primary[
            (primary["example_id"] == example.example_id)
            & (primary["method"] == "full_context")
        ].iloc[0]
        key = f"{example.example_id}|full_6k"
        if key not in existing:
            rows.append(
                copied_primary_row(
                    source,
                    "full_6k",
                    {
                        "context_level": "6k",
                        "target_position": "only",
                        "distractor_count": 0,
                        "target_chunks_in_top3": "",
                    },
                )
            )
            existing.add(key)

        target, distractors = stress_corpus(example, articles, 7, args.seed)
        for level, count in (("20k", 3), ("40k", 7)):
            selected_distractors = distractors[:count]
            for position in ("begin", "middle", "end"):
                variant = f"full_{level}_{position}"
                key = f"{example.example_id}|{variant}"
                if key in existing:
                    continue
                documents = arrange_documents(target, selected_distractors, position)
                prompt = build_multi_document_prompt(example, documents)
                if args.dry_run:
                    rows.append(
                        {
                            "example_id": example.example_id,
                            "article_id": example.article_id,
                            "variant": variant,
                            "estimated_input_tokens": count_tokens(prompt),
                        }
                    )
                elif can_call(args, new_calls):
                    generation = ask_valid_answer(client, prompt, example.choices)
                    rows.append(
                        result_row(
                            example,
                            variant,
                            prompt,
                            generation,
                            count_tokens,
                            {
                                "context_level": level,
                                "target_position": position,
                                "distractor_count": count,
                                "target_chunks_in_top3": "",
                                "source": "new_call",
                                "model": args.model,
                            },
                        )
                    )
                    new_calls += 1
                else:
                    save_rows(rows, output_path)
                    return
                existing.add(key)
                save_rows(rows, output_path)

            documents = [target, *selected_distractors]
            chunks, article_ids = chunk_documents(
                documents, args.chunk_size, args.overlap
            )
            ranked, scores = retriever.rank(chunks, query_text(example))
            indices = ranked[:3]
            evidence = [chunks[index] for index in indices]
            target_hits = sum(
                article_ids[index] == example.article_id for index in indices
            )
            prompts = {
                f"plain_{level}": build_plain_prompt(example, evidence),
                f"structured_{level}": build_structured_prompt(example, evidence),
            }
            for variant, prompt in prompts.items():
                key = f"{example.example_id}|{variant}"
                if key in existing:
                    continue
                if args.dry_run:
                    rows.append(
                        {
                            "example_id": example.example_id,
                            "article_id": example.article_id,
                            "variant": variant,
                            "estimated_input_tokens": count_tokens(prompt),
                            "target_chunks_in_top3": target_hits,
                        }
                    )
                elif can_call(args, new_calls):
                    generation = ask_valid_answer(client, prompt, example.choices)
                    rows.append(
                        result_row(
                            example,
                            variant,
                            prompt,
                            generation,
                            count_tokens,
                            {
                                "context_level": level,
                                "target_position": "retrieved",
                                "distractor_count": count,
                                "target_chunks_in_top3": target_hits,
                                "retrieved_chunk_indices": json.dumps(indices),
                                "retrieval_scores": json.dumps(scores[:3]),
                                "source": "new_call",
                                "model": args.model,
                            },
                        )
                    )
                    new_calls += 1
                else:
                    save_rows(rows, output_path)
                    return
                existing.add(key)
                save_rows(rows, output_path)
    save_rows(rows, output_path)


def write_manifest(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "extension_manifest.json"
    manifest = {
        "experiment": args.experiment,
        "dataset_path": str(args.dataset_path.resolve()),
        "primary_results": str(args.primary_results.resolve()),
        "limit": args.limit,
        "stress_limit": args.stress_limit,
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "sentence_model": args.sentence_model,
        "cross_encoder_model": args.cross_encoder_model,
        "model": args.model,
        "base_url": args.base_url,
    }
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise SystemExit(f"Settings differ from {path}; use another output directory.")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    write_manifest(args)
    if args.experiment == "prompt":
        run_prompt(args)
    elif args.experiment == "retrieval":
        run_retrieval(args)
    else:
        run_stress(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Small QuALITY pilot comparing full context, retrieval, and structured retrieval."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATASET_URL = (
    "https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/"
    "QuALITY.v1.0.1.htmlstripped.dev"
)
LETTERS = "ABCD"
VALID_ANSWERS = set(LETTERS)
ABLATION_KS = (1, 3, 5)


@dataclass
class Example:
    example_id: str
    article_id: str
    title: str
    article: str
    question: str
    choices: list[str]
    correct_answer: str
    difficult: int
    context_requirement: float


@dataclass
class GenerationResult:
    text: str
    latency_seconds: float
    provider_prompt_tokens: int | None = None
    provider_output_tokens: int | None = None


class Retriever:
    def __init__(self, kind: str, sentence_model: str):
        self.kind = kind
        self.sentence_model_name = sentence_model
        self.sentence_model = None
        if kind == "sentence-transformer":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Install sentence-transformers before using "
                    "--retriever sentence-transformer."
                ) from exc
            self.sentence_model = SentenceTransformer(sentence_model)

    def rank(self, chunks: list[str], query: str) -> tuple[list[int], list[float]]:
        if self.kind == "tfidf":
            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform(chunks + [query])
            scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        else:
            embeddings = self.sentence_model.encode(
                chunks + [query], normalize_embeddings=True
            )
            scores = cosine_similarity(embeddings[-1:], embeddings[:-1]).ravel()

        ranked = scores.argsort()[::-1].tolist()
        return ranked, [float(scores[index]) for index in ranked]


class LlamaClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        api_style: str,
        timeout: int,
        max_retries: int,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_style = api_style
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def ask(self, prompt: str) -> GenerationResult:
        model_lower = self.model.lower()
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if not model_lower.startswith(("gpt-5", "o1", "o3", "o4")):
            payload["temperature"] = 0
        if self.api_style == "meta":
            payload["max_completion_tokens"] = 8
        elif model_lower.startswith(("gpt-5", "o1", "o3", "o4")):
            payload["max_completion_tokens"] = 1024
            payload["reasoning_effort"] = "low"
        else:
            # Reasoning-capable APIs may consume several tokens before emitting
            # the requested answer letter.
            payload["max_tokens"] = (
                1024 if "llama-4-scout" in self.model.lower() else 256
            )
        if model_lower.startswith("openai/gpt-oss"):
            payload["reasoning_effort"] = "low"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                response = self.session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                prompt_tokens, output_tokens = extract_usage(payload)
                return GenerationResult(
                    text=extract_response_text(payload),
                    latency_seconds=time.perf_counter() - start_time,
                    provider_prompt_tokens=prompt_tokens,
                    provider_output_tokens=output_tokens,
                )
            except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"Llama API request failed: {exc}") from exc
                retry_after = 0.0
                status_code = ""
                error_detail = ""
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    status_code = str(exc.response.status_code)
                    retry_after = float(exc.response.headers.get("retry-after", 0))
                    try:
                        error_payload = exc.response.json()
                        error_detail = str(error_payload.get("error", error_payload))
                    except ValueError:
                        error_detail = exc.response.text[:300]
                wait_seconds = max(retry_after, 2**attempt)
                print(
                    f"API request failed{f' with HTTP {status_code}' if status_code else ''}; "
                    f"waiting {wait_seconds:.1f}s before retry {attempt + 2}/"
                    f"{self.max_retries + 1}. {error_detail}",
                    flush=True,
                )
                time.sleep(wait_seconds)
        raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=30, help="Number of QA examples.")
    parser.add_argument("--seed", type=int, default=175)
    parser.add_argument(
        "--sampling",
        choices=("random", "paired-article-difficulty"),
        default="random",
        help="Paired sampling selects one difficult and one non-difficult question per article.",
    )
    parser.add_argument(
        "--exclude-pilot-articles",
        type=Path,
        help="CSV whose article_id values must be excluded from sampling.",
    )
    parser.add_argument("--chunk-size", type=int, default=400, help="Words per chunk.")
    parser.add_argument("--overlap", type=int, default=50, help="Overlapping words.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/QuALITY.v1.0.1.htmlstripped.dev"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--retriever",
        choices=("tfidf", "sentence-transformer"),
        default="sentence-transformer",
        help="Sentence-transformers is the default; TF-IDF avoids a model download.",
    )
    parser.add_argument(
        "--sentence-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLAMA_API_CLIENT_BASE_URL", "https://api.llama.com/v1"),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLAMA_MODEL", "Llama-4-Maverick-17B-128E-Instruct-FP8"),
    )
    parser.add_argument("--api-key-env", default="LLAMA_API_KEY")
    parser.add_argument(
        "--api-style",
        choices=("meta", "openai"),
        default="meta",
        help="Controls max token parameter and response parsing.",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--max-new-examples",
        type=int,
        help="Process at most this many incomplete examples before exiting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and token counts without calling the LLM.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoint CSVs and start over.",
    )
    return parser.parse_args()


def download_dataset(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading QuALITY dev split to {path} ...")
    response = requests.get(DATASET_URL, timeout=180)
    response.raise_for_status()
    path.write_bytes(response.content)


def context_requirement(question: dict) -> float:
    values = [
        int(item["untimed_eval2_context"])
        for item in question.get("validation", [])
        if item.get("untimed_eval2_context") is not None
    ]
    return float(statistics.median(values)) if values else float("nan")


def make_example(article: dict, question: dict) -> Example:
    return Example(
        example_id=question["question_unique_id"],
        article_id=article["article_id"],
        title=article["title"],
        article=article["article"],
        question=question["question"],
        choices=question["options"],
        correct_answer=LETTERS[int(question["gold_label"]) - 1],
        difficult=int(question.get("difficult", 0)),
        context_requirement=context_requirement(question),
    )


def load_excluded_article_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"Exclusion CSV does not exist: {path}")
    frame = pd.read_csv(path, dtype={"article_id": str})
    if "article_id" not in frame.columns:
        raise ValueError(f"Exclusion CSV has no article_id column: {path}")
    return set(frame["article_id"].astype(str))


def load_examples(
    path: Path,
    limit: int,
    seed: int,
    sampling: str = "random",
    excluded_article_ids: set[str] | None = None,
) -> list[Example]:
    examples: list[Example] = []
    excluded_article_ids = excluded_article_ids or set()
    with path.open(encoding="utf-8") as file:
        for line in file:
            article = json.loads(line)
            if article["article_id"] in excluded_article_ids:
                continue
            for question in article["questions"]:
                examples.append(make_example(article, question))

    rng = random.Random(seed)
    if sampling == "random":
        rng.shuffle(examples)
        return examples[:limit]

    if limit % 2:
        raise ValueError("--limit must be even for paired-article-difficulty sampling")
    by_article: dict[str, dict[int, list[Example]]] = {}
    for example in examples:
        by_article.setdefault(example.article_id, {0: [], 1: []})[
            example.difficult
        ].append(example)
    eligible = [
        article_id
        for article_id, groups in by_article.items()
        if groups[0] and groups[1]
    ]
    rng.shuffle(eligible)
    article_count = limit // 2
    if len(eligible) < article_count:
        raise ValueError(
            f"Requested {article_count} paired articles, but only {len(eligible)} are eligible"
        )
    selected: list[Example] = []
    for article_id in eligible[:article_count]:
        for difficult in (0, 1):
            candidates = sorted(
                by_article[article_id][difficult], key=lambda item: item.example_id
            )
            selected.append(rng.choice(candidates))
    return selected


def chunk_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("--overlap must be smaller than --chunk-size")
    words = text.split()
    step = chunk_size - overlap
    return [
        " ".join(words[start : start + chunk_size])
        for start in range(0, len(words), step)
    ]


def format_choices(choices: list[str]) -> str:
    return "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, choices))


def build_full_prompt(example: Example) -> str:
    return (
        f"Document:\n{example.article}\n\n"
        f"Question:\n{example.question}\n\n"
        f"Choices:\n{format_choices(example.choices)}\n\n"
        "Instruction: Answer the multiple-choice question using the document. "
        "Respond with only A, B, C, or D."
    )


def build_plain_prompt(example: Example, evidence: list[str]) -> str:
    return (
        f"Retrieved context:\n{chr(10).join(evidence)}\n\n"
        f"Question:\n{example.question}\n\n"
        f"Choices:\n{format_choices(example.choices)}\n\n"
        "Instruction: Answer the multiple-choice question using the retrieved context. "
        "Respond with only A, B, C, or D."
    )


def build_structured_prompt(example: Example, evidence: list[str]) -> str:
    evidence_text = "\n\n".join(
        f"Evidence {index}\n{chunk}" for index, chunk in enumerate(evidence, start=1)
    )
    return (
        f"Question\n{example.question}\n\n"
        f"{evidence_text}\n\n"
        f"Choices\n{format_choices(example.choices)}\n\n"
        "Instruction: Answer only using the evidence above. Choose A, B, C, or D. "
        "If evidence is insufficient, choose the best-supported answer. "
        "Respond with only the letter."
    )


def make_token_counter() -> tuple[Callable[[str], int], str]:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(encoding.encode(text)), "tiktoken_cl100k_base"
    except ImportError:
        pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)
        return lambda text: len(pattern.findall(text)), "regex_approximation"


def extract_response_text(payload: dict) -> str:
    if "completion_message" in payload:
        content = payload["completion_message"].get("content", "")
    else:
        content = payload["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        return str(content.get("text", "")).strip()
    if isinstance(content, list):
        return " ".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content).strip()


def extract_usage(payload: dict) -> tuple[int | None, int | None]:
    usage = payload.get("usage") or {}
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    output = usage.get("completion_tokens", usage.get("output_tokens"))
    return (
        int(prompt) if prompt is not None else None,
        int(output) if output is not None else None,
    )


def parse_answer(raw_output: str, choices: list[str]) -> str:
    cleaned = raw_output.strip()
    match = re.fullmatch(r"([ABCD])(?:[\s.)\]:-]*)", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    leading_choice = re.match(r"([ABCD])[\s.)\]:-]+\S", cleaned, re.IGNORECASE)
    if leading_choice:
        return leading_choice.group(1).upper()
    explicit = re.search(
        r"(?:final\s+answer|answer|choice|option)\s*(?:is)?\s*[:\-]?\s*([ABCD])\b",
        cleaned,
        re.IGNORECASE,
    )
    if explicit:
        return explicit.group(1).upper()
    trailing = re.search(r"\b([ABCD])(?:[\s.)\]:-]*)$", cleaned, re.IGNORECASE)
    if trailing:
        return trailing.group(1).upper()
    normalized = re.sub(r"\W+", " ", cleaned).strip().lower()
    for letter, choice in zip(LETTERS, choices):
        if normalized == re.sub(r"\W+", " ", choice).strip().lower():
            return letter
    return ""


def ask_valid_answer(
    client: LlamaClient, prompt: str, choices: list[str], attempts: int = 5
) -> GenerationResult:
    last_output = ""
    attempt_prompt = prompt
    for _ in range(attempts):
        generation = client.ask(attempt_prompt)
        last_output = generation.text
        if parse_answer(generation.text, choices) in VALID_ANSWERS:
            return generation
        attempt_prompt = (
            prompt
            + "\n\nYour previous response was invalid. You must choose the best-supported "
            "option even when the context is incomplete. Refusal, E, and insufficient-"
            "evidence responses are invalid. Return exactly one uppercase letter: "
            "A, B, C, or D. Do not explain your answer."
        )
    raise RuntimeError(
        f"Model did not return A, B, C, or D after {attempts} attempts: {last_output!r}"
    )


def query_text(example: Example) -> str:
    return f"{example.question}\n{format_choices(example.choices)}"


def read_checkpoint(path: Path, resume: bool) -> pd.DataFrame:
    if resume and path.exists():
        return pd.read_csv(path, keep_default_na=False)
    return pd.DataFrame()


def save_rows(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def manifest_payload(args: argparse.Namespace, examples: list[Example]) -> dict:
    return {
        "schema_version": 2,
        "dataset_path": str(args.dataset_path.resolve()),
        "limit": args.limit,
        "seed": args.seed,
        "sampling": args.sampling,
        "excluded_pilot_articles_path": (
            str(args.exclude_pilot_articles.resolve())
            if args.exclude_pilot_articles
            else None
        ),
        "chunk_size_words": args.chunk_size,
        "overlap_words": args.overlap,
        "main_top_k": 3,
        "ablation_top_k": list(ABLATION_KS),
        "retriever": args.retriever,
        "sentence_model": args.sentence_model,
        "base_url": args.base_url,
        "model": args.model,
        "api_style": args.api_style,
        "token_fallback": "tiktoken_cl100k_base",
        "selected_examples": [
            {
                "example_id": item.example_id,
                "article_id": item.article_id,
                "difficult": item.difficult,
                "context_requirement": item.context_requirement,
            }
            for item in examples
        ],
    }


def write_or_validate_manifest(output_dir: Path, manifest: dict, resume: bool) -> None:
    path = output_dir / "experiment_manifest.json"
    if path.exists() and resume:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise SystemExit(
                "Experiment settings differ from experiment_manifest.json. "
                "Use a new --output-dir or --no-resume."
            )
        return
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    selected = pd.DataFrame(manifest["selected_examples"])
    selected.to_csv(output_dir / "selected_examples.csv", index=False)


def existing_output(
    rows: list[dict], example_id: str, method: str, top_k: int | str
) -> dict | None:
    for row in rows:
        if (
            row["example_id"] == example_id
            and row["method"] == method
            and str(row["top_k"]) == str(top_k)
        ):
            return row
    return None


def result_row(
    example: Example,
    method: str,
    top_k: int | str,
    prompt: str,
    token_count: int,
    token_counter: str,
    generation: GenerationResult,
    chunk_indices: list[int],
    retrieval_scores: list[float],
    model: str,
    retriever: str,
) -> dict:
    prediction = parse_answer(generation.text, example.choices)
    if prediction not in VALID_ANSWERS:
        raise ValueError(
            f"Refusing to save invalid prediction for {example.example_id}: "
            f"{generation.text!r}"
        )
    provider_tokens = generation.provider_prompt_tokens
    return {
        "example_id": example.example_id,
        "article_id": example.article_id,
        "title": example.title,
        "question": example.question,
        "choices_json": json.dumps(example.choices, ensure_ascii=True),
        "correct_answer": example.correct_answer,
        "difficult": example.difficult,
        "context_requirement": example.context_requirement,
        "context_scope": ("local" if example.context_requirement <= 2 else "global"),
        "method": method,
        "top_k": top_k,
        "prediction": prediction,
        "is_correct": int(prediction == example.correct_answer),
        "raw_output": generation.text,
        "response_latency_seconds": round(generation.latency_seconds, 6),
        "input_tokens": provider_tokens if provider_tokens is not None else token_count,
        "estimated_input_tokens": token_count,
        "provider_prompt_tokens": (
            provider_tokens if provider_tokens is not None else ""
        ),
        "provider_output_tokens": (
            generation.provider_output_tokens
            if generation.provider_output_tokens is not None
            else ""
        ),
        "input_token_source": (
            "provider_reported" if provider_tokens is not None else token_counter
        ),
        "token_counter": token_counter,
        "retrieved_chunk_indices": json.dumps(chunk_indices),
        "retrieval_scores": json.dumps([round(score, 6) for score in retrieval_scores]),
        "model": model,
        "retriever": retriever,
    }


def make_summaries(main_df: pd.DataFrame, ablation_df: pd.DataFrame) -> pd.DataFrame:
    main_summary = (
        main_df.groupby("method", as_index=False)
        .agg(
            accuracy=("is_correct", "mean"),
            avg_input_tokens=("input_tokens", "mean"),
            avg_latency_seconds=("response_latency_seconds", "mean"),
            n=("example_id", "count"),
        )
        .assign(table_type="main")
    )
    ablation_summary = (
        ablation_df.groupby("top_k", as_index=False)
        .agg(
            accuracy=("is_correct", "mean"),
            avg_input_tokens=("input_tokens", "mean"),
            avg_latency_seconds=("response_latency_seconds", "mean"),
            n=("example_id", "count"),
        )
        .assign(
            table_type="structured_ablation",
            method=lambda frame: "structured_top_" + frame["top_k"].astype(str),
        )
    )
    columns = [
        "table_type",
        "method",
        "accuracy",
        "avg_input_tokens",
        "avg_latency_seconds",
        "n",
    ]
    return pd.concat(
        [main_summary[columns], ablation_summary[columns]], ignore_index=True
    )


def write_qualitative_examples(
    main_df: pd.DataFrame, path: Path, count: int = 3
) -> None:
    pivot = main_df.pivot_table(
        index="example_id", columns="method", values="prediction", aggfunc="first"
    )
    differing_ids = pivot[pivot.nunique(axis=1) > 1].index.tolist()
    selected_ids = differing_ids[:count]
    if len(selected_ids) < count:
        fallback = [item for item in pivot.index if item not in selected_ids]
        selected_ids.extend(fallback[: count - len(selected_ids)])

    lines = ["# Qualitative Examples", ""]
    for number, example_id in enumerate(selected_ids, start=1):
        rows = main_df[main_df["example_id"] == example_id]
        first = rows.iloc[0]
        lines.extend(
            [
                f"## Example {number}: {example_id}",
                "",
                f"**Question:** {first['question']}",
                "",
                f"**Correct answer:** {first['correct_answer']}",
                "",
                "| Method | Prediction | Correct | Raw output |",
                "|---|---:|---:|---|",
            ]
        )
        for _, row in rows.iterrows():
            raw = str(row["raw_output"]).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {row['method']} | {row['prediction']} | {row['is_correct']} | {raw} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_dry_run(
    examples: list[Example],
    retriever: Retriever,
    args: argparse.Namespace,
    count_tokens: Callable[[str], int],
    token_counter_name: str,
) -> None:
    rows: list[dict] = []
    preview: dict = {}
    for example in examples:
        chunks = chunk_words(example.article, args.chunk_size, args.overlap)
        ranked, scores = retriever.rank(chunks, query_text(example))
        selected = [chunks[index] for index in ranked[:5]]
        prompts = {
            "full_context": build_full_prompt(example),
            "plain_retrieval_top_3": build_plain_prompt(example, selected[:3]),
            "structured_top_1": build_structured_prompt(example, selected[:1]),
            "structured_top_3": build_structured_prompt(example, selected[:3]),
            "structured_top_5": build_structured_prompt(example, selected[:5]),
        }
        for method, prompt in prompts.items():
            rows.append(
                {
                    "example_id": example.example_id,
                    "article_id": example.article_id,
                    "difficult": example.difficult,
                    "context_requirement": example.context_requirement,
                    "method": method,
                    "input_tokens": count_tokens(prompt),
                    "token_counter": token_counter_name,
                }
            )
        if not preview:
            preview = {
                "example_id": example.example_id,
                "correct_answer": example.correct_answer,
                "retrieved_chunk_indices": ranked[:5],
                "retrieval_scores": scores[:5],
                "prompts": prompts,
            }
    dry_df = pd.DataFrame(rows)
    dry_df.to_csv(args.output_dir / "dry_run_prompt_stats.csv", index=False)
    dry_summary = dry_df.groupby("method", as_index=False).agg(
        avg_input_tokens=("input_tokens", "mean"),
        n=("example_id", "count"),
    )
    dry_summary.to_csv(args.output_dir / "dry_run_token_summary.csv", index=False)
    (args.output_dir / "prompt_preview.json").write_text(
        json.dumps(preview, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print("\nDry-run average input tokens:")
    print(dry_summary.to_markdown(index=False))


def run_experiment(
    examples: list[Example],
    retriever: Retriever,
    client: LlamaClient,
    args: argparse.Namespace,
    count_tokens: Callable[[str], int],
    token_counter_name: str,
) -> None:
    main_path = args.output_dir / "results_main.csv"
    ablation_path = args.output_dir / "results_ablation.csv"
    main_rows = read_checkpoint(main_path, not args.no_resume).to_dict("records")
    ablation_rows = read_checkpoint(ablation_path, not args.no_resume).to_dict(
        "records"
    )
    processed_new_examples = 0

    for number, example in enumerate(examples, start=1):
        main_complete = all(
            existing_output(main_rows, example.example_id, method, top_k)
            for method, top_k in (
                ("full_context", ""),
                ("plain_retrieval", 3),
                ("structured_retrieval", 3),
            )
        )
        ablation_complete = all(
            existing_output(
                ablation_rows, example.example_id, "structured_retrieval", top_k
            )
            for top_k in ABLATION_KS
        )
        if main_complete and ablation_complete:
            continue
        if (
            args.max_new_examples is not None
            and processed_new_examples >= args.max_new_examples
        ):
            print(
                f"Reached --max-new-examples={args.max_new_examples}; checkpoint saved."
            )
            break
        processed_new_examples += 1
        print(
            f"[{number}/{len(examples)}] {example.example_id}: {example.question[:70]}"
        )
        chunks = chunk_words(example.article, args.chunk_size, args.overlap)
        ranked, scores = retriever.rank(chunks, query_text(example))
        evidence = [chunks[index] for index in ranked[:5]]
        prompts = {
            "full_context": build_full_prompt(example),
            "plain_retrieval": build_plain_prompt(example, evidence[:3]),
            "structured_1": build_structured_prompt(example, evidence[:1]),
            "structured_3": build_structured_prompt(example, evidence[:3]),
            "structured_5": build_structured_prompt(example, evidence[:5]),
        }
        calls: dict[str, GenerationResult] = {}

        for method, key, top_k in (
            ("full_context", "full_context", ""),
            ("plain_retrieval", "plain_retrieval", 3),
            ("structured_retrieval", "structured_3", 3),
        ):
            old = existing_output(main_rows, example.example_id, method, top_k)
            if old:
                calls[key] = GenerationResult(
                    text=str(old["raw_output"]),
                    latency_seconds=float(old.get("response_latency_seconds", 0)),
                    provider_prompt_tokens=(
                        int(old["provider_prompt_tokens"])
                        if str(old.get("provider_prompt_tokens", "")).strip()
                        else None
                    ),
                    provider_output_tokens=(
                        int(old["provider_output_tokens"])
                        if str(old.get("provider_output_tokens", "")).strip()
                        else None
                    ),
                )
                continue
            generation = ask_valid_answer(client, prompts[key], example.choices)
            calls[key] = generation
            indices = [] if method == "full_context" else ranked[:3]
            method_scores = [] if method == "full_context" else scores[:3]
            main_rows.append(
                result_row(
                    example,
                    method,
                    top_k,
                    prompts[key],
                    count_tokens(prompts[key]),
                    token_counter_name,
                    generation,
                    indices,
                    method_scores,
                    args.model,
                    args.retriever,
                )
            )
            save_rows(main_rows, main_path)

        for top_k in ABLATION_KS:
            old = existing_output(
                ablation_rows, example.example_id, "structured_retrieval", top_k
            )
            if old:
                continue
            key = f"structured_{top_k}"
            generation = calls.get(key)
            if generation is None:
                generation = ask_valid_answer(client, prompts[key], example.choices)
                calls[key] = generation
            ablation_rows.append(
                result_row(
                    example,
                    "structured_retrieval",
                    top_k,
                    prompts[key],
                    count_tokens(prompts[key]),
                    token_counter_name,
                    generation,
                    ranked[:top_k],
                    scores[:top_k],
                    args.model,
                    args.retriever,
                )
            )
            save_rows(ablation_rows, ablation_path)

    main_df = pd.DataFrame(main_rows)
    ablation_df = pd.DataFrame(ablation_rows)
    if main_df.empty or ablation_df.empty:
        return
    summaries = make_summaries(main_df, ablation_df)
    summaries.to_csv(args.output_dir / "summary_tables.csv", index=False)
    write_qualitative_examples(main_df, args.output_dir / "qualitative_examples.md")

    print("\nMain comparison:")
    print(
        summaries[summaries["table_type"] == "main"][
            ["method", "accuracy", "avg_input_tokens", "avg_latency_seconds", "n"]
        ].to_markdown(index=False)
    )
    print("\nStructured retrieval ablation:")
    print(
        summaries[summaries["table_type"] == "structured_ablation"][
            ["method", "accuracy", "avg_input_tokens", "avg_latency_seconds", "n"]
        ].to_markdown(index=False)
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    download_dataset(args.dataset_path)
    excluded_article_ids = load_excluded_article_ids(args.exclude_pilot_articles)
    examples = load_examples(
        args.dataset_path,
        args.limit,
        args.seed,
        sampling=args.sampling,
        excluded_article_ids=excluded_article_ids,
    )
    write_or_validate_manifest(
        args.output_dir, manifest_payload(args, examples), not args.no_resume
    )
    count_tokens, token_counter_name = make_token_counter()
    retriever = Retriever(args.retriever, args.sentence_model)

    print(
        f"Loaded {len(examples)} examples; retriever={args.retriever}; "
        f"token_counter={token_counter_name}"
    )
    if args.dry_run:
        run_dry_run(examples, retriever, args, count_tokens, token_counter_name)
        return

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"Set {args.api_key_env} before running, or use --dry-run to validate prompts."
        )
    client = LlamaClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        api_style=args.api_style,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    run_experiment(examples, retriever, client, args, count_tokens, token_counter_name)


if __name__ == "__main__":
    main()

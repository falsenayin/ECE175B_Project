#!/usr/bin/env python3
"""Run a supplemental generator through the user's ChatGPT Codex subscription."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from codex_subscription import CodexBatchClient, answer_batch_schema
from pilot_experiment import (
    build_full_prompt,
    build_plain_prompt,
    build_structured_prompt,
    chunk_words,
    make_example,
    make_summaries,
    write_qualitative_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-batches", type=int)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/QuALITY.v1.0.1.htmlstripped.dev"),
    )
    return parser.parse_args()


def load_examples(path: Path) -> dict[str, object]:
    examples = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            article = json.loads(line)
            for question in article["questions"]:
                example = make_example(article, question)
                examples[example.example_id] = example
    return examples


def normalize_top_k(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return str(int(float(text)))


def row_key(row: pd.Series | dict) -> str:
    return f"{row['example_id']}|{row['method']}|{normalize_top_k(row.get('top_k', ''))}"


def build_prompt(row: pd.Series, example: object, chunk_size: int, overlap: int) -> str:
    if row["method"] == "full_context":
        return build_full_prompt(example)
    chunks = chunk_words(example.article, chunk_size, overlap)
    indices = [int(item) for item in json.loads(str(row["retrieved_chunk_indices"]))]
    evidence = [chunks[index] for index in indices]
    if row["method"] == "plain_retrieval":
        return build_plain_prompt(example, evidence)
    return build_structured_prompt(example, evidence)


def batch_prompt(tasks: list[tuple[str, str]]) -> str:
    parts = [
        "Answer each independent multiple-choice task below.",
        "Use only the context supplied inside that task.",
        "Return exactly one A/B/C/D answer for every task ID in the JSON schema.",
        "Do not use tools, inspect files, or discuss the answers.",
        "",
    ]
    for task_id, prompt in tasks:
        parts.extend([f'<task id="{task_id}">', prompt, "</task>", ""])
    return "\n".join(parts)


def output_row(
    template: pd.Series,
    answer: str,
    model: str,
    reasoning_effort: str,
    latency: float,
    batch_result: object,
) -> dict:
    row = template.to_dict()
    row.update(
        {
            "prediction": answer,
            "is_correct": int(answer == str(template["correct_answer"])),
            "raw_output": answer,
            "response_latency_seconds": round(latency, 6),
            "input_tokens": int(template["estimated_input_tokens"]),
            "provider_prompt_tokens": "",
            "provider_output_tokens": "",
            "input_token_source": "tiktoken_cl100k_base",
            "model": model,
            "codex_reasoning_effort": reasoning_effort,
            "codex_batch_input_tokens": batch_result.input_tokens or "",
            "codex_batch_cached_input_tokens": batch_result.cached_input_tokens or "",
            "codex_batch_output_tokens": batch_result.output_tokens or "",
        }
    )
    return row


def save(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def copy_top3_from_main(
    templates: pd.DataFrame, main_rows: list[dict], ablation_rows: list[dict]
) -> None:
    main_lookup = {row_key(row): row for row in main_rows}
    existing = {row_key(row) for row in ablation_rows}
    for _, template in templates.iterrows():
        key = row_key(template)
        if normalize_top_k(template["top_k"]) != "3" or key in existing:
            continue
        main_key = f"{template['example_id']}|structured_retrieval|3"
        if main_key not in main_lookup:
            continue
        copied = dict(main_lookup[main_key])
        copied["top_k"] = 3
        ablation_rows.append(copied)
        existing.add(key)


def run_file(
    templates: pd.DataFrame,
    output_path: Path,
    existing_rows: list[dict],
    examples: dict[str, object],
    manifest: dict,
    client: CodexBatchClient,
    args: argparse.Namespace,
    batch_budget: list[int],
) -> list[dict]:
    existing = {row_key(row) for row in existing_rows}
    pending = [row for _, row in templates.iterrows() if row_key(row) not in existing]
    for start in range(0, len(pending), args.batch_size):
        if args.max_new_batches is not None and batch_budget[0] >= args.max_new_batches:
            break
        batch = pending[start : start + args.batch_size]
        tasks = [
            (
                row_key(row),
                build_prompt(
                    row,
                    examples[str(row["example_id"])],
                    int(manifest["chunk_size_words"]),
                    int(manifest["overlap_words"]),
                ),
            )
            for row in batch
        ]
        print(
            f"{output_path.name}: batch {batch_budget[0] + 1}, "
            f"{len(existing_rows)}/{len(templates)} rows complete",
            flush=True,
        )
        response = client.ask(batch_prompt(tasks), answer_batch_schema())
        answers = {
            str(item["id"]): str(item["answer"]).upper()
            for item in response.payload["results"]
        }
        expected = {task_id for task_id, _ in tasks}
        if set(answers) != expected:
            raise RuntimeError(
                f"Codex returned mismatched task IDs. Missing={expected - set(answers)}, "
                f"extra={set(answers) - expected}"
            )
        per_item_latency = response.latency_seconds / len(batch)
        for row in batch:
            key = row_key(row)
            existing_rows.append(
                output_row(
                    row,
                    answers[key],
                    args.model,
                    args.reasoning_effort,
                    per_item_latency,
                    response,
                )
            )
            existing.add(key)
        save(existing_rows, output_path)
        batch_budget[0] += 1
    return existing_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(
        (args.source_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    manifest = dict(source_manifest)
    manifest.update(
        {
            "model": args.model,
            "base_url": "chatgpt-managed-codex-subscription",
            "api_style": "codex-exec-batched",
            "codex_reasoning_effort": args.reasoning_effort,
            "token_measurement_note": (
                "input_tokens are the original experiment prompt's tiktoken estimate; "
                "Codex hidden agent instructions are excluded."
            ),
            "source_template_dir": str(args.source_dir.resolve()),
        }
    )
    manifest_path = args.output_dir / "experiment_manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise SystemExit("Existing output manifest differs; use another output directory.")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        pd.read_csv(args.source_dir / "selected_examples.csv").to_csv(
            args.output_dir / "selected_examples.csv", index=False
        )

    examples = load_examples(args.dataset_path)
    main_templates = pd.read_csv(args.source_dir / "results_main.csv", keep_default_na=False)
    ablation_templates = pd.read_csv(
        args.source_dir / "results_ablation.csv", keep_default_na=False
    )
    main_path = args.output_dir / "results_main.csv"
    ablation_path = args.output_dir / "results_ablation.csv"
    main_rows = (
        pd.read_csv(main_path, keep_default_na=False).to_dict("records")
        if main_path.exists()
        else []
    )
    ablation_rows = (
        pd.read_csv(ablation_path, keep_default_na=False).to_dict("records")
        if ablation_path.exists()
        else []
    )
    client = CodexBatchClient(args.model, args.reasoning_effort)
    budget = [0]
    main_rows = run_file(
        main_templates,
        main_path,
        main_rows,
        examples,
        manifest,
        client,
        args,
        budget,
    )
    copy_top3_from_main(ablation_templates, main_rows, ablation_rows)
    save(ablation_rows, ablation_path)
    ablation_rows = run_file(
        ablation_templates,
        ablation_path,
        ablation_rows,
        examples,
        manifest,
        client,
        args,
        budget,
    )

    main_df = pd.DataFrame(main_rows)
    ablation_df = pd.DataFrame(ablation_rows)
    if len(main_df) == len(main_templates) and len(ablation_df) == len(ablation_templates):
        summaries = make_summaries(main_df, ablation_df)
        summaries.to_csv(args.output_dir / "summary_tables.csv", index=False)
        write_qualitative_examples(
            main_df, args.output_dir / "qualitative_examples.md"
        )
        print(summaries.to_markdown(index=False))
    else:
        print(f"Checkpointed {len(main_df)} main and {len(ablation_df)} ablation rows.")


if __name__ == "__main__":
    main()

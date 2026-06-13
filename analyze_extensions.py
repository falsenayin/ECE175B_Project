#!/usr/bin/env python3
"""Analyze prompt, retrieval, and long-context stress extensions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from analyze_final import paired_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extensions-dir", type=Path, default=Path("outputs/extensions")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/extensions/analysis")
    )
    return parser.parse_args()


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "accuracy": ("is_correct", "mean"),
        "avg_input_tokens": ("input_tokens", "mean"),
        "avg_latency_seconds": ("response_latency_seconds", "mean"),
        "n": ("example_id", "count"),
    }
    if "target_chunks_in_top3" in frame:
        numeric = pd.to_numeric(frame["target_chunks_in_top3"], errors="coerce")
        frame = frame.copy()
        frame["target_chunks_in_top3_numeric"] = numeric
        aggregations["avg_target_chunks_in_top3"] = (
            "target_chunks_in_top3_numeric",
            "mean",
        )
        aggregations["target_document_recall"] = (
            "target_chunks_in_top3_numeric",
            lambda values: (
                (values.dropna() > 0).mean() if values.notna().any() else float("nan")
            ),
        )
    if "judge_supported" in frame:
        frame = frame.copy()
        frame["grounded_correct"] = frame["is_correct"] * frame["judge_supported"]
        aggregations["grounding_support_rate"] = ("judge_supported", "mean")
        aggregations["grounded_accuracy"] = ("grounded_correct", "mean")
    return frame.groupby("variant", as_index=False).agg(**aggregations)


def paired(frame: pd.DataFrame, baseline: str) -> pd.DataFrame:
    correct = frame.pivot(index="example_id", columns="variant", values="is_correct")
    tokens = frame.pivot(index="example_id", columns="variant", values="input_tokens")
    rows = []
    for variant in sorted(item for item in correct.columns if item != baseline):
        pair = correct[[variant, baseline]].dropna()
        a = pair[variant].to_numpy(dtype=float)
        b = pair[baseline].to_numpy(dtype=float)
        low, high = paired_bootstrap(a, b, 10000, 175)
        a_only = int(((pair[variant] == 1) & (pair[baseline] == 0)).sum())
        b_only = int(((pair[variant] == 0) & (pair[baseline] == 1)).sum())
        discordant = a_only + b_only
        token_pair = tokens[[variant, baseline]].dropna()
        rows.append(
            {
                "variant": variant,
                "baseline": baseline,
                "paired_accuracy_delta": float(np.mean(a - b)),
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
                "mcnemar_variant_only_correct": a_only,
                "mcnemar_baseline_only_correct": b_only,
                "mcnemar_exact_p_value": (
                    float(binomtest(a_only, discordant, 0.5).pvalue)
                    if discordant
                    else 1.0
                ),
                "avg_input_token_delta": float(
                    (token_pair[variant] - token_pair[baseline]).mean()
                ),
                "n": len(pair),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, path: Path, title: str) -> None:
    ordered = summary.sort_values("accuracy")
    figure, axis = plt.subplots(figsize=(10, max(4, len(ordered) * 0.42)))
    axis.barh(ordered["variant"], ordered["accuracy"])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Accuracy")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sections = ["# Extension Experiment Results", ""]
    for experiment, filename, baseline in (
        ("prompt", "results_prompt_ablation.csv", "plain"),
        ("retrieval", "results_retrieval_ablation.csv", "dense"),
        ("stress", "results_stress_test.csv", "full_6k"),
    ):
        path = args.extensions_dir / experiment / filename
        frame = pd.read_csv(path, keep_default_na=False)
        if experiment == "prompt":
            grounding_path = (
                args.extensions_dir
                / experiment
                / "grounding"
                / "grounding_judgments.csv"
            )
            if grounding_path.exists():
                grounding = pd.read_csv(grounding_path, keep_default_na=False)
                frame = frame.merge(
                    grounding[["example_id", "variant", "judge_supported"]],
                    on=["example_id", "variant"],
                    how="left",
                )
                frame["judge_supported"] = pd.to_numeric(
                    frame["judge_supported"], errors="coerce"
                )
        summary = summarize(frame)
        comparisons = paired(frame, baseline)
        summary.to_csv(args.output_dir / f"{experiment}_summary.csv", index=False)
        comparisons.to_csv(
            args.output_dir / f"{experiment}_paired_comparisons.csv", index=False
        )
        plot_summary(
            summary,
            args.output_dir / f"{experiment}_accuracy.png",
            f"{experiment.title()} Extension Accuracy",
        )
        sections.extend(
            [
                f"## {experiment.title()}",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
            ]
        )
    (args.output_dir / "extension_summary.md").write_text(
        "\n".join(sections), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

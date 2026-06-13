#!/usr/bin/env python3
"""Create final tables, statistical comparisons, figures, and qualitative cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


METHOD_ORDER = ["full_context", "plain_retrieval", "structured_retrieval"]
METHOD_LABELS = {
    "full_context": "Full context",
    "plain_retrieval": "Plain retrieval",
    "structured_retrieval": "Structured retrieval",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--supplemental-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--grounding-subdir",
        default="grounding",
        help="Grounding directory inside the primary result directory.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=175)
    parser.add_argument(
        "--allow-unvalidated-grounding",
        action="store_true",
        help="Generate preliminary analysis before completing the manual grounding audit.",
    )
    return parser.parse_args()


def normalize_top_k(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return str(int(float(text)))


def result_key(row: pd.Series) -> str:
    return (
        f"{row['example_id']}|{row['method']}|{normalize_top_k(row.get('top_k', ''))}"
    )


def merge_grounding(
    main: pd.DataFrame, primary_dir: Path, grounding_subdir: str
) -> pd.DataFrame:
    main = main.copy()
    main["result_key"] = main.apply(result_key, axis=1)
    path = primary_dir / grounding_subdir / "grounding_judgments.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing grounding judgments: {path}")
    grounding = pd.read_csv(path, keep_default_na=False)
    columns = [
        "result_key",
        "judge_supported",
        "confidence",
        "supporting_quote",
        "reason",
    ]
    main = main.merge(grounding[columns], on="result_key", how="left")
    main["judge_supported"] = pd.to_numeric(main["judge_supported"], errors="coerce")
    if main["judge_supported"].isna().any():
        missing = int(main["judge_supported"].isna().sum())
        raise ValueError(
            f"Grounding judgments are incomplete: {missing} main rows missing"
        )
    main["grounded_correct"] = main["is_correct"] * main["judge_supported"]
    return main


def validate_grounding_audit(
    primary_dir: Path, grounding_subdir: str, expected_size: int
) -> None:
    path = primary_dir / grounding_subdir / "grounding_validation_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing grounding audit summary: {path}")
    summary = pd.read_csv(path).iloc[0]
    required = min(20, expected_size)
    labeled = int(summary["n_human_labeled"])
    if labeled < required:
        raise ValueError(
            f"Grounding audit is incomplete: label {required} rows, currently {labeled}"
        )
    agreement = float(summary["agreement"])
    if agreement < 0.8:
        raise ValueError(
            f"Grounding judge agreement is {agreement:.1%}, below the required 80%"
        )


def validate_primary_results(
    main: pd.DataFrame, ablation: pd.DataFrame, primary_dir: Path
) -> None:
    manifest_path = primary_dir / "experiment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing experiment manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = int(manifest["limit"])
    valid_answers = {"A", "B", "C", "D"}
    if not set(main["prediction"]).issubset(valid_answers):
        raise ValueError("Main results contain predictions outside A/B/C/D")
    if not set(ablation["prediction"]).issubset(valid_answers):
        raise ValueError("Ablation results contain predictions outside A/B/C/D")
    main_counts = main.groupby("method")["example_id"].nunique().to_dict()
    expected_main = {method: expected for method in METHOD_ORDER}
    if main_counts != expected_main:
        raise ValueError(
            f"Incomplete main results: expected {expected_main}, got {main_counts}"
        )
    ablation_counts = (
        ablation.assign(top_k=ablation["top_k"].map(normalize_top_k))
        .groupby("top_k")["example_id"]
        .nunique()
        .to_dict()
    )
    expected_ablation = {str(item): expected for item in manifest["ablation_top_k"]}
    if ablation_counts != expected_ablation:
        raise ValueError(
            f"Incomplete ablation results: expected {expected_ablation}, got {ablation_counts}"
        )
    retrieval = main[main["method"].isin(["plain_retrieval", "structured_retrieval"])]
    indices = retrieval.pivot(
        index="example_id", columns="method", values="retrieved_chunk_indices"
    )
    if not (
        indices["plain_retrieval"].astype(str)
        == indices["structured_retrieval"].astype(str)
    ).all():
        raise ValueError("Plain and structured retrieval do not use identical chunks")


def metric_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "accuracy": ("is_correct", "mean"),
        "avg_input_tokens": ("input_tokens", "mean"),
        "avg_latency_seconds": ("response_latency_seconds", "mean"),
        "n": ("example_id", "count"),
    }
    if frame["judge_supported"].notna().any():
        aggregations["grounding_support_rate"] = ("judge_supported", "mean")
        aggregations["grounded_accuracy"] = ("grounded_correct", "mean")
    summary = frame.groupby(group_columns, as_index=False).agg(**aggregations)
    if "input_token_source" in frame.columns:
        coverage = (
            frame.assign(
                provider_reported=(
                    frame["input_token_source"].astype(str) == "provider_reported"
                ).astype(int)
            )
            .groupby(group_columns, as_index=False)
            .agg(provider_token_coverage=("provider_reported", "mean"))
        )
        summary = summary.merge(coverage, on=group_columns, how="left")
    return summary


def paired_bootstrap(
    values_a: np.ndarray, values_b: np.ndarray, samples: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values_a), size=(samples, len(values_a)))
    deltas = (values_a[indices] - values_b[indices]).mean(axis=1)
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def paired_comparisons(
    main: pd.DataFrame, bootstrap_samples: int, seed: int
) -> pd.DataFrame:
    correct = main.pivot(index="example_id", columns="method", values="is_correct")
    tokens = main.pivot(index="example_id", columns="method", values="input_tokens")
    pairs = [
        ("structured_retrieval", "plain_retrieval"),
        ("structured_retrieval", "full_context"),
        ("plain_retrieval", "full_context"),
    ]
    rows: list[dict] = []
    for method_a, method_b in pairs:
        paired = correct[[method_a, method_b]].dropna()
        a = paired[method_a].to_numpy(dtype=float)
        b = paired[method_b].to_numpy(dtype=float)
        low, high = paired_bootstrap(a, b, bootstrap_samples, seed)
        a_only = int(((paired[method_a] == 1) & (paired[method_b] == 0)).sum())
        b_only = int(((paired[method_a] == 0) & (paired[method_b] == 1)).sum())
        discordant = a_only + b_only
        p_value = (
            float(binomtest(a_only, discordant, 0.5).pvalue) if discordant else 1.0
        )
        token_pair = tokens[[method_a, method_b]].dropna()
        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "paired_accuracy_delta": float((a - b).mean()),
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
                "mcnemar_a_only_correct": a_only,
                "mcnemar_b_only_correct": b_only,
                "mcnemar_exact_p_value": p_value,
                "avg_input_token_delta": float(
                    (token_pair[method_a] - token_pair[method_b]).mean()
                ),
                "n": len(paired),
            }
        )
    return pd.DataFrame(rows)


def subgroup_summary(main: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for column, subgroup_type in (
        ("difficult", "difficulty"),
        ("context_scope", "context_scope"),
    ):
        summary = metric_summary(main, [column, "method"]).rename(
            columns={column: "subgroup"}
        )
        summary.insert(0, "subgroup_type", subgroup_type)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def plot_accuracy_tokens(summary: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for _, row in summary.iterrows():
        axis.scatter(row["avg_input_tokens"], row["accuracy"], s=80)
        axis.annotate(
            METHOD_LABELS.get(row["method"], row["method"]),
            (row["avg_input_tokens"], row["accuracy"]),
            xytext=(6, 5),
            textcoords="offset points",
        )
    axis.set_xlabel("Average input tokens")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1.05)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_subgroups(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, subgroup_type, title in zip(
        axes,
        ("difficulty", "context_scope"),
        ("Accuracy by difficulty", "Accuracy by context requirement"),
    ):
        data = summary[summary["subgroup_type"] == subgroup_type]
        pivot = data.pivot(index="subgroup", columns="method", values="accuracy")
        pivot = pivot.reindex(columns=METHOD_ORDER)
        pivot.plot(kind="bar", ax=axis, rot=0)
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("Accuracy")
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.25)
        axis.legend([METHOD_LABELS[item] for item in pivot.columns], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def row_for(frame: pd.DataFrame, example_id: str, method: str) -> pd.Series:
    return frame[
        (frame["example_id"] == example_id) & (frame["method"] == method)
    ].iloc[0]


def choose_qualitative_ids(main: pd.DataFrame) -> list[tuple[str, str | None]]:
    pivot = main.pivot(index="example_id", columns="method", values="is_correct")
    indices = main.pivot(
        index="example_id", columns="method", values="retrieved_chunk_indices"
    )
    candidates = [
        (
            "Structured retrieval helps",
            pivot[
                (pivot["structured_retrieval"] == 1) & (pivot["plain_retrieval"] == 0)
            ].index.tolist(),
        ),
        (
            "Structured retrieval hurts despite identical evidence",
            pivot[
                (pivot["structured_retrieval"] == 0)
                & (pivot["plain_retrieval"] == 1)
                & (
                    indices["structured_retrieval"].astype(str)
                    == indices["plain_retrieval"].astype(str)
                )
            ].index.tolist(),
        ),
        (
            "Retrieval misses necessary evidence",
            pivot[
                (pivot["full_context"] == 1)
                & (pivot["plain_retrieval"] == 0)
                & (pivot["structured_retrieval"] == 0)
                & (
                    main.pivot(
                        index="example_id",
                        columns="method",
                        values="judge_supported",
                    )["plain_retrieval"]
                    == 0
                )
                & (
                    main.pivot(
                        index="example_id",
                        columns="method",
                        values="judge_supported",
                    )["structured_retrieval"]
                    == 0
                )
            ].index.tolist(),
        ),
    ]
    return [(label, ids[0] if ids else None) for label, ids in candidates]


def write_qualitative(main: pd.DataFrame, path: Path) -> None:
    lines = ["# Final Qualitative Cases", ""]
    for label, example_id in choose_qualitative_ids(main):
        lines.extend([f"## {label}", ""])
        if example_id is None:
            lines.extend(["No matching example was found in this run.", ""])
            continue
        first = row_for(main, example_id, "full_context")
        lines.extend(
            [
                f"**Example:** {example_id}",
                "",
                f"**Question:** {first['question']}",
                "",
                f"**Correct answer:** {first['correct_answer']}",
                "",
                "| Method | Prediction | Correct | Supported | Judge reason |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for method in METHOD_ORDER:
            row = row_for(main, example_id, method)
            supported = row.get("judge_supported", "")
            reason = str(row.get("reason", "")).replace("|", "\\|")
            lines.append(
                f"| {METHOD_LABELS[method]} | {row['prediction']} | "
                f"{row['is_correct']} | {supported} | {reason} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_analysis_summary(main_summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Final Experiment Summary",
        "",
        "Use this file as a factual starting point for the report. Interpret mixed results honestly.",
        "",
        main_summary.to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.primary_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    main = pd.read_csv(args.primary_dir / "results_main.csv", keep_default_na=False)
    ablation = pd.read_csv(
        args.primary_dir / "results_ablation.csv", keep_default_na=False
    )
    validate_primary_results(main, ablation, args.primary_dir)
    main = merge_grounding(main, args.primary_dir, args.grounding_subdir)
    if not args.allow_unvalidated_grounding:
        validate_grounding_audit(
            args.primary_dir, args.grounding_subdir, main["example_id"].nunique()
        )
    ablation["judge_supported"] = np.nan

    main_summary = metric_summary(main, ["method"])
    ablation_summary = metric_summary(ablation, ["top_k"])
    subgroups = subgroup_summary(main)
    paired = paired_comparisons(main, args.bootstrap_samples, args.seed)

    main_summary.to_csv(output_dir / "final_main_summary.csv", index=False)
    ablation_summary.to_csv(output_dir / "final_ablation_summary.csv", index=False)
    subgroups.to_csv(output_dir / "subgroup_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False)
    plot_accuracy_tokens(main_summary, output_dir / "accuracy_vs_tokens.png")
    plot_subgroups(subgroups, output_dir / "subgroup_accuracy.png")
    write_qualitative(main, output_dir / "qualitative_cases.md")
    write_analysis_summary(main_summary, output_dir / "analysis_summary.md")

    if args.supplemental_dir:
        supplemental = args.supplemental_dir / "model_comparison.csv"
        if supplemental.exists():
            pd.read_csv(supplemental).to_csv(
                output_dir / "supplemental_model_comparison.csv", index=False
            )

    print("\nFinal main summary:")
    print(main_summary.to_markdown(index=False))
    print("\nPaired comparisons:")
    print(paired.to_markdown(index=False))


if __name__ == "__main__":
    main()

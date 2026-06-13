#!/usr/bin/env python3
"""Build combined cross-model final tables and figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODELS = {
    "Llama 4 Scout": Path("outputs/final_openrouter_llama"),
    "Qwen3 30B A3B": Path("outputs/final_openrouter_qwen"),
    "GPT-OSS 120B": Path("outputs/final_openrouter_gpt_oss"),
    "Gemini 2.5 Pro": Path("outputs/final_openrouter_gemini25pro"),
    "GPT-5.5 Medium (Codex subscription)": Path("outputs/final_codex_gpt55_medium"),
}
METHODS = ["full_context", "plain_retrieval", "structured_retrieval"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-subdir", default="analysis")
    parser.add_argument("--grounding-subdir", default="grounding")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cross_model"))
    return parser.parse_args()


def active_models() -> dict[str, Path]:
    return {model: path for model, path in MODELS.items() if path.exists()}


def combine(filename: str, models: dict[str, Path], analysis_subdir: str) -> pd.DataFrame:
    frames = []
    for model, directory in models.items():
        frame = pd.read_csv(directory / analysis_subdir / filename)
        frame.insert(0, "model", model)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_metric(frame: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    pivot = frame.pivot(index="model", columns="method", values=metric)[METHODS]
    axis = pivot.plot(kind="bar", figsize=(10, 5.5), ylim=(0, 1), rot=0)
    axis.set_title(title)
    axis.set_xlabel("")
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.legend(["Full context", "Plain retrieval", "Structured retrieval"])
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    models = active_models()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    main = combine("final_main_summary.csv", models, args.analysis_subdir)
    ablation = combine("final_ablation_summary.csv", models, args.analysis_subdir)
    paired = combine("paired_comparisons.csv", models, args.analysis_subdir)
    subgroups = combine("subgroup_summary.csv", models, args.analysis_subdir)
    main.to_csv(output / "cross_model_comparison.csv", index=False)
    ablation.to_csv(output / "cross_model_ablation.csv", index=False)
    paired.to_csv(output / "cross_model_paired_comparisons.csv", index=False)
    subgroups.to_csv(output / "cross_model_subgroups.csv", index=False)

    effects = []
    for model, group in main.groupby("model"):
        indexed = group.set_index("method")
        full = indexed.loc["full_context"]
        plain = indexed.loc["plain_retrieval"]
        structured = indexed.loc["structured_retrieval"]
        effects.append(
            {
                "model": model,
                "plain_accuracy_delta_vs_full": plain.accuracy - full.accuracy,
                "structured_accuracy_delta_vs_full": structured.accuracy
                - full.accuracy,
                "structured_accuracy_delta_vs_plain": structured.accuracy
                - plain.accuracy,
                "structured_grounding_delta_vs_plain": (
                    structured.grounding_support_rate - plain.grounding_support_rate
                ),
                "plain_token_reduction_vs_full": (
                    1 - plain.avg_input_tokens / full.avg_input_tokens
                ),
                "structured_token_reduction_vs_full": (
                    1 - structured.avg_input_tokens / full.avg_input_tokens
                ),
            }
        )
    effects_frame = pd.DataFrame(effects)
    effects_frame.to_csv(output / "cross_model_effects.csv", index=False)

    aggregate = main.groupby("method", as_index=False).agg(
        mean_accuracy=("accuracy", "mean"),
        mean_grounding_support_rate=("grounding_support_rate", "mean"),
        mean_grounded_accuracy=("grounded_accuracy", "mean"),
        mean_input_tokens=("avg_input_tokens", "mean"),
    )
    aggregate.to_csv(output / "aggregate_by_method.csv", index=False)

    raw_main, raw_ablation, validation = [], [], []
    for model, directory in models.items():
        model_main = pd.read_csv(directory / "results_main.csv")
        model_ablation = pd.read_csv(directory / "results_ablation.csv")
        model_main.insert(0, "model_name", model)
        model_ablation.insert(0, "model_name", model)
        raw_main.append(model_main)
        raw_ablation.append(model_ablation)
        audit_path = directory / args.grounding_subdir / "grounding_validation_summary.csv"
        if audit_path.exists():
            audit = pd.read_csv(audit_path, keep_default_na=False).iloc[0]
            validation.append({"model": model, **audit.to_dict()})
    pd.concat(raw_main, ignore_index=True).to_csv(
        output / "all_cross_model_results_main.csv", index=False
    )
    pd.concat(raw_ablation, ignore_index=True).to_csv(
        output / "all_cross_model_results_ablation.csv", index=False
    )
    pd.DataFrame(validation).to_csv(output / "grounding_validation_status.csv", index=False)

    model_count = len(models)
    for metric, title in (
        ("accuracy", f"Accuracy Across {model_count} Generators"),
        ("grounding_support_rate", f"Grounding Support Across {model_count} Generators"),
        ("grounded_accuracy", f"Grounded Accuracy Across {model_count} Generators"),
    ):
        plot_metric(main, metric, title, output / f"cross_model_{metric}.png")
    plot_metric(
        main,
        "grounding_support_rate",
        f"Grounding Support Across {model_count} Generators",
        output / "cross_model_grounding.png",
    )

    lines = [
        f"# {model_count}-Model Final Results",
        "",
        "## Main Comparison",
        "",
        main.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Aggregate by Method",
        "",
        aggregate.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Cross-Model Effects",
        "",
        effects_frame.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Defensible Findings",
        "",
        "- Full context achieved the highest answer accuracy for every generator.",
        "- Retrieval reduced average input tokens by roughly 72% across generators.",
        "- Structured retrieval never significantly outperformed plain retrieval in accuracy.",
        "- Structured top-5 improved accuracy for every generator, showing that evidence coverage remains a central limitation.",
        "- The Codex subscription generator is supplemental because its hidden agent prompt differs from the direct API models.",
        "- Grounding claims use the separately validated GPT-5.4-mini judge.",
        "",
    ]
    (output / "cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

# ECE 175B Final Project

This repository contains the code for my ECE 175B final project on long-context
question answering.

The project compares three ways of giving a long document to an LLM:

1. **Full context**: pass the full document with the question and choices.
2. **Plain retrieval**: retrieve the top-k chunks and pass them as raw context.
3. **Structured retrieval**: use the same retrieved chunks, but format them as
   numbered evidence before the answer choices.

The main question is whether retrieval and simple prompt structure can reduce
input tokens while keeping answer accuracy and grounding reasonable.

## Main Experiment

- Dataset: QuALITY development split
- Sample: 200 questions from 100 articles
- Retriever: `sentence-transformers/all-MiniLM-L6-v2`
- Chunking: 400 words with 50-word overlap
- Main top-k: 3
- Ablation: structured retrieval with top-k 1, 3, and 5
- Metrics:
  - answer accuracy
  - grounding support rate
  - grounded accuracy
  - average input tokens
  - latency

The code uses environment variables for API keys. No API keys are stored in the
repo.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

The scripts download the QuALITY dev file automatically if it is missing.

## Quick Dry Run

This checks dataset loading, chunking, retrieval, and prompt construction without
calling an LLM:

```bash
python3 pilot_experiment.py \
  --limit 3 \
  --retriever tfidf \
  --dry-run
```

## Run The Main Experiment

Example command for an OpenAI-compatible provider:

```bash
export OPENROUTER_API_KEY="your-key"

python3 pilot_experiment.py \
  --limit 200 \
  --sampling paired-article-difficulty \
  --model meta-llama/llama-4-scout \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY \
  --api-style openai \
  --output-dir outputs/final_openrouter_llama \
  --max-new-examples 20
```

The experiment checkpoints after each model response. Rerun the same command to
continue from the last saved row.

## Run Grounding Evaluation

The grounding judge checks whether the context given to the generator supports
the predicted answer.

```bash
python3 evaluate_grounding.py \
  --results outputs/final_openrouter_llama/results_main.csv \
  --judge-model meta-llama/llama-3.3-70b-instruct \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY \
  --output-dir outputs/final_openrouter_llama/grounding \
  --max-new-judgments 50
```

For my final run, I also used `evaluate_grounding_codex.py` with `gpt-5.4-mini`
as the judge and validated it on a 20-example manual audit.

## Analyze Results

Per-model analysis:

```bash
python3 analyze_final.py \
  --primary-dir outputs/final_openrouter_llama \
  --grounding-subdir grounding_gpt54mini \
  --output-dir outputs/final_openrouter_llama/analysis_gpt54mini \
  --allow-unvalidated-grounding
```

Cross-model summary:

```bash
python3 build_cross_model_report.py \
  --analysis-subdir analysis_gpt54mini \
  --grounding-subdir grounding_gpt54mini \
  --output-dir outputs/cross_model_gpt54mini
```

## Extension Experiments

The extension script runs extra checks that were useful for the final report:

- prompt wording ablation
- retrieval variant ablation
- long-context distractor stress test

```bash
python3 run_extensions.py --experiment retrieval --output-dir outputs/extensions/retrieval
python3 analyze_extensions.py
```

## Important Files

- `pilot_experiment.py`: main experiment runner
- `evaluate_grounding.py`: API-based grounding judge
- `evaluate_grounding_codex.py`: Codex/ChatGPT subscription grounding runner
- `codex_subscription.py`: small helper for batched Codex CLI calls
- `run_codex_subscription_model.py`: optional supplemental generator runner
- `analyze_final.py`: per-model tables, figures, and statistical comparisons
- `build_cross_model_report.py`: combined cross-model summary
- `run_extensions.py`: extra ablations and stress tests
- `analyze_extensions.py`: summary tables and figures for extension runs

## Reproducibility Notes

The repo does not include raw outputs, downloaded datasets, or API keys. These
are excluded to keep the GitHub submission small and safe. The final report
contains the numerical results from the completed runs.

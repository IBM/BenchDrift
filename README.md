# BenchDrift

**Struggling to find edge cases for your prompts? Wondering why different models interpret the same problem differently?**

BenchDrift automatically generates syntactic variations of your test problems (preserving semantic meaning) to reveal hidden model brittleness, discover edge cases, and detect performance drift — helping you debug failing tasks and validate model robustness.

## What is BenchDrift?

BenchDrift generates syntactic variations of test problems (same meaning, different presentation) and detects when model performance changes unexpectedly (drift detection).

**Example:** Your model solves "What is 15 + 25?" correctly but fails on "Calculate the sum of 15 and 25" — BenchDrift finds these inconsistencies automatically. Use it to discover edge cases, test robustness, compare models, debug failures, and validate prompts.

## Demo

https://github.com/user-attachments/assets/76effad5-44b8-445c-9eef-06b8d6bfd90c

## Quick Start

```bash
# 1. Install
pip install -e .
python -m spacy download en_core_web_sm

# 2. Install & start Ollama (https://ollama.com)
ollama pull qwen3:8b
ollama pull mistral:7b

# 3. Test drift on a single problem
python benchdrift_cli.py \
  --problem "A store sells apples for \$2 each. John buys 3. How much?" \
  --answer "6" \
  --gen-model qwen3:8b \
  --target-model mistral:7b \
  --top-k 10

# 4. Launch the interactive UI
pip install gradio
python app.py
```

For detailed setup (cloud backends, batch experiments, configuration), see [docs/RUNNING_THE_PIPELINE.md](docs/RUNNING_THE_PIPELINE.md).

## How It Works

BenchDrift uses a **variation taxonomy** developed from analyzing drifts observed across benchmark problems in multiple domains: math, natural language, temporal reasoning, long context, logical propositions, and more. This taxonomy guides the generation of surface-form variations (preserving meaning) that are most likely to reveal model brittleness.

**7 axes, 63 transformations** across linguistic, referential, pragmatic, structural, persona, long-context, and constraint-targeted dimensions. Per-problem relevance selection ranks the most impactful transformations automatically.

## Pipeline Stages

1. **Variations** - Generate syntactic variations of input problems (same meaning, different form)
2. **Validation** - Validate that variations are equivalent to originals
3. **Responses** - Get model responses for all variations
4. **Evaluation** - Detect positive/negative drift in model performance

## Documentation

- **[docs/INSTALLATION.md](docs/INSTALLATION.md)** — Detailed setup guide
- **[docs/RUNNING_THE_PIPELINE.md](docs/RUNNING_THE_PIPELINE.md)** — CLI, batch pipeline, interactive app, all configuration options
- **[docs/LONG_CONTEXT_VARIATIONS_README.md](docs/LONG_CONTEXT_VARIATIONS_README.md)** — Long context variation types

## Repository Structure

```
BenchDrift/
├── src/benchdrift/        # Installable package (pip install -e .)
│   ├── pipeline/          # Pipeline stages & variation engines
│   ├── models/            # Model clients (Ollama, RITS, Groq, vLLM)
│   └── eval/              # Evaluation & visualization
├── app/                   # Gradio interactive app modules
├── app.py                 # App entry point
├── benchdrift_cli.py      # Single-problem CLI tool
├── data/                  # Benchmark datasets (gsm8k, mmlu, math-hard)
├── config/                # Model & taxonomy configurations
├── scripts/               # Batch experiment runner scripts
├── docs/                  # Documentation
├── figures/               # System overview & result figures
├── notebooks/             # Demo notebooks
└── logs/                  # Pipeline logs
```

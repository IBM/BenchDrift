# BenchDrift Installation Guide

## Package Structure

```
BenchDrift/
├── src/benchdrift/        # Installable Python package
│   ├── pipeline/          # Pipeline stages & variation engines
│   ├── models/            # Model clients (Ollama, RITS, Groq, vLLM)
│   └── eval/              # Evaluation & visualization
├── app/                   # Gradio interactive app modules
├── app.py                 # App entry point
├── benchdrift_cli.py      # Single-problem CLI tool
├── data/                  # Benchmark datasets
├── config/                # Model & taxonomy configurations
├── pyproject.toml         # Package configuration
└── setup.py               # Setup script
```

## Prerequisites

### Ollama (recommended for local models)
```bash
# Install from https://ollama.com
ollama pull qwen3:8b       # Generator/judge model
ollama pull mistral:7b     # Target model
```

### Optional cloud backends
```bash
export RITS_API_KEY='...'   # IBM RITS cluster
export GROQ_API_KEY='...'   # Groq cloud API
```

## Installation

### Option 1: Install as Package (Recommended)

```bash
cd BenchDrift
pip install -e .
```

After installation, you can import from anywhere:
```python
from benchdrift.pipeline.unified_batched_pipeline_semantic import UnifiedBatchedPipeline
from benchdrift.eval.comprehensive_results_visualizer import visualize_results
```

### Option 2: Use Without Installation

For notebooks, add this at the top:
```python
import sys
import os
sys.path.insert(0, os.path.abspath('../src'))
```

For scripts, set PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## Verification

Test the installation:
```bash
python3 -c "from benchdrift.pipeline.unified_batched_pipeline_semantic import UnifiedBatchedPipeline; print('Working!')"
```

## Quick Usage

```bash
# Single-problem CLI (Ollama required)
python benchdrift_cli.py --problem "What is 15 + 25?" --answer "40" --gen-model qwen3:8b

# Interactive UI
python app.py

# Batch pipeline
python -m benchdrift.pipeline.unified_batched_pipeline_semantic \
  --unified-file results.json --input data/gsm8k/test_500.jsonl \
  --all-stages --model-name ollama/qwen3:8b --response-model ollama/mistral:7b
```

See [RUNNING_THE_PIPELINE.md](RUNNING_THE_PIPELINE.md) for full usage guide.

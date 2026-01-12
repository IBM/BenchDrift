# BenchDrift Installation Guide

## Package Structure

```
BenchDrift-Pipeline/
├── src/
│   ├── pipeline/      # Pipeline modules
│   ├── models/        # Model clients
│   └── eval/          # Evaluation modules
├── notebooks/         # Jupyter notebooks
├── data/             # Data files
├── config/           # Configuration
├── pyproject.toml    # Package configuration
└── setup.py          # Setup script
```

## Installation

### Option 1: Install as Package (Recommended)

```bash
cd BenchDrift-Pipeline
pip install -e .
```

After installation, you can import from anywhere:
```python
from pipeline.unified_batched_pipeline_semantic import UnifiedBatchedPipeline
from eval.comprehensive_results_visualizer import visualize_results
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
python3 -c "from pipeline.unified_batched_pipeline_semantic import UnifiedBatchedPipeline; print('✅ Working!')"
```

## Usage

All shell scripts automatically set PYTHONPATH. Just run:
```bash
./example_semantic_pipeline.sh
```

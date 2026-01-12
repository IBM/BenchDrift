# Running the Semantic Pipeline

Complete guide for running the BenchDrift semantic clustering pipeline.

## Prerequisites

Set your API keys:
```bash
# For RITS (required for response generation - target model)
export RITS_API_KEY='your_rits_api_key_here'

# For Gemini (optional - for variation/validation/evaluation only)
export GEMINI_API_KEY='your_gemini_api_key_here'
```

**Note:** Gemini can be used for variation generation, validation, and evaluation. Response generation (target model) always uses RITS.

---

## Method 1: Jupyter Notebook (Interactive)

**Best for:** Exploration and visualization

```bash
jupyter notebook demo_semantic.ipynb
```

Run all cells for full pipeline execution with built-in visualizations.

---

## Method 2: Shell Script (Quick Run)

**Best for:** Simple end-to-end runs

### Basic run:
```bash
./run_semantic_pipeline.sh
```

### Custom settings:
```bash
./run_semantic_pipeline.sh \
  --input my_problems.json \
  --output results.json \
  --batch-size 100
```

### Enable long context:
```bash
./run_semantic_pipeline.sh \
  --use-long-context \
  --semantic-threshold 0.4
```

### Help:
```bash
./run_semantic_pipeline.sh --help
```

---

## Method 3: Python Script (Full Pipeline)

**Best for:** Integration and automation

```python
from unified_batched_pipeline_semantic import UnifiedBatchedPipeline

config = {
    'unified_file': 'output.json',
    'input_problems': 'problems.json',
    'batch_size': 50,
    'client_type': 'rits',  # For variation/validation/evaluation
    'model_name': 'phi-4',
    'response_model': 'granite-3-3-8b',
    'response_client_type': 'rits',  # Target model (always rits)
    'use_llm_judge': True,
    'judge_model': 'llama_3_3_70b',
    'embedding_model': 'all-MiniLM-L6-v2',
    'semantic_threshold': 0.35,
    'use_generic': True,
    'use_cluster_variations': True,
}

pipeline = UnifiedBatchedPipeline(config)
pipeline.stage1_generate_variations_batched()
pipeline.stage_validation()
pipeline.stage2_generate_responses()
pipeline.stage3_add_evaluation_metrics()
```

---

## Method 4: Command-Line (Direct Pipeline Script)

**Best for:** Individual stage control and scripting

### Run all stages:
```bash
python unified_batched_pipeline_semantic.py \
  --unified-file output.json \
  --input problems.json \
  --all-stages \
  --batch-size 50
```

### Run specific stage:

**Stage 1 - Generate variations:**
```bash
python unified_batched_pipeline_semantic.py \
  --unified-file output.json \
  --input problems.json \
  --stage variations \
  --batch-size 50 \
  --use-generic \
  --use-cluster-variations
```

**Stage 2 - Validate variations:**
```bash
python unified_batched_pipeline_semantic.py \
  --unified-file output.json \
  --stage validation \
  --use-llm-judge \
  --judge-model llama_3_3_70b
```

**Stage 3 - Generate responses:**
```bash
python unified_batched_pipeline_semantic.py \
  --unified-file output.json \
  --stage responses \
  --response-model granite-3-3-8b \
  --batch-size 30
```

**Stage 4 - Evaluate:**
```bash
python unified_batched_pipeline_semantic.py \
  --unified-file output.json \
  --stage evaluation \
  --use-llm-judge \
  --judge-model llama_3_3_70b
```

**Export CSV:**
```bash
python unified_batched_pipeline_semantic.py \
  --unified-file output.json \
  --stage export-csv
```

### Advanced stage control:

**Run stages 1-4 (skip candidate detection):**
```bash
python unified_batched_pipeline_semantic.py \
  --unified-file output.json \
  --input problems.json \
  --stages-1-4 \
  --batch-size 100
```

**Resume from specific stage:**
```bash
# Stage 1-2 already complete, continue from stage 3
python unified_batched_pipeline_semantic.py \
  --unified-file output.json \
  --stage responses \
  --response-model granite-3-3-8b

python unified_batched_pipeline_semantic.py \
  --unified-file output.json \
  --stage evaluation \
  --use-llm-judge
```

---

## Configuration Options

### Core Settings
```bash
--unified-file FILE          # Output JSON file (required)
--input FILE                 # Input problems file (required for variations stage)
--batch-size N               # Processing batch size (default: 50)
--max-workers N              # Parallel workers (default: 4)
--max-problems N             # Limit number of problems to process
```

### Debug Levels

**Three levels of output:**

1. **Default (Clean)** - High-level progress only:
   ```bash
   python unified_batched_pipeline_semantic.py --all-stages --input problems.json
   ```
   Output:
   ```
   🔄 Stage 1: Generating Variations...
   Processing batches: 100%|████████| 10/10 [02:30<00:00]
   ✅ Stage 1 complete: Generated 500 entries
   ```

2. **Verbose (Detailed)** - All debug information:
   ```python
   # In Python config
   config = {'verbose': True, ...}
   ```
   Output:
   ```
   🔄 Stage 1: Generating Variations...
   📦 Processing batch 1/10 (50 problems)
      🔍 DEBUG: Candidate Embeddings
      [0] 'morning' @ (10, 17)
      ...
   ```

3. **Log File (Complete)** - Everything saved automatically:
   ```
   pipeline_debug.log  # Created automatically, contains all debug output
   ```

**Quick start:**
- Default: Just run the pipeline (clean output)
- Debug: Add `'verbose': True` to config (full debug output)
- Logs: Check `pipeline_debug.log` if something goes wrong

### Model Settings
```bash
--client-type TYPE           # API client for variation/validation/evaluation: rits, vllm, gemini (default: rits)
--model-name MODEL           # Variation generation model (default: mistral_small_3_2_instruct)
--response-model MODEL       # Response generation model - always uses RITS (default: mistral_small_3_2_instruct)
--judge-model MODEL          # LLM judge model (can use gemini if client-type=gemini)
--max-model-len N            # Max context length for VLLM (default: 8192)
--max-tokens N               # Max output tokens (default: 1024)
--temperature N              # Temperature for generation (default: 0.1)
```

**Note:** `--client-type` applies to variation/validation/evaluation stages only. Response generation always uses RITS to ensure the target model is not affected.

### Semantic Clustering
```bash
--embedding-model MODEL      # Embedding model (default: all-MiniLM-L6-v2)
--semantic-threshold N       # Clustering threshold (default: 0.35)
```

### Variation Types
```bash
--use-generic                # Generic transformations (default: True)
--no-generic                 # Disable generic transformations
--use-cluster-variations     # Cluster-based variations (default: True)
--no-cluster-variations      # Disable cluster variations
--use-persona                # Persona variations (default: False)
--use-long-context           # Long context variations for >500 char prompts (default: False)
```

### Evaluation
```bash
--use-llm-judge              # Use LLM judge for evaluation
--rectify-invalid            # Rectify invalid variations instead of dropping
--force-regenerate           # Force regenerate responses
--disable-cot                # Disable chain-of-thought reasoning
```

---

## Example Workflows

### Quick test:
```bash
python unified_batched_pipeline_semantic.py \
  --unified-file test_output.json \
  --input demo_problems.json \
  --all-stages \
  --batch-size 10 \
  --use-generic \
  --use-cluster-variations
```

### Production run with all variation types:
```bash
python unified_batched_pipeline_semantic.py \
  --unified-file production.json \
  --input problems.json \
  --all-stages \
  --batch-size 100 \
  --max-workers 8 \
  --use-generic \
  --use-cluster-variations \
  --use-persona \
  --use-long-context \
  --use-llm-judge \
  --judge-model llama_3_3_70b
```

### Custom semantic threshold:
```bash
python unified_batched_pipeline_semantic.py \
  --unified-file output.json \
  --input problems.json \
  --stage variations \
  --semantic-threshold 0.4 \
  --embedding-model all-mpnet-base-v2
```

### Using Gemini for variation/validation/evaluation:
```bash
# Gemini for variation generation and evaluation, RITS for target model responses
python unified_batched_pipeline_semantic.py \
  --unified-file output.json \
  --input problems.json \
  --all-stages \
  --client-type gemini \
  --model-name gemini-2.0-flash-exp \
  --judge-model gemini-2.0-flash-exp \
  --response-model granite-3-3-8b \
  --use-llm-judge
```

### Debug single stage:
```bash
# Run only validation stage
python unified_batched_pipeline_semantic.py \
  --unified-file output.json \
  --stage validation \
  --rectify-invalid
```

---

## Visualization

After running the pipeline:
```python
from comprehensive_results_visualizer import visualize_results
import pandas as pd
import json

with open('output.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
viz = visualize_results(df)
```

Creates 4-panel visualization with drift analysis.

---

## Calibrating Thresholds

Find optimal clustering threshold for your data:
```bash
python calibrate_semantic_thresholds.py \
  --input problems.json \
  --output calibration.json
```

---

## Input Format

Problems in JSON format:
```json
[
  {
    "problem": "Question text...",
    "answer": "Expected answer"
  }
]
```


## Output Format

```json
{
  "config": {
    "target_model": "phi-4",
    "composite_method": "semantic",
    "use_generic_variations": true,
    ...
  },
  "stats": {
    "total_problems": 10,
    "total_segments": 45,
    "total_dependencies": 23,
    "total_variations_generated": 100,
    "total_variations_verified": 87,
    "total_llm_calls": 65,
    "timing": {
      "stage1": 12.5,
      "stage2": 45.2,
      "stage3": 23.1,
      ...
    }
  },
  "data": [
    {
      "problem_idx": 0,
      "original_problem": "What is 15 + 25?",
      "expected_answer": "40",
      "candidates": [...],
      "segments": [...],
      "linked_groups": [...],
      "dependency_graph": {...},
      "variations": [
        {
          "type": "counterfactual",
          "variation": "If you had 15 apples and received 25 more...",
          "target_response": "40",
          "answer_preserved": true,
          "judge_verdict": "yes"
        },
        ...
      ]
    },
    ...
  ]
}
```

---

## Stage Dependencies

1. **Variations** - Generates variations from input problems
2. **Validation** - Validates all variations (depends on: variations)
3. **Responses** - Generates model responses (depends on: validation)
4. **Evaluation** - Evaluates drift (depends on: responses)
5. **Export CSV** - Exports results (depends on: evaluation)

Run stages in order or use `--all-stages` to run all sequentially.

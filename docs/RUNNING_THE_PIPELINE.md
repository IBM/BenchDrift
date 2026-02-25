# Running the BenchDrift Pipeline

Complete guide for running BenchDrift — variation generation, validation, response collection, and drift evaluation.

## Prerequisites

### Ollama (recommended for local models)
```bash
# Install from https://ollama.com
ollama pull qwen3:8b       # Generator/judge model
ollama pull mistral:7b     # Target model
```

### RITS (IBM cluster)
```bash
export RITS_API_KEY='your_rits_api_key_here'
```

### Groq (cloud API)
```bash
export GROQ_API_KEY='your_groq_api_key_here'
```

---

## Model Specification Format

All `--model-name`, `--response-model`, and `--judge-model` flags accept **client/model** format:

```bash
--model-name ollama/qwen3:8b          # Ollama + qwen3:8b
--response-model rits/granite-3-3-8b  # RITS + granite
--judge-model groq/llama-3.3-70b      # Groq + llama

# Or use --client-type with bare model names (backward compatible):
--client-type ollama --model-name qwen3:8b
```

When using the `client/model` format, `--client-type` is optional — the client is inferred from the prefix. The generator model's client becomes the pipeline-wide default.

---

## Method 1: Single-Problem CLI (Quick Testing)

```bash
# Feature analysis only (instant, no LLM)
python benchdrift_cli.py \
  --problem "A store sells apples for \$2 each. John buys 3. How much?" \
  --answer "6" \
  --no-enrich --no-generate

# Generate variations (Ollama required)
python benchdrift_cli.py \
  --problem "A store sells apples for \$2 each. John buys 3. How much?" \
  --answer "6" \
  --gen-model qwen3:8b \
  --top-k 10

# Full drift test (baseline + variations against target model)
python benchdrift_cli.py \
  --problem "A store sells apples for \$2 each. John buys 3. How much?" \
  --answer "6" \
  --gen-model qwen3:8b \
  --target-model mistral:7b \
  --top-k 10

# JSON output
python benchdrift_cli.py \
  --problem "What is 15 + 25?" --answer "40" \
  --gen-model qwen3:8b --target-model mistral:7b \
  --top-k 5 --json --output results.json

# Read from stdin
echo "What is 15 + 25?" | python benchdrift_cli.py --answer "40" --gen-model qwen3:8b
```

---

## Method 2: Batch Pipeline (Full Experiments)

### Unified Batched Pipeline (primary — `unified_batched_pipeline_semantic.py`)

Run as a Python module from the project root:

```bash
cd /path/to/BenchDrift-Pipeline-v2-AppRefactor
```

#### Stage 1: Generate Variations
```bash
python -m benchdrift.pipeline.unified_batched_pipeline_semantic \
  --unified-file experiments/gsm8k_results.json \
  --input data/gsm8k/test_500.jsonl \
  --stage variations \
  --batch-size 50 \
  --client-type ollama \
  --model-name qwen3:8b \
  --use-relevance-selection \
  --relevance-top-k 10 \
  --num-variations 3 \
  --temperature 0.3 \
  --max-tokens 1024 \
  --save-every-batch

# Or with client/model format (no --client-type needed):
python -m benchdrift.pipeline.unified_batched_pipeline_semantic \
  --unified-file experiments/gsm8k_results.json \
  --input data/gsm8k/test_500.jsonl \
  --stage variations \
  --batch-size 50 \
  --model-name ollama/qwen3:8b \
  --use-relevance-selection \
  --relevance-top-k 10
```

#### Stage 2: Validate Variations
```bash
python -m benchdrift.pipeline.unified_batched_pipeline_semantic \
  --unified-file experiments/gsm8k_results.json \
  --stage validation \
  --client-type ollama \
  --model-name qwen3:8b

# Skip validation entirely:
python -m benchdrift.pipeline.unified_batched_pipeline_semantic \
  --unified-file experiments/gsm8k_results.json \
  --stage validation \
  --validation-method none

# Use council validation (multi-judge via OpenRouter):
python -m benchdrift.pipeline.unified_batched_pipeline_semantic \
  --unified-file experiments/gsm8k_results.json \
  --stage validation \
  --validation-method council \
  --council-models "openai/gpt-4o-mini,anthropic/claude-3-haiku,google/gemini-flash-1.5" \
  --openrouter-api-key "$OPENROUTER_API_KEY"
```

#### Stage 3: Collect Target Model Responses
```bash
python -m benchdrift.pipeline.unified_batched_pipeline_semantic \
  --unified-file experiments/gsm8k_results.json \
  --stage responses \
  --client-type ollama \
  --response-model mistral:7b

# Or mixed backends:
python -m benchdrift.pipeline.unified_batched_pipeline_semantic \
  --unified-file experiments/gsm8k_results.json \
  --stage responses \
  --response-model rits/granite-3-3-8b
```

#### Stage 4: Evaluate
```bash
# String matching (default, fast):
python -m benchdrift.pipeline.unified_batched_pipeline_semantic \
  --unified-file experiments/gsm8k_results.json \
  --stage evaluation \
  --client-type ollama \
  --model-name qwen3:8b

# LLM judge (more accurate for ambiguous answers):
python -m benchdrift.pipeline.unified_batched_pipeline_semantic \
  --unified-file experiments/gsm8k_results.json \
  --stage evaluation \
  --use-llm-judge \
  --judge-model ollama/qwen3:8b
```

#### All Stages at Once
```bash
python -m benchdrift.pipeline.unified_batched_pipeline_semantic \
  --unified-file experiments/gsm8k_results.json \
  --input data/gsm8k/test_500.jsonl \
  --all-stages \
  --batch-size 50 \
  --model-name ollama/qwen3:8b \
  --response-model ollama/mistral:7b \
  --use-relevance-selection \
  --relevance-top-k 10

# Quick test with 10 problems:
python -m benchdrift.pipeline.unified_batched_pipeline_semantic \
  --unified-file experiments/test_10.json \
  --input data/gsm8k/test_500.jsonl \
  --all-stages \
  --batch-size 10 \
  --max-problems 10 \
  --model-name ollama/qwen3:8b \
  --response-model ollama/mistral:7b
```

### Complete Variation Pipeline (simpler, progressive)

```bash
python -m benchdrift.pipeline.complete_variation_pipeline \
  --unified-file experiments/gsm8k_progressive.json \
  --input data/gsm8k/test_500.jsonl \
  --all-stages \
  --client-type ollama \
  --model-name qwen3:8b \
  --eval-model mistral:7b \
  --batch-size 50 \
  --num-variations 3
```

---

## Method 3: Interactive App (Gradio)

```bash
python app.py              # Local at http://localhost:7860
python app.py --share      # Public URL
python app.py --port 7861  # Custom port
```

---

## Configuration Reference

### Model Settings
```bash
--client-type TYPE           # rits, ollama, ollama_logits, vllm, vllm_logits, groq, openai
--model-name MODEL           # Generator model (accepts client/model format)
--response-model MODEL       # Target model for responses (accepts client/model format)
--judge-model MODEL          # Judge model for evaluation (accepts client/model format)
--max-model-len N            # Max context length for VLLM (default: 8192)
--max-tokens N               # Max output tokens (default: 1024)
--temperature N              # Temperature for generation (default: 0.1)
--max-workers N              # Parallel workers (default: 4)
```

### Batching
```bash
--batch-size N               # Processing batch size (default: 50)
--max-problems N             # Limit number of problems
--save-every-batch           # Save after every batch (default: True)
```

### Variation Control
```bash
--use-axes AXES              # Comma-separated taxonomy axes to enable
                             # Valid: linguistic,referential,pragmatic,structural,
                             #        persona,long_context,constraint_targeted,all
                             # Subtract with minus: "all,-persona"
                             # Default: linguistic,referential,pragmatic,structural,constraint_targeted
--num-variations N           # Number of variations per problem (default: 3)
--use-relevance-selection    # Rank transformations by relevance per problem
--relevance-top-k N          # Top-k ranked transformations to generate (default: 10)
--use-cluster-variations     # Cluster-based variations (default: True)
--no-cluster-variations      # Disable cluster variations
```

### Validation
```bash
--validation-method METHOD   # single (default), council, or none
--rectify-invalid            # Rectify invalid variations instead of dropping
--use-council                # DEPRECATED: use --validation-method council
--council-models MODELS      # OpenRouter model IDs for council judges
--chairman-model MODEL       # OpenRouter model for council chairman
--openrouter-api-key KEY     # OpenRouter API key
```

### Evaluation
```bash
--use-llm-judge              # Use LLM judge (default: string matching)
--disable-cot                # Disable chain-of-thought for faster responses
--force-regenerate           # Force regenerate responses
```

### Semantic Clustering
```bash
--embedding-model MODEL      # Embedding model (default: all-MiniLM-L6-v2)
--semantic-threshold N       # Clustering threshold (default: 0.35)
```

### Logging
```bash
--verbose / -v               # Enable verbose/debug output
```
All logs are saved to `logs/pipeline_debug.log`.

---

## Input Format

### JSONL (recommended for large datasets)
```json
{"question": "What is 15 + 25?", "answer": "40"}
{"question": "Solve: 3x + 5 = 20", "answer": "5"}
```
Both `question` and `problem` keys are accepted.

### JSON array
```json
[
  {"problem": "What is 15 + 25?", "answer": "40"},
  {"problem": "Solve: 3x + 5 = 20", "answer": "5"}
]
```

---

## Stage Dependencies

1. **Variations** — generates variations from input (requires `--input`)
2. **Validation** — validates variations (depends on: variations)
3. **Responses** — collects target model responses (depends on: validation or variations)
4. **Evaluation** — computes drift metrics (depends on: responses)
5. **Export CSV** — exports to CSV (depends on: evaluation)

Run stages in order, or use `--all-stages` to run all sequentially.

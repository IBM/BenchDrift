#!/bin/bash
#
# BenchDrift Experiment Runner
# Usage: ./run_experiment.sh <response_model> <benchmark> [max_problems] [--council]
# Example: ./run_experiment.sh ibm-granite/granite-3.3-8b-instruct gsm8k 100
# Example: ./run_experiment.sh ibm-granite/granite-3.3-8b-instruct gsm8k 100 --council
#

set -e

# Arguments
RESPONSE_MODEL="$1"
BENCHMARK="$2"
MAX_PROBLEMS="${3:-}"

if [[ -z "$RESPONSE_MODEL" || -z "$BENCHMARK" ]]; then
    echo "Usage: $0 <response_model> <benchmark> [max_problems]"
    echo "Example: $0 ibm-granite/granite-3.3-8b-instruct gsm8k"
    echo "Example: $0 ibm-granite/granite-3.3-8b-instruct gsm8k 100"
    echo ""
    echo "Response models:"
    echo "  ibm-granite/granite-3.3-8b-instruct"
    echo "  microsoft/phi-4"
    echo "  Qwen/Qwen3-8B"
    echo "  mistralai/Mistral-7B-Instruct-v0.2"
    echo ""
    echo "Benchmarks: gsm8k, mmlu, tot_arithmetic"
    echo ""
    echo "Options:"
    echo "  max_problems: Optional limit on number of problems (default: all)"
    exit 1
fi

# Fixed configuration
VARIATION_MODEL="microsoft/phi-4"  # Used for variation generation AND single-judge validation
EVAL_JUDGE="/proj/data-eng/users/shailja/tmp/models/Qwen3-Coder-30B-A3B-Instruct"
BATCH_SIZE=5000
MAX_MODEL_LEN=8192
WAIT_SECONDS=15

# Validation configuration
USE_COUNCIL=false  # Set to true for council-based validation, false for single-judge

# Council configuration (used when USE_COUNCIL=true)
COUNCIL_JUDGE_1="microsoft/phi-4"
COUNCIL_JUDGE_2="mistralai/Mistral-7B-Instruct-v0.2"
COUNCIL_JUDGE_3="Qwen/Qwen2-7B-Instruct"
COUNCIL_CHAIRMAN="microsoft/phi-4"
MIN_JUDGES=2

# Derive paths
MODEL_SHORT=$(basename "$RESPONSE_MODEL")
EXPERIMENT_ID="${MODEL_SHORT}_${BENCHMARK}"
OUTPUT_FILE="experiments/${EXPERIMENT_ID}/${EXPERIMENT_ID}_results.json"
INPUT_FILE="data/${BENCHMARK}/test.jsonl"

# Create output directory
mkdir -p "experiments/${EXPERIMENT_ID}"

echo "========================================"
echo "BenchDrift Experiment"
echo "========================================"
echo "Response Model: $RESPONSE_MODEL"
echo "Benchmark: $BENCHMARK"
echo "Max Problems: ${MAX_PROBLEMS:-all}"
echo "Validation: $(if $USE_COUNCIL; then echo 'COUNCIL (multi-judge)'; else echo 'Single Judge'; fi)"
echo "Output: $OUTPUT_FILE"
echo "========================================"

# Build max-problems flag if specified
MAX_PROBLEMS_FLAG=""
if [[ -n "$MAX_PROBLEMS" ]]; then
    MAX_PROBLEMS_FLAG="--max-problems $MAX_PROBLEMS"
fi

# Stage 1: Generate Variations
echo ""
echo "[Stage 1/4] Generating Variations..."
python -m benchdrift.pipeline.unified_batched_pipeline_semantic \
    --unified-file "$OUTPUT_FILE" \
    --input "$INPUT_FILE" \
    --stage variations \
    --client-type vllm \
    --model-name "$VARIATION_MODEL" \
    --batch-size $BATCH_SIZE \
    --max-model-len $MAX_MODEL_LEN \
    --max-tokens 1024 \
    --temperature 0.1 \
    --max-workers 4 \
    --embedding-model all-MiniLM-L6-v2 \
    --semantic-threshold 0.35 \
    --use-generic \
    --no-cluster-variations \
    --save-every-batch \
    $MAX_PROBLEMS_FLAG

echo "Stage 1 complete. Waiting ${WAIT_SECONDS}s..."
sleep $WAIT_SECONDS

# Stage 2: Validate Variations
if $USE_COUNCIL; then
    # COUNCIL-BASED VALIDATION (multiple judges)
    echo ""
    echo "[Stage 2/4] Validating Variations (COUNCIL MODE)..."
    echo "  Using ${MIN_JUDGES}+ judges for consensus-based validation"

    # Clear any previous verdicts
    echo ""
    echo "[Stage 2a] Clearing previous council verdicts..."
    python -m benchdrift.pipeline.council_validator \
        --unified-file "$OUTPUT_FILE" \
        --mode clear

    # Judge 1
    echo ""
    echo "[Stage 2b] Running Judge 1: $COUNCIL_JUDGE_1..."
    python -m benchdrift.pipeline.council_validator \
        --unified-file "$OUTPUT_FILE" \
        --mode judge \
        --judge-id judge_1 \
        --model-name "$COUNCIL_JUDGE_1" \
        --client-type vllm \
        --batch-size $BATCH_SIZE \
        --max-model-len $MAX_MODEL_LEN

    echo "Judge 1 complete. Waiting ${WAIT_SECONDS}s..."
    sleep $WAIT_SECONDS

    # Judge 2
    echo ""
    echo "[Stage 2c] Running Judge 2: $COUNCIL_JUDGE_2..."
    python -m benchdrift.pipeline.council_validator \
        --unified-file "$OUTPUT_FILE" \
        --mode judge \
        --judge-id judge_2 \
        --model-name "$COUNCIL_JUDGE_2" \
        --client-type vllm \
        --batch-size $BATCH_SIZE \
        --max-model-len $MAX_MODEL_LEN

    echo "Judge 2 complete. Waiting ${WAIT_SECONDS}s..."
    sleep $WAIT_SECONDS

    # Judge 3
    echo ""
    echo "[Stage 2d] Running Judge 3: $COUNCIL_JUDGE_3..."
    python -m benchdrift.pipeline.council_validator \
        --unified-file "$OUTPUT_FILE" \
        --mode judge \
        --judge-id judge_3 \
        --model-name "$COUNCIL_JUDGE_3" \
        --client-type vllm \
        --batch-size $BATCH_SIZE \
        --max-model-len $MAX_MODEL_LEN

    echo "Judge 3 complete. Waiting ${WAIT_SECONDS}s..."
    sleep $WAIT_SECONDS

    # Council Synthesis
    echo ""
    echo "[Stage 2e] Running Council Synthesis: $COUNCIL_CHAIRMAN..."
    python -m benchdrift.pipeline.council_validator \
        --unified-file "$OUTPUT_FILE" \
        --mode council \
        --model-name "$COUNCIL_CHAIRMAN" \
        --client-type vllm \
        --batch-size $BATCH_SIZE \
        --max-model-len $MAX_MODEL_LEN \
        --min-judges $MIN_JUDGES \
        --rectify-invalid

    echo "Council validation complete. Waiting ${WAIT_SECONDS}s..."
    sleep $WAIT_SECONDS

else
    # SINGLE-JUDGE VALIDATION (original behavior)
    echo ""
    echo "[Stage 2/4] Validating Variations..."
    python -m benchdrift.pipeline.unified_batched_pipeline_semantic \
        --unified-file "$OUTPUT_FILE" \
        --stage validation \
        --client-type vllm \
        --batch-size $BATCH_SIZE \
        --use-llm-judge \
        --judge-model "$VARIATION_MODEL" \
        --rectify-invalid

    echo "Stage 2 complete. Waiting ${WAIT_SECONDS}s..."
    sleep $WAIT_SECONDS
fi

# Stage 3: Generate Responses (with logprobs collection)
echo ""
echo "[Stage 3/4] Generating Responses..."
python -m benchdrift.pipeline.unified_batched_pipeline_semantic \
    --unified-file "$OUTPUT_FILE" \
    --stage responses \
    --client-type vllm_logits \
    --response-model "$RESPONSE_MODEL" \
    --batch-size $BATCH_SIZE \
    --max-model-len $MAX_MODEL_LEN

echo "Stage 3 complete. Waiting ${WAIT_SECONDS}s..."
sleep $WAIT_SECONDS

# Stage 4: Evaluate Drift
echo ""
echo "[Stage 4/4] Evaluating Drift..."
python -m benchdrift.pipeline.unified_batched_pipeline_semantic \
    --unified-file "$OUTPUT_FILE" \
    --stage evaluation \
    --client-type vllm \
    --use-llm-judge \
    --judge-model "$EVAL_JUDGE" \
    --batch-size $BATCH_SIZE

echo ""
echo "========================================"
echo "Experiment Complete!"
echo "Results: $OUTPUT_FILE"
if $USE_COUNCIL; then
    echo "Council verdicts: ${OUTPUT_FILE%.json}_council_verdicts.json"
fi
echo "========================================"

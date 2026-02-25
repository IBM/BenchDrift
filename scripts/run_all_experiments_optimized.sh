#!/bin/bash
#
# BenchDrift v2 Optimized Batch Experiment Runner
# Simplified 2-role setup: Generator/Judge + Target models
#
# Key design decisions:
# - ONE model handles generation, validation, AND evaluation (simpler, fewer assumptions)
# - No council — single strong judge is sufficient (v1 showed 94% 3-way agreement)
# - Sequential stage execution with GPU memory cleanup between phases
# - Each model loads ONCE for ALL benchmarks (internal Python looping)
#
# Model loads: 8 total
# - 1 for variation generation (all benchmarks)
# - 1 for validation (all benchmarks)
# - 5 for target model responses (each processes all benchmarks)
# - 1 for evaluation (all experiments)
#
set -e

# ========================================
# CONFIGURATION — edit these for your setup
# ========================================

# Generator/Judge model — generates variations, validates them, evaluates answers
JUDGE_MODEL="/proj/data-eng/users/shailja/tmp/models/Qwen3-Coder-30B-A3B-Instruct"

# Target models — models under test (responses evaluated for drift)
MODELS=(
    "Qwen/Qwen3-8B"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "microsoft/phi-4"
    "ibm-granite/granite-3.3-8b-instruct"
    "/proj/data-eng/granite-debug/models/gpt-oss-20b"
)

# Infrastructure
BATCH_SIZE=5000
MAX_MODEL_LEN=8192
MAX_PROBLEMS=500
WAIT_SECONDS=15

# Note: benchmarks are hardcoded in the _batch Python scripts (gsm8k, mmlu, math-hard)
# Update unified_batched_pipeline_semantic_batch.py BENCHMARKS list to change

echo "========================================"
echo "BenchDrift v2 — Simplified 2-Role Runner"
echo "========================================"
echo "Generator/Judge: $(basename $JUDGE_MODEL)"
echo "Target Models: ${#MODELS[@]}"
echo "Benchmarks: 3 (gsm8k, mmlu, math-hard)"
echo "Total experiments: $((${#MODELS[@]} * 3))"
echo "Problems per benchmark: $MAX_PROBLEMS"
echo "Expected model loads: 8"
echo "========================================"
echo ""

########################################
# PHASE 1: Generate variations (1 model load)
# Generator model produces variations for ALL benchmarks
########################################
echo ""
echo "########################################"
echo "# PHASE 1: Generate Variations"
echo "# Model: $(basename $JUDGE_MODEL) — loads ONCE"
echo "########################################"

python -m benchdrift.pipeline.unified_batched_pipeline_semantic_batch \
    --stage variations \
    --client-type vllm \
    --model-name "$JUDGE_MODEL" \
    --batch-size $BATCH_SIZE \
    --max-model-len $MAX_MODEL_LEN \
    --max-tokens 1024 \
    --temperature 0.1 \
    --max-workers 4 \
    --embedding-model all-MiniLM-L6-v2 \
    --semantic-threshold 0.35 \
    --use-axes "all" \
    --use-cluster-variations \
    --save-every-batch \
    --max-problems $MAX_PROBLEMS

wait
echo "Phase 1 complete. Waiting ${WAIT_SECONDS}s for GPU cleanup..."
sleep $WAIT_SECONDS

########################################
# PHASE 2: Validate variations (1 model load)
# Same judge model validates semantic equivalence
########################################
echo ""
echo "########################################"
echo "# PHASE 2: Validate Variations"
echo "# Judge: $(basename $JUDGE_MODEL) — loads ONCE"
echo "########################################"

python -m benchdrift.pipeline.unified_batched_pipeline_semantic_batch \
    --stage validation \
    --client-type vllm \
    --use-llm-judge \
    --judge-model "$JUDGE_MODEL" \
    --batch-size $BATCH_SIZE \
    --max-model-len $MAX_MODEL_LEN

wait
echo "Phase 2 complete. Waiting ${WAIT_SECONDS}s for GPU cleanup..."
sleep $WAIT_SECONDS

########################################
# Save validated variation templates
# Other target models copy these instead of regenerating
########################################
echo ""
echo "# Saving validated variation templates..."
BENCHMARKS=("gsm8k" "mmlu" "math-hard")
FIRST_MODEL_SHORT=$(basename "${MODELS[0]}")
for BENCHMARK in "${BENCHMARKS[@]}"; do
    SOURCE="experiments/${FIRST_MODEL_SHORT}_${BENCHMARK}/${FIRST_MODEL_SHORT}_${BENCHMARK}_results.json"
    TEMPLATE="experiments/variations_template_${BENCHMARK}.json"
    if [ -f "$SOURCE" ]; then
        cp "$SOURCE" "$TEMPLATE"
        echo "  Saved: $TEMPLATE"
    else
        echo "  WARNING: $SOURCE not found — skipping"
    fi
done

########################################
# PHASE 3: Generate responses (1 load per target model)
# Each target model evaluates baseline + all variations
########################################
echo ""
echo "########################################"
echo "# PHASE 3: Generate Responses"
echo "# ${#MODELS[@]} target models, each loads ONCE"
echo "########################################"

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(basename "$MODEL")
    echo ""
    echo "========================================"
    echo "Target: $MODEL_SHORT — loads ONCE for all benchmarks"
    echo "========================================"

    python -m benchdrift.pipeline.unified_batched_pipeline_semantic_batch \
        --stage responses \
        --client-type vllm_logits \
        --response-model "$MODEL" \
        --batch-size $BATCH_SIZE \
        --max-model-len $MAX_MODEL_LEN

    wait
    echo "Done: $MODEL_SHORT. Waiting ${WAIT_SECONDS}s..."
    sleep $WAIT_SECONDS
done

########################################
# PHASE 4: Evaluate drift (1 model load)
# Judge model scores all responses across all experiments
########################################
echo ""
echo "########################################"
echo "# PHASE 4: Evaluate Drift"
echo "# Judge: $(basename $JUDGE_MODEL) — loads ONCE for all $((${#MODELS[@]} * 3)) experiments"
echo "########################################"

python -m benchdrift.pipeline.unified_batched_pipeline_semantic_batch \
    --stage evaluation \
    --client-type vllm \
    --use-llm-judge \
    --judge-model "$JUDGE_MODEL" \
    --batch-size $BATCH_SIZE \
    --max-model-len $MAX_MODEL_LEN

wait
echo ""
echo "========================================"
echo "All experiments completed!"
echo "Model loads: 8 (1 gen + 1 val + ${#MODELS[@]} resp + 1 eval)"
echo "========================================"

#!/bin/bash
#
# BenchDrift Pipeline - Sequential Stage Runner (VLLM Safe)
# Runs each pipeline stage separately with waits to ensure GPU is released
#
# Usage:
#   ./run_stages_sequential.sh                    # Run with defaults
#   ./run_stages_sequential.sh --model phi-4 --benchmark gsm8k
#   ./run_stages_sequential.sh --help
#

set -e  # Exit on error

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Default values
MODEL="microsoft/phi-4"
MODEL_SHORT="phi-4"
BENCHMARK="gsm8k"
CLIENT_TYPE="vllm"
BATCH_SIZE=50
MAX_MODEL_LEN=8192
JUDGE_MODEL="llama_3_3_70b"
SEMANTIC_THRESHOLD=0.35
WAIT_SECONDS=10  # Wait time between stages to ensure GPU release

# Additional parameters (matching run_semantic_pipeline.sh and paper spec)
EMBEDDING_MODEL="all-MiniLM-L6-v2"
MAX_WORKERS=4
MAX_TOKENS=1024
TEMPERATURE=0.1

# Paths (relative to script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$SCRIPT_DIR/../src/benchdrift/pipeline"
DATA_DIR="$SCRIPT_DIR/../data"
OUTPUT_BASE="$SCRIPT_DIR/../experiments"

# Benchmark paths
declare -A BENCHMARK_PATHS
BENCHMARK_PATHS["gsm8k"]="$DATA_DIR/gsm8k/test.jsonl"
BENCHMARK_PATHS["mmlu"]="$DATA_DIR/mmlu/test_1319.jsonl"
BENCHMARK_PATHS["tot_arithmetic"]="$DATA_DIR/tot/test_sampled_1000.jsonl"

# Model mappings
declare -A MODEL_PATHS
MODEL_PATHS["phi-4"]="microsoft/phi-4"
MODEL_PATHS["qwen3-8b"]="Qwen/Qwen3-8B"
MODEL_PATHS["mistral-7b"]="mistralai/Mistral-7B-Instruct-v0.2"
MODEL_PATHS["granite-3.3-8b"]="ibm-granite/granite-3.3-8b-instruct"
MODEL_PATHS["gpt-oss-20b"]="openai/gpt-oss-20b"

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run BenchDrift pipeline stages sequentially (VLLM safe)"
    echo ""
    echo "Options:"
    echo "  --model MODEL         Model to use (phi-4, qwen3-8b, mistral-7b, granite-3.3-8b, gpt-oss-20b)"
    echo "  --benchmark BENCH     Benchmark to use (gsm8k, mmlu, tot_arithmetic)"
    echo "  --client CLIENT       Client type (vllm, rits) [default: vllm]"
    echo "  --batch-size N        Batch size [default: 50]"
    echo "  --max-model-len N     Max context length for VLLM [default: 8192]"
    echo "  --judge-model MODEL   Model for LLM judge [default: llama_3_3_70b]"
    echo "  --semantic-threshold  Clustering threshold [default: 0.35]"
    echo "  --embedding-model M   Embedding model [default: all-MiniLM-L6-v2]"
    echo "  --max-workers N       Max parallel workers for RITS [default: 4]"
    echo "  --max-tokens N        Max output tokens [default: 1024]"
    echo "  --temperature T       Generation temperature [default: 0.1]"
    echo "  --wait-seconds N      Wait time between stages [default: 10]"
    echo "  --output-dir DIR      Output directory [default: experiments/]"
    echo "  --start-stage N       Start from stage N (1-4) [default: 1]"
    echo "  --end-stage N         End at stage N (1-4) [default: 4]"
    echo "  --help                Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --model phi-4 --benchmark gsm8k"
    echo "  $0 --model granite-3.3-8b --benchmark mmlu --client rits"
    echo "  $0 --start-stage 3 --end-stage 4  # Resume from stage 3"
}

START_STAGE=1
END_STAGE=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_SHORT="$2"
            MODEL="${MODEL_PATHS[$2]:-$2}"
            shift 2
            ;;
        --benchmark)
            BENCHMARK="$2"
            shift 2
            ;;
        --client)
            CLIENT_TYPE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --judge-model)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --semantic-threshold)
            SEMANTIC_THRESHOLD="$2"
            shift 2
            ;;
        --embedding-model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --wait-seconds)
            WAIT_SECONDS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --start-stage)
            START_STAGE="$2"
            shift 2
            ;;
        --end-stage)
            END_STAGE="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# ==============================================================================
# SETUP
# ==============================================================================

# Get benchmark input path
INPUT_FILE="${BENCHMARK_PATHS[$BENCHMARK]}"
if [[ -z "$INPUT_FILE" ]]; then
    echo "❌ Unknown benchmark: $BENCHMARK"
    echo "   Available: gsm8k, mmlu, tot_arithmetic"
    exit 1
fi

# Create output directory
EXPERIMENT_ID="${MODEL_SHORT}_${BENCHMARK}"
OUTPUT_DIR="$OUTPUT_BASE/$EXPERIMENT_ID"
OUTPUT_FILE="$OUTPUT_DIR/${EXPERIMENT_ID}_results.json"

mkdir -p "$OUTPUT_DIR"

# ==============================================================================
# LOGGING
# ==============================================================================

LOG_FILE="$OUTPUT_DIR/${EXPERIMENT_ID}_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_separator() {
    echo "" | tee -a "$LOG_FILE"
    echo "========================================================================" | tee -a "$LOG_FILE"
    echo "$1" | tee -a "$LOG_FILE"
    echo "========================================================================" | tee -a "$LOG_FILE"
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

log_separator "BENCHDRIFT SEQUENTIAL PIPELINE"
log "Experiment ID: $EXPERIMENT_ID"
log "Model: $MODEL"
log "Benchmark: $BENCHMARK"
log "Input: $INPUT_FILE"
log "Output: $OUTPUT_FILE"
log "Client: $CLIENT_TYPE"
log "Batch size: $BATCH_SIZE"
log "Max model len: $MAX_MODEL_LEN"
log "Max tokens: $MAX_TOKENS"
log "Temperature: $TEMPERATURE"
log "Judge model: $JUDGE_MODEL"
log "Embedding model: $EMBEDDING_MODEL"
log "Semantic threshold: $SEMANTIC_THRESHOLD"
log "Max workers: $MAX_WORKERS"
log "Wait between stages: ${WAIT_SECONDS}s"
log "Stages: $START_STAGE to $END_STAGE"
log "Log file: $LOG_FILE"

# Check input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    log "❌ ERROR: Input file not found: $INPUT_FILE"
    log "   Please prepare benchmark data first"
    exit 1
fi

cd "$SCRIPT_DIR/.."

# ------------------------------------------------------------------------------
# STAGE 1: GENERATE VARIATIONS
# ------------------------------------------------------------------------------
if [[ $START_STAGE -le 1 && $END_STAGE -ge 1 ]]; then
    log_separator "STAGE 1: GENERATE VARIATIONS"
    log "Starting variation generation..."

    python -m benchdrift.pipeline.unified_batched_pipeline_semantic \
        --unified-file "$OUTPUT_FILE" \
        --input "$INPUT_FILE" \
        --stage variations \
        --client-type "$CLIENT_TYPE" \
        --model-name "$MODEL" \
        --batch-size "$BATCH_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --max-workers "$MAX_WORKERS" \
        --embedding-model "$EMBEDDING_MODEL" \
        --semantic-threshold "$SEMANTIC_THRESHOLD" \
        --use-generic \
        --use-cluster-variations \
        --use-persona \
        --use-long-context \
        2>&1 | tee -a "$LOG_FILE"

    STAGE1_EXIT=$?
    if [[ $STAGE1_EXIT -ne 0 ]]; then
        log "❌ Stage 1 failed with exit code: $STAGE1_EXIT"
        exit $STAGE1_EXIT
    fi

    log "✅ Stage 1 complete"
    log "⏳ Waiting ${WAIT_SECONDS}s for GPU memory release..."
    sleep $WAIT_SECONDS
fi

# ------------------------------------------------------------------------------
# STAGE 2: VALIDATE VARIATIONS
# ------------------------------------------------------------------------------
if [[ $START_STAGE -le 2 && $END_STAGE -ge 2 ]]; then
    log_separator "STAGE 2: VALIDATE VARIATIONS"
    log "Starting validation..."

    python -m benchdrift.pipeline.unified_batched_pipeline_semantic \
        --unified-file "$OUTPUT_FILE" \
        --stage validation \
        --client-type "$CLIENT_TYPE" \
        --model-name "$MODEL" \
        --use-llm-judge \
        --judge-model "$JUDGE_MODEL" \
        --rectify-invalid \
        2>&1 | tee -a "$LOG_FILE"

    STAGE2_EXIT=$?
    if [[ $STAGE2_EXIT -ne 0 ]]; then
        log "❌ Stage 2 failed with exit code: $STAGE2_EXIT"
        exit $STAGE2_EXIT
    fi

    log "✅ Stage 2 complete"
    log "⏳ Waiting ${WAIT_SECONDS}s for GPU memory release..."
    sleep $WAIT_SECONDS
fi

# ------------------------------------------------------------------------------
# STAGE 3: GENERATE RESPONSES
# ------------------------------------------------------------------------------
if [[ $START_STAGE -le 3 && $END_STAGE -ge 3 ]]; then
    log_separator "STAGE 3: GENERATE RESPONSES"
    log "Starting response generation..."

    python -m benchdrift.pipeline.unified_batched_pipeline_semantic \
        --unified-file "$OUTPUT_FILE" \
        --stage responses \
        --client-type "$CLIENT_TYPE" \
        --response-model "$MODEL" \
        --batch-size "$BATCH_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --max-workers "$MAX_WORKERS" \
        2>&1 | tee -a "$LOG_FILE"

    STAGE3_EXIT=$?
    if [[ $STAGE3_EXIT -ne 0 ]]; then
        log "❌ Stage 3 failed with exit code: $STAGE3_EXIT"
        exit $STAGE3_EXIT
    fi

    log "✅ Stage 3 complete"
    log "⏳ Waiting ${WAIT_SECONDS}s for GPU memory release..."
    sleep $WAIT_SECONDS
fi

# ------------------------------------------------------------------------------
# STAGE 4: EVALUATE DRIFT
# ------------------------------------------------------------------------------
if [[ $START_STAGE -le 4 && $END_STAGE -ge 4 ]]; then
    log_separator "STAGE 4: EVALUATE DRIFT"
    log "Starting evaluation..."

    python -m benchdrift.pipeline.unified_batched_pipeline_semantic \
        --unified-file "$OUTPUT_FILE" \
        --stage evaluation \
        --client-type "$CLIENT_TYPE" \
        --use-llm-judge \
        --judge-model "$JUDGE_MODEL" \
        2>&1 | tee -a "$LOG_FILE"

    STAGE4_EXIT=$?
    if [[ $STAGE4_EXIT -ne 0 ]]; then
        log "❌ Stage 4 failed with exit code: $STAGE4_EXIT"
        exit $STAGE4_EXIT
    fi

    log "✅ Stage 4 complete"
fi

# ==============================================================================
# COMPLETION
# ==============================================================================

log_separator "EXPERIMENT COMPLETE"
log "Experiment: $EXPERIMENT_ID"
log "Results: $OUTPUT_FILE"
log "Log: $LOG_FILE"
log "Finished: $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo "🎉 All stages completed successfully!"
echo "   Results: $OUTPUT_FILE"
echo "   Log: $LOG_FILE"

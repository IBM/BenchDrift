#!/bin/bash

# Semantic Pipeline Runner
# Runs the semantic clustering pipeline with embedding-based variation generation

set -e  # Exit on any error

# Default configuration (can be overridden via command-line)
INPUT_FILE="demo_problems.json"
OUTPUT_FILE="demo_output_semantic.json"
BATCH_SIZE=50
MAX_WORKERS=4
CLIENT_TYPE="rits"
MODEL_NAME="phi-4"
RESPONSE_MODEL="granite-3-3-8b"
JUDGE_MODEL="llama_3_3_70b"
USE_LLM_JUDGE="true"
RECTIFY_INVALID="true"
MAX_MODEL_LEN=8192
MAX_NEW_TOKENS=1000
EMBEDDING_MODEL="all-MiniLM-L6-v2"
SEMANTIC_THRESHOLD=0.35
USE_CAGRAD_DEPS="false"
USE_GENERIC="true"
USE_CLUSTER_VARIATIONS="true"
USE_PERSONA="false"
USE_LONG_CONTEXT="false"

# Show usage
show_usage() {
    echo "Semantic Pipeline Runner"
    echo "========================"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Pipeline Options:"
    echo "  --input FILE              Input problems file (default: demo_problems.json)"
    echo "  --output FILE             Output results file (default: demo_output_semantic.json)"
    echo "  --batch-size N            Batch size for processing (default: 50)"
    echo "  --max-workers N           Max parallel workers (default: 4)"
    echo ""
    echo "Model Options:"
    echo "  --client-type TYPE        Client type: rits, vllm, or gemini (default: rits)"
    echo "  --model-name MODEL        Model for variation generation (default: phi-4)"
    echo "  --response-model MODEL    Model for responses (default: granite-3-3-8b)"
    echo "  --judge-model MODEL       Model for LLM judge (default: llama_3_3_70b)"
    echo "  --max-model-len N         Max context length for VLLM (default: 8192)"
    echo "  --max-new-tokens N        Max output tokens (default: 1000)"
    echo ""
    echo "Semantic Clustering Options:"
    echo "  --embedding-model MODEL   Embedding model (default: all-MiniLM-L6-v2)"
    echo "  --semantic-threshold N    Clustering threshold (default: 0.35)"
    echo "  --use-cagrad-deps         Enable CAGrad dependency testing (default: false)"
    echo ""
    echo "Variation Types:"
    echo "  --use-generic             Enable generic variations (default: true)"
    echo "  --use-cluster-variations  Enable cluster variations (default: true)"
    echo "  --use-persona             Enable persona variations (default: false)"
    echo "  --use-long-context        Enable long context variations (default: false)"
    echo ""
    echo "Evaluation Options:"
    echo "  --use-llm-judge           Use LLM judge for evaluation (default: true)"
    echo "  --no-llm-judge            Disable LLM judge"
    echo "  --rectify-invalid         Rectify invalid answers (default: true)"
    echo "  --no-rectify-invalid      Disable answer rectification"
    echo ""
    echo "Examples:"
    echo "  $0 --input my_problems.json --output results.json"
    echo "  $0 --use-long-context --semantic-threshold 0.4"
    echo "  $0 --client-type gemini --model-name gemini-2.0-flash-exp"
    echo "  $0 --client-type vllm --model-name llama-3-8b"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --client-type)
            CLIENT_TYPE="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --response-model)
            RESPONSE_MODEL="$2"
            shift 2
            ;;
        --judge-model)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --embedding-model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --semantic-threshold)
            SEMANTIC_THRESHOLD="$2"
            shift 2
            ;;
        --use-cagrad-deps)
            USE_CAGRAD_DEPS="true"
            shift
            ;;
        --use-generic)
            USE_GENERIC="true"
            shift
            ;;
        --no-generic)
            USE_GENERIC="false"
            shift
            ;;
        --use-cluster-variations)
            USE_CLUSTER_VARIATIONS="true"
            shift
            ;;
        --no-cluster-variations)
            USE_CLUSTER_VARIATIONS="false"
            shift
            ;;
        --use-persona)
            USE_PERSONA="true"
            shift
            ;;
        --no-persona)
            USE_PERSONA="false"
            shift
            ;;
        --use-long-context)
            USE_LONG_CONTEXT="true"
            shift
            ;;
        --no-long-context)
            USE_LONG_CONTEXT="false"
            shift
            ;;
        --use-llm-judge)
            USE_LLM_JUDGE="true"
            shift
            ;;
        --no-llm-judge)
            USE_LLM_JUDGE="false"
            shift
            ;;
        --rectify-invalid)
            RECTIFY_INVALID="true"
            shift
            ;;
        --no-rectify-invalid)
            RECTIFY_INVALID="false"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required files
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "❌ Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Display configuration
echo "🔬 Semantic Pipeline Configuration"
echo "===================================="
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Batch size: $BATCH_SIZE"
echo "Client: $CLIENT_TYPE"
echo "Variation model: $MODEL_NAME"
echo "Response model: $RESPONSE_MODEL"
echo "Judge model: $JUDGE_MODEL"
echo "Embedding model: $EMBEDDING_MODEL"
echo "Semantic threshold: $SEMANTIC_THRESHOLD"
echo ""
echo "Variation types:"
echo "  Generic: $USE_GENERIC"
echo "  Cluster-based: $USE_CLUSTER_VARIATIONS"
echo "  Persona: $USE_PERSONA"
echo "  Long context: $USE_LONG_CONTEXT"
echo ""

# Run the pipeline using Python
python3 << EOF
from unified_batched_pipeline_semantic import UnifiedBatchedPipeline

config = {
    'unified_file': '$OUTPUT_FILE',
    'input_problems': '$INPUT_FILE',
    'batch_size': $BATCH_SIZE,
    'max_workers': $MAX_WORKERS,
    'client_type': '$CLIENT_TYPE',
    'model_name': '$MODEL_NAME',
    'response_model': '$RESPONSE_MODEL',
    'judge_model': '$JUDGE_MODEL',
    'use_llm_judge': $USE_LLM_JUDGE,
    'rectify_invalid': $RECTIFY_INVALID,
    'max_model_len': $MAX_MODEL_LEN,
    'max_new_tokens': $MAX_NEW_TOKENS,
    'embedding_model': '$EMBEDDING_MODEL',
    'semantic_threshold': $SEMANTIC_THRESHOLD,
    'use_cagrad_dependencies': $USE_CAGRAD_DEPS,
    'use_generic': $USE_GENERIC,
    'use_cluster_variations': $USE_CLUSTER_VARIATIONS,
    'use_persona': $USE_PERSONA,
    'use_long_context': $USE_LONG_CONTEXT,
}

print("🚀 Running semantic pipeline...")
pipeline = UnifiedBatchedPipeline(config)

print("   Stage 1: Generating variations...")
pipeline.stage1_generate_variations_batched()

print("   Stage 2: Validating variations...")
pipeline.stage_validation()

print("   Stage 3: Generating responses...")
pipeline.stage2_generate_responses()

print("   Stage 4: Evaluating results...")
pipeline.stage3_add_evaluation_metrics()

print(f"\n✅ Pipeline complete! Results saved to $OUTPUT_FILE")
EOF

if [[ $? -eq 0 ]]; then
    echo ""
    echo "🎉 Semantic Pipeline Complete!"
    echo "==============================="
    echo "Results saved to: $OUTPUT_FILE"
    echo ""
else
    echo ""
    echo "❌ Pipeline failed!"
    exit 1
fi

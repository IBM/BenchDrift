#!/bin/bash
#
# BenchDrift Batch Experiment Runner
# Runs all model x benchmark combinations sequentially
#

set -e

# Models
MODELS=(
    "Qwen/Qwen3-8B"
    "mistralai/Ministral-3-8B-Instruct-2512"
    "microsoft/phi-4"
    "ibm-granite/granite-3.3-8b-instruct"
    "/proj/data-eng/granite-debug/models/gpt-oss-20b"
)

# Benchmarks
BENCHMARKS=(
    "gsm8k"
    "mmlu"
    "tot"
    "math-hard"
    "triviaqa"
)

# Number of problems per experiment
MAX_PROBLEMS=500

echo "========================================"
echo "BenchDrift Batch Experiment Runner"
echo "========================================"
echo "Models: ${#MODELS[@]}"
echo "Benchmarks: ${#BENCHMARKS[@]}"
echo "Total experiments: $((${#MODELS[@]} * ${#BENCHMARKS[@]}))"
echo "Problems per experiment: $MAX_PROBLEMS"
echo "========================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for MODEL in "${MODELS[@]}"; do
    for BENCHMARK in "${BENCHMARKS[@]}"; do
        echo ""
        echo "========================================"
        echo "Running: $MODEL x $BENCHMARK"
        echo "========================================"

        "$SCRIPT_DIR/run_experiment.sh" "$MODEL" "$BENCHMARK" "$MAX_PROBLEMS"
        wait

        echo ""
        echo "Completed: $MODEL x $BENCHMARK"
        echo "========================================"
    done
done

echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"

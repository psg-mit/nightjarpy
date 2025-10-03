#!/bin/bash

# ./docker_eval.sh interpreter_python_eager anthropic/claude-sonnet-4-20250514 1

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <mode: interpreter_bytecode|interpreter_python|interpreter_python_eager|interpreter_python_eager_effectcount_300|interpreter_python_eager_effectcount_20|interpreter_python_eager_parallel|interpreter_python_var_eager|interpreter_jit_python_eager|compiler_aot|compiler_aot_source|manual> <model: openai/gpt-4.1-2025-04-14|anthropic/claude-sonnet-4-20250514|anthropic/claude-sonnet-4-5-20250929> <number_of_runs>"
    exit 1
fi

MODE=$1
MODEL=$2
RUNS=$3

cd ..

# Build the Docker image
echo "Building Docker image"
docker build -t nightjar .

# remove . and / and - from model name
case $MODEL in
    "openai/gpt-4.1-2025-04-14")
        model_name="gpt41"
        ;;
    "anthropic/claude-sonnet-4-20250514")
        model_name="sonnet4"
        ;;
    "anthropic/claude-sonnet-4-5-20250929")
        model_name="sonnet45"
        ;;
    *)
        echo "Invalid model. Use: openai/gpt-4.1-2025-04-14, anthropic/claude-sonnet-4-20250514"
        exit 1
        ;;
esac

container_name="$MODE$model_name"

# Remove the container if it exists
docker rm $container_name

# Run the container in detached mode
echo "Running container"
docker run --name $container_name --env-file .env -d -it nightjar 
# $(docker ps --latest --format "{{.Names}}")
echo "Container name: $container_name"

echo "Benchmark running in container $container_name with mode $MODE and $RUNS runs"

case $MODE in
    "interpreter_python_eager_effectcount_300")
        actual_mode="interpreter_python_eager_effectcount"
        max_effects="300"
        ;;
    "interpreter_python_eager_effectcount_20")
        actual_mode="interpreter_python_eager_effectcount"
        max_effects="20"
        ;;
    *)
        actual_mode=$MODE
        max_effects="300"
        ;;
esac

# Base command with common arguments
base_cmd="python scripts/harness.py \
    --benchmark_dir benchmarks/programs \
    --output_file benchmarks/results_$MODE.jsonl \
    --runs $RUNS \
    --timeout 1000 \
    --model $MODEL \
    --max_tool_calls $max_effects \
    --handler_name $actual_mode"

# Create results directory if it doesn't exist
mkdir -p benchmarks/results/final/$MODEL/$MODE

# Copy to docker if file exists
if [ -f "benchmarks/results/final/$MODEL/$MODE/results_$MODE.jsonl" ]; then
    docker cp benchmarks/results/final/$MODEL/$MODE/results_$MODE.jsonl $container_name:/nightjar/benchmarks/
fi

echo "Running command > $base_cmd"
# Execute the command in the container
timeout -k 10 3h docker exec $container_name $base_cmd
# docker exec $container_name $final_cmd

# Copy results back
mkdir -p benchmarks/results/final/$MODEL/$MODE

# Copy the results and traces files
docker cp $container_name:/nightjar/benchmarks/results_$MODE.jsonl benchmarks/results/final/$MODEL/$MODE/

# Stop the container
docker stop $container_name

cd scripts

say "[[volm 0.5]] Benchmark finished running with mode $MODE and $RUNS runs"

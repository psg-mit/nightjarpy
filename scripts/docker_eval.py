#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

from harness import HANDLERS
from tap import Tap


class Args(Tap):
    mode: HANDLERS

    model: Literal[
        "openai/gpt-4.1-2025-04-14",
        "anthropic/claude-sonnet-4-20250514",
        "anthropic/claude-sonnet-4-5-20250929",
    ]

    runs: int

    temperature: float


def run_command(cmd: list[str], check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=True)


def main():
    args = Args().parse_args()

    # Validate model
    model_mapping = {
        "openai/gpt-4.1-2025-04-14": "gpt41",
        "anthropic/claude-sonnet-4-20250514": "sonnet4",
        "anthropic/claude-sonnet-4-5-20250929": "sonnet45",
    }

    if args.model not in model_mapping:
        print(f"Invalid model. Use: {', '.join(model_mapping.keys())}")
        sys.exit(1)

    model_name = model_mapping[args.model]

    # Change to parent directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)

    # Build Docker image
    print("Building Docker image")
    run_command(["docker", "build", "-t", "nightjar", "."])

    # Container name
    container_name = f"{args.mode}{model_name}{str(args.temperature).replace('.', 'p')}"

    # Remove container if it exists
    print(f"Removing existing container {container_name} if it exists")
    run_command(["docker", "rm", container_name], check=False)

    # Run container in detached mode
    print("Running container")
    run_command(["docker", "run", "--name", container_name, "--env-file", ".env", "-d", "-it", "nightjar"])
    print(f"Container name: {container_name}")

    print(f"Benchmark running in container {container_name} with mode {args.mode} and {args.runs} runs")

    actual_mode = args.mode
    max_effects = "300"

    # Construct base command
    base_cmd = [
        "python",
        "scripts/harness.py",
        "--benchmark_dir",
        "benchmarks/programs",
        "--output_file",
        f"benchmarks/results_{args.mode}_t{str(args.temperature).replace('.', 'p')}.jsonl",
        "--trace_file",
        f"benchmarks/trace_{args.mode}_t{str(args.temperature).replace('.', 'p')}.jsonl",
        "--runs",
        str(args.runs),
        "--timeout",
        "1000",
        "--model",
        args.model,
        "--max_tool_calls",
        max_effects,
        "--handler_name",
        actual_mode,
        "--temperature",
        str(args.temperature),
    ]

    # Create results directory
    results_dir = Path(f"benchmarks/results/final/{args.model}/{args.mode}_t{str(args.temperature).replace('.', 'p')}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Copy existing results to docker if file exists
    results_file = results_dir / f"results_{args.mode}_t{str(args.temperature).replace('.', 'p')}.jsonl"
    if results_file.exists():
        print(f"Copying existing results to container")
        run_command(["docker", "cp", str(results_file), f"{container_name}:/nightjar/benchmarks/"])

    # Copy existing trace to docker if file exists
    trace_file = results_dir / f"trace_{args.mode}_t{str(args.temperature).replace('.', 'p')}.jsonl"
    if trace_file.exists():
        print(f"Copying existing trace to container")
        run_command(["docker", "cp", str(trace_file), f"{container_name}:/nightjar/benchmarks/"])

    # Execute command in container with timeout
    print(f"Running command > {' '.join(base_cmd)}")
    timeout_cmd = ["timeout", "-k", "10", "3h", "docker", "exec", container_name] + base_cmd

    try:
        run_command(timeout_cmd, check=False)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Copy results back even if interrupted
        print("Copying results back from container")
        results_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                "docker",
                "cp",
                f"{container_name}:/nightjar/benchmarks/results_{args.mode}_t{str(args.temperature).replace('.', 'p')}.jsonl",
                str(results_dir) + "/",
            ],
            check=False,
        )

        # Copy trace back from container
        print("Copying trace back from container")
        run_command(
            [
                "docker",
                "cp",
                f"{container_name}:/nightjar/benchmarks/trace_{args.mode}_t{str(args.temperature).replace('.', 'p')}.jsonl",
                str(results_dir) + "/",
            ],
            check=False,
        )

        # Stop container
        print(f"Stopping container {container_name}")
        run_command(["docker", "stop", container_name])

    # Change back to scripts directory
    os.chdir(script_dir)

    # Announce completion (macOS specific)
    run_command(
        ["say", "[[volm 0.5]]", f"Benchmark finished running with mode {args.mode} and {args.runs} runs"], check=False
    )


if __name__ == "__main__":
    main()

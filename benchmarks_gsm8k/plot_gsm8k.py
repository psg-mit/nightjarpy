"""
# Default output (GitHub markdown)
python benchmarks_gsm8k/plot_gsm8k.py

# LaTeX output
python benchmarks_gsm8k/plot_gsm8k.py --latex

# Custom results directory
python benchmarks_gsm8k/plot_gsm8k.py --results_dir /Users/ellieyhc/Documents/Research/nightjar-private/benchmarks_gsm8k/results/final
"""

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate
from tap import Tap

# Suppress pandas FutureWarning for fillna downcasting
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")


class ArgumentParser(Tap):
    results_dir: Optional[Path] = None  # Base directory containing the results (e.g., benchmarks_gsm8k/results/final)
    latex: bool = False  # Whether to print the table in LaTeX format


MODEL_MAPPING = {
    "openai/gpt-4.1-2025-04-14": "GPT-4.1",
    "anthropic/claude-sonnet-4-20250514": "Sonnet 4",
    "anthropic/claude-sonnet-4-5-20250929": "Sonnet 4.5",
}

METHOD_MAPPING = {
    "manual": "Programmer Impl",
    "customtool": "Programmer Impl of Exec",
    "codetool": "Code Interpreter",
    "codetool_reuse": "Code Interpreter (Reuse Container)",
    "nightjar": "Nightjar (Ours)",
    "nightjaroptimize": "Nightjar (Ours) (Optimized)",
    "nightjarcache": "Nightjar (Ours) (Caching)",
}

TABLE_METHODS = [
    "manual",
    "codetool",
    "codetool_reuse",
    "customtool",
    "nightjar",
    "nightjaroptimize",
    "nightjarcache",
]

TABLE_RUNS = list(range(3))


def discover_models(base_results_dir: Path) -> List[Tuple[str, str, Path]]:
    """Discover all models in the nested directory structure.

    Args:
        base_results_dir: Base directory containing results (e.g., benchmarks_gsm8k/results/final)

    Returns:
        List of tuples: (provider, model_name, model_path)
    """
    models = []

    if not base_results_dir.exists():
        print(f"Warning: Base results directory {base_results_dir} does not exist")
        return models

    def find_innermost_models(directory: Path, current_provider: str = "") -> List[Tuple[str, str, Path]]:
        """Recursively find the innermost model directories."""
        found_models = []

        for item in directory.iterdir():
            if not item.is_dir():
                continue

            # Check if this directory contains method subdirectories with results files
            subdirs = [subdir for subdir in item.iterdir() if subdir.is_dir()]

            if subdirs:
                # Check if any of the subdirectories contain results files (method directories)
                has_method_subdirs = False
                for subdir in subdirs:
                    results_files = list(subdir.glob("results_*.jsonl"))
                    if results_files:  # If subdir has results files, it's a method directory
                        has_method_subdirs = True
                        break

                if has_method_subdirs:
                    # This is a model directory (contains method subdirectories)
                    model_name = item.name
                    provider = current_provider if current_provider else item.parent.name
                    found_models.append((provider, model_name, item))
                else:
                    # Continue searching deeper
                    new_provider = current_provider if current_provider else item.name
                    found_models.extend(find_innermost_models(item, new_provider))

        return found_models

    # Start the recursive search
    models = find_innermost_models(base_results_dir)
    return models


def compute_metrics(data: pd.DataFrame) -> Dict[str, float]:
    """Compute metrics for the given results.

    Args:
        data: DataFrame containing test results

    Returns:
        Dictionary of computed metrics
    """
    # Compute score metrics (eval_result is boolean)
    # Calculate mean score per run, then get avg and std across runs
    scores_per_run = data.groupby("run")["eval_result"].mean()
    avg_score = scores_per_run.mean()
    std_score = scores_per_run.std()

    # Compute runtime metrics
    runtimes = data["runtime"].dropna()
    avg_runtime = runtimes.mean()
    min_runtime = runtimes.min()
    max_runtime = runtimes.max()

    # Compute compile time metrics (if available)
    # Convert compile_time to numeric (it might be datetime/NaT or None)
    data = data.copy()
    # Replace NaT with None first to avoid conversion issues
    compile_times = data["compile_time"].dropna()
    if len(compile_times) > 0:
        avg_compile_time = compile_times.mean()
        min_compile_time = compile_times.min()
        max_compile_time = compile_times.max()
    else:
        avg_compile_time = 0.0
        min_compile_time = 0.0
        max_compile_time = 0.0

    # Compute total time
    data["total_time"] = data["runtime"] + data["compile_time"]
    total_times = data["total_time"].dropna()
    avg_total_time = float(total_times.mean())
    min_total_time = float(total_times.min())
    max_total_time = float(total_times.max())

    # Compute tool call metrics
    tool_calls = data["n_tool_calls"].dropna()
    if len(tool_calls) > 0:
        avg_tool_calls = tool_calls.mean()
        min_tool_calls = tool_calls.min()
        max_tool_calls = tool_calls.max()
    else:
        avg_tool_calls = np.nan
        min_tool_calls = np.nan
        max_tool_calls = np.nan

    # Compute token metrics
    def get_token_count(x, key):
        """Get token count handling different structures."""
        if x is None or not isinstance(x, dict):
            return np.nan
        return x.get(key, np.nan) or np.nan

    data["input_tokens"] = data["token_count"].apply(lambda x: get_token_count(x=x, key="input_tokens"))
    data["output_tokens"] = data["token_count"].apply(lambda x: get_token_count(x=x, key="output_tokens"))

    input_tokens = data["input_tokens"].dropna()
    avg_input_tokens = input_tokens.mean()
    min_input_tokens = input_tokens.min()
    max_input_tokens = input_tokens.max()

    output_tokens = data["output_tokens"].dropna()
    avg_output_tokens = output_tokens.mean()
    min_output_tokens = output_tokens.min()
    max_output_tokens = output_tokens.max()

    return {
        "avg_score": float(avg_score),
        "std_score": float(std_score),
        "avg_runtime": float(avg_runtime),
        "min_runtime": float(min_runtime),
        "max_runtime": float(max_runtime),
        "avg_compile_time": float(avg_compile_time),
        "min_compile_time": float(min_compile_time),
        "max_compile_time": float(max_compile_time),
        "avg_total_time": float(avg_total_time),
        "min_total_time": float(min_total_time),
        "max_total_time": float(max_total_time),
        "avg_tool_calls": float(avg_tool_calls),
        "min_tool_calls": float(min_tool_calls),
        "max_tool_calls": float(max_tool_calls),
        "avg_input_tokens": float(avg_input_tokens),
        "min_input_tokens": float(min_input_tokens),
        "max_input_tokens": float(max_input_tokens),
        "avg_output_tokens": float(avg_output_tokens),
        "min_output_tokens": float(min_output_tokens),
        "max_output_tokens": float(max_output_tokens),
    }


def get_model_display_name(provider: str, model_name: str) -> str:
    """Get a display name for the model based on provider and model name."""
    full_name = f"{provider}/{model_name}"
    return MODEL_MAPPING.get(full_name, model_name)


def main():
    args = ArgumentParser().parse_args()

    if args.results_dir is None:
        results_dirname = "benchmarks_gsm8k/results/final"

        # Try multiple possible paths to support running from root or scripts/ directory
        possible_paths = [
            Path(results_dirname),  # From root directory
            Path("..") / results_dirname,  # From scripts/ directory
        ]

        for path in possible_paths:
            if path.exists():
                args.results_dir = path
                break

        if args.results_dir is None:
            # If neither path exists, default to the root directory path
            args.results_dir = Path(results_dirname)

    # Discover all models in the directory structure
    models = discover_models(args.results_dir)

    if not models:
        print(f"No models found in {args.results_dir}")
        return

    print(f"Found {len(models)} models:")
    for provider, model_name, model_path in models:
        display_name = get_model_display_name(provider, model_name)
        print(f"  - {display_name} ({provider}/{model_name})")

    # Group models by provider for better organization
    models_by_provider = {}
    for provider, model_name, model_path in models:
        if provider not in models_by_provider:
            models_by_provider[provider] = []
        models_by_provider[provider].append((model_name, model_path))

    summary_data = []

    # Process each model and method
    for provider in sorted(models_by_provider.keys()):
        for model_name, model_path in models_by_provider[provider]:
            model_display_name = get_model_display_name(provider, model_name)

            # Process each method in the order specified in TABLE_METHODS
            for method_name in TABLE_METHODS:
                method_path = model_path / method_name
                if not method_path.exists():
                    continue

                # Find the results file
                results_files = list(method_path.glob("results_*.jsonl"))
                if not results_files:
                    print(f"Warning: No results file found for {model_display_name}/{method_name}")
                    continue

                results_file = results_files[0]

                # Read the results
                with open(results_file, "r") as f:
                    results = pd.read_json(f, lines=True, convert_dates=False, dtype={"compile_time": "float64"})

                print(f"\nProcessing results for {model_display_name}/{method_name}")
                print(f"  -> Number of rows: {len(results)}")

                if results.empty:
                    continue

                results = results[results["run"].isin(TABLE_RUNS)]

                if results.empty:
                    print(f"  -> No data for runs {TABLE_RUNS}")
                    continue

                present_runs = set(results["run"].unique())
                expected_runs = set(TABLE_RUNS)
                missing_runs = expected_runs - present_runs
                incomplete = len(missing_runs) > 0

                if incomplete:
                    print(f"  -> Missing runs: {sorted(missing_runs)}")

                # Compute metrics
                metrics = compute_metrics(results)

                # Format the row for the summary table
                method_display_name = METHOD_MAPPING.get(method_name, method_name)
                if incomplete:
                    method_display_name = f"*{method_display_name}"

                # Choose plus-minus symbol based on output format
                pm_symbol = "PLUSMINUS" if args.latex else "±"

                row = {
                    "Model": model_display_name,
                    "Method": method_display_name,
                    "Pass Rate": f"{metrics['avg_score']:.2f}{pm_symbol}{metrics['std_score']:.2f}",
                    "Runtime": f"{metrics['avg_runtime']:.1f} ({metrics['min_runtime']:.1f}-{metrics['max_runtime']:.1f})",
                    "Tool Calls": f"{metrics['avg_tool_calls']:.1f} ({metrics['min_tool_calls']:.0f}-{metrics['max_tool_calls']:.0f})",
                    "Input Tokens": f"{metrics['avg_input_tokens']:.0f} ({metrics['min_input_tokens']:.0f}-{metrics['max_input_tokens']:.0f})",
                    "Output Tokens": f"{metrics['avg_output_tokens']:.0f} ({metrics['min_output_tokens']:.0f}-{metrics['max_output_tokens']:.0f})",
                    "Compile Time": f"{metrics['avg_compile_time']:.1f} ({metrics['min_compile_time']:.1f}-{metrics['max_compile_time']:.1f})",
                    "Total Time": f"{metrics['avg_total_time']:.1f} ({metrics['min_total_time']:.1f}-{metrics['max_total_time']:.1f})",
                }

                summary_data.append(row)

            # Add separator after each model
            summary_data.append({})

    # Print the summary table
    print(f"\n{'='*100}")
    print("GSM8K Benchmark Results")
    print(f"{'='*100}")
    table_output = tabulate(
        summary_data,
        headers="keys",
        tablefmt="latex" if args.latex else "github",
        showindex=False,
    )
    # Replace placeholder with LaTeX pm symbol
    if args.latex:
        table_output = table_output.replace("PLUSMINUS", "$\\pm$")
    print(table_output)


if __name__ == "__main__":
    main()

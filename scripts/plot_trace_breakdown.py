import json
import os
from collections import defaultdict
from fileinput import filename
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tap import Tap

# Methods to include in the plot
METHODS = [
    "interpreter_base",
    "jit_base_5",
    "jit_base_cache_5",
    "jit_base_nodiscard_5",
    "jit_base_nodiscard_cache_5",
    "jit_base_10",
    "jit_base_nodiscard_10",
    # v1
    "jit_base_5_v1",
    "jit_base_cache_5_v1",
    # v2
    "jit_base_cache_5_v2",
    # json
    "interpreter_base_json",
    "jit_base_json_5",
    "manual",
]


class ArgumentParser(Tap):
    output_dir: Path  # Directory containing trace files (e.g., benchmarks/results/final)
    output_plot: Path  # Path to save the generated plot (e.g., time_breakdown.pdf)


def discover_trace_files(base_results_dir: Path) -> List[Tuple[str, str, str, Path, Path]]:
    """Discover all trace files and their corresponding results files in the nested directory structure.

    Args:
        base_results_dir: Base directory containing results (e.g., benchmarks/results/final)

    Returns:
        List of tuples: (provider, model_name, method, trace_file_path, results_file_path)
    """
    trace_files = []

    if not base_results_dir.exists():
        print(f"Warning: Base results directory {base_results_dir} does not exist")
        return trace_files

    # Traverse provider/model/method structure
    for provider_dir in base_results_dir.iterdir():
        if not provider_dir.is_dir():
            continue

        for model_dir in provider_dir.iterdir():
            if not model_dir.is_dir():
                continue

            for method_dir in model_dir.iterdir():
                if not method_dir.is_dir():
                    continue

                method = method_dir.name

                # Look for trace files and results files in method directory
                trace_file = None
                results_file = None

                for file in method_dir.iterdir():
                    if file.is_file():
                        if file.name.startswith("trace_") and file.name.endswith(".jsonl"):
                            trace_file = file
                        elif file.name == f"results_{method}.jsonl":
                            results_file = file

                # Only add if both trace and results files exist and method is in METHODS list
                if trace_file and results_file:
                    if method in METHODS:
                        provider = provider_dir.name
                        model_name = model_dir.name
                        trace_files.append((provider, model_name, method, trace_file, results_file))
                elif trace_file:
                    print(
                        f"Warning: Found trace file but no results file for {provider_dir.name}/{model_dir.name}/{method}"
                    )

    return trace_files


def parse_trace_times(row):
    """Parse a single trace and extract time metrics for different phases.

    Args:
        trace: List of trace entries with role, time, and tool_calls fields
        total_runtime: Total runtime from results file

    Returns:
        Dictionary with keys:
            - assistant_successful: Time spent in assistant with successful tool calls
            - assistant_discarded: Time spent in assistant with discarded tool calls
            - tool_noncompute: Time spent executing non-compute tools
            - tool_compute: Time spent executing compute tools
            - unaccounted: Difference between total runtime and sum of traced times
    """
    times = {
        "file_name": row["file_name"],
        "run": row["run"],
        "compile_time": row["compile_time"],
        "runtime": row["runtime"],
        "assistant_successful": 0.0,
        "assistant_error": 0.0,
        "assistant_discarded": 0.0,
        "tool_noncompute": 0.0,
        "tool_compute": 0.0,
        "unaccounted": 0.0,
    }

    trace = row["trace"]
    if trace is not None:

        # Build a mapping from tool_call_id to tool name
        tool_call_id_to_name = {}
        for entry in trace:
            if entry["role"] == "assistant" and entry.get("tool_calls"):
                for tool_call in entry["tool_calls"]:
                    tool_call_id_to_name[tool_call["id"]] = tool_call["name"]

        # Build a mapping from tool_call_id to whether it resulted in an error
        tool_call_id_to_error = {}
        for entry in trace:
            if entry["role"] == "tool":
                tool_call_id = entry.get("tool_call_id")
                content = entry.get("content", "")
                if tool_call_id and "Error during tool call" in content:
                    tool_call_id_to_error[tool_call_id] = True

        # Process trace entries
        for entry in trace:
            time_value = entry.get("time")
            if time_value is None:
                continue

            role = entry["role"]

            if role == "assistant":
                # Check if this assistant call has discarded tool calls
                discarded_tool_calls = entry.get("discarded_tool_calls")
                if discarded_tool_calls is not None and len(discarded_tool_calls) > 0:
                    times["assistant_discarded"] += time_value
                else:
                    # Check if any tool calls resulted in errors
                    tool_calls = entry.get("tool_calls", [])
                    has_error = any(tool_call["id"] in tool_call_id_to_error for tool_call in tool_calls)
                    if has_error:
                        times["assistant_error"] += time_value
                    else:
                        times["assistant_successful"] += time_value

            elif role == "tool":
                # Determine if this is a compute tool or not
                tool_call_id = entry.get("tool_call_id")
                if tool_call_id and tool_call_id in tool_call_id_to_name:
                    tool_name = tool_call_id_to_name[tool_call_id]
                    if tool_name == "compute":
                        times["tool_compute"] += time_value
                    else:
                        times["tool_noncompute"] += time_value
            else:
                print(entry)

    # Calculate unaccounted time
    recorded_total = (
        times["compile_time"]
        + times["assistant_successful"]
        + times["assistant_error"]
        + times["assistant_discarded"]
        + times["tool_noncompute"]
        + times["tool_compute"]
    )

    total_time = times["compile_time"] + times["runtime"]

    times["unaccounted"] = max(0.0, total_time - recorded_total)

    return pd.Series(times)


def aggregate_trace_data(trace_files: List[Tuple[str, str, str, Path, Path]]) -> pd.DataFrame:
    """Parse all trace files and aggregate time metrics.

    Args:
        trace_files: List of (provider, model_name, method, trace_file_path, results_file_path) tuples

    Returns:
        DataFrame with columns: method, assistant_successful, assistant_discarded,
                                tool_noncompute, tool_compute, unaccounted
    """

    agg_data = []
    for provider, model_name, method, trace_file, results_file in trace_files:
        print(f"Processing: {provider}/{model_name}/{method}/{trace_file.name}")

        # Read results file to get runtime values
        with open(results_file, "r") as f:
            results = pd.read_json(results_file, lines=True)

        with open(trace_file, "r") as f:
            traces = pd.read_json(trace_file, lines=True)

        if traces.empty:
            results["trace"] = None
        else:
            results = (
                results.set_index(["model", "file_name", "run"])
                .join(traces.set_index(["model", "file_name", "run"]))
                .reset_index()
            )
        results = results.apply(parse_trace_times, axis=1)

        row: Dict[str, Any] = {
            "model": f"{provider}/{model_name}",
            "method": method,
        }
        for col in [
            "compile_time",
            "assistant_successful",
            "assistant_error",
            "assistant_discarded",
            "tool_noncompute",
            "tool_compute",
            "unaccounted",
        ]:
            row[col] = results.groupby(["file_name"])[col].mean().mean()

        agg_data.append(row)

    agg_df = pd.DataFrame(agg_data)

    return agg_df


def plot_stacked_bar_chart(data: pd.DataFrame, output_path: Path):
    """Create a stacked bar chart showing time breakdown by method.

    Args:
        data: DataFrame with columns: method, assistant_successful, assistant_discarded,
                                     tool_noncompute, tool_compute, unaccounted
        output_path: Path to save the plot
    """
    if data.empty:
        print("Warning: No data to plot")
        return

    # Sort methods according to METHODS list order
    data["method_order"] = data["method"].apply(lambda x: METHODS.index(x) if x in METHODS else len(METHODS))
    data = data.sort_values("method_order")
    data = data.drop(columns=["method_order"])

    methods = data["method"].tolist()

    # Set up the figure
    fig, ax = plt.subplots(figsize=(5, 4))

    # Define colors for each segment
    colors = {
        "assistant_successful": "#4971AC",  # Blue
        "assistant_error": "#E88F30",  # Orange
        "assistant_discarded": "#DE6059",  # Red
        "tool_noncompute": "#9879a4",  # Purple
        "tool_compute": "#8263A9",  # Yellow/Gold
        "compile_time": "#7CC775",  # Green
        "unaccounted": "#CCCCCC",  # Gray
    }

    # Create the stacked bar chart
    x = np.arange(len(methods))
    width = 0.6

    # Stack bars from bottom to top
    bottom = np.zeros(len(methods))

    bars = []
    labels = [
        "Tool Generation (Successful)",
        "Tool Generation (Error)",
        "Tool Generation (Discarded)",
        "Tool Execution (Others)",
        "Tool Execution (Compute)",
        "Compilation",
        "Unaccounted",
    ]

    for i, (key, label) in enumerate(
        zip(
            [
                "assistant_successful",
                "assistant_error",
                "assistant_discarded",
                "tool_noncompute",
                "tool_compute",
                "compile_time",
                "unaccounted",
            ],
            labels,
        )
    ):
        values = data[key].tolist()
        bar = ax.bar(x, values, width, bottom=bottom, label=label, color=colors[key], alpha=0.85)
        bars.append(bar)
        bottom += np.array(values)

    # Customize the plot
    ax.set_xlabel("Method", fontsize=14)
    ax.set_ylabel("Average Time (seconds)", fontsize=14)
    ax.set_title("Time Breakdown by Method (Averaged Across Runs and Benchmarks)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=11)

    # Apply grid and styling
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Save the plot
    os.makedirs(output_path.parent, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"\nPlot saved to: {output_path}")


def print_summary_statistics(data: pd.DataFrame):
    """Print summary statistics to console.

    Args:
        data: DataFrame with time breakdown by method
    """
    print("\n" + "=" * 80)
    print("Time Breakdown Summary Statistics")
    print("=" * 80)
    print("\nAverage times (in seconds) across runs and benchmarks:\n")

    # Format data for display
    display_data = data.copy()
    display_data["Total"] = (
        display_data["assistant_successful"]
        + display_data["assistant_error"]
        + display_data["assistant_discarded"]
        + display_data["tool_noncompute"]
        + display_data["tool_compute"]
        + display_data["unaccounted"]
        + display_data["compile_time"]
    )

    # Reorder columns for display
    display_columns = [
        "method",
        "compile_time",
        "assistant_successful",
        "assistant_error",
        "assistant_discarded",
        "tool_noncompute",
        "tool_compute",
        "unaccounted",
        "Total",
    ]

    # Rename columns for better display
    display_data = display_data[display_columns].rename(
        columns={
            "method": "Method",
            "compile_time": "Compilation",
            "assistant_successful": "Tool Generation (Success)",
            "assistant_error": "Tool Generation (Error)",
            "assistant_discarded": "Tool Generation (Discard)",
            "tool_noncompute": "Tool Execution (Others)",
            "tool_compute": "Tool Execution (Compute)",
            "unaccounted": "Unaccounted",
        }
    )  # type:ignore

    print(display_data.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n" + "=" * 80)


def main():
    args = ArgumentParser().parse_args()

    print(f"Discovering trace files in: {args.output_dir}")

    # Discover all trace files
    trace_files = discover_trace_files(base_results_dir=args.output_dir)

    if not trace_files:
        print(f"No trace files found in {args.output_dir}")
        return

    print(f"Found {len(trace_files)} trace files")

    # Parse and aggregate trace data
    data = aggregate_trace_data(trace_files=trace_files)

    if data.empty:
        print("No data to process")
        return

    # Print summary statistics
    print_summary_statistics(data=data)

    # Create visualization
    plot_stacked_bar_chart(data=data, output_path=args.output_plot)


if __name__ == "__main__":
    main()

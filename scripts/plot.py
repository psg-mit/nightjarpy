import json
import os
from copy import copy
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas._typing import Scalar
from tabulate import SEPARATING_LINE, tabulate
from tap import Tap


class ArgumentParser(Tap):
    results_dir: Optional[Path] = None  # Base directory containing the results (e.g., benchmarks/results/final)
    benchmark_suite: str = "interop"  # Which benchmark suite: "interop" or "cpython"
    latex: bool = False  # Whether to print the table in LaTeX format
    output_plot: Optional[Path] = None  # Directory to save plots


TABLE_METHODS = [
    "manual",
    "manual_code_t1p0",
    "interpreter_base",
    "jit_base_5_v4",
    "jit_base_cache_5_v4",
    "jit_base_cache_parallel_5_v4",
    "jit_base_10_v4",
    "jit_base_cache_10_v4",
    "jit_base_cache_parallel_10_v4",
    "jit_basenocompute_cache_5",
    "jit_basenocompute_cache_10",
    "compiler_base",
    "compiler_base_cache",
    "compiler_base_cache_parallel",
    "compiler_python",
    "interpreter_base_noreg_json",
    "interpreter_python_json",
    "interpreter_python_cache_json",
    "interpreter_python_eager_cache_json",
    "interpreter_python_eager_cache_json_t0p0",
    "interpreter_python_base_isolated_json_t1p0",
    "interpreter_python_base_json_t1p0",
    # "manual",
    # "interpreter_base",
    # "interpreter_base_cache",
    # "interpreter_base_json",
    # # "interpreter_base_cache_json",
    # "jit_base_5",
    # "jit_base_cache_5",
    # "jit_base_nodiscard_5",
    # "jit_base_nodiscard_cache_5",
    # # "jit_base_10",
    # # "jit_base_nodiscard_10",
    # # "jit_base_nodiscard_cache_10",
    # # v1
    # "jit_base_5_v1",
    # "jit_base_cache_5_v1",
    # # v2
    # "jit_base_cache_5_v2",
    # # v3
    # "jit_base_cache_5_v3",
    # # v5
    # "jit_base_cache_parallel_10_v5",
    # # no compute
    # "jit_basenocompute_cache_5",
    # # json
    # # "jit_base_json_1",
    # "jit_base_json_5",
    # # "jit_base_json_10",
    # # "jit_base_json_nodiscard_5",
    # # "jit_base_json_nodiscard_10",
    # # "jit_base_cache_json_1",
    # "jit_base_cache_json_5",
    # # "jit_base_cache_json_10",
    # "compiler_base_json",
    # "compiler_aot",
    # "compiler_aot_source",
    # "interpreter_bytecode",
    # "interpreter_python",
    # "interpreter_python_eager",
    # "interpreter_python_var_eager",
    # "interpreter_jit_python_eager",
    # "interpreter_jit_python_eager_iter1",
    # "interpreter_python_eager_effectcount_300",
    # "interpreter_python_eager_effectcount_20",
    # "interpreter_bytecode_eager",
]


PLOT_METHODS = [
    "manual",
    "manual_code_t1p0",
    # "compiler_python",
    "interpreter_base_noreg_json",
    # "interpreter_python_json",
    # "interpreter_python_cache_json",
    # "interpreter_python_base_isolated_json_t1p0",
    # "interpreter_python_base_json_t1p0",
    "interpreter_python_eager_cache_json",
    # "interpreter_base",
    # "jit_base_5_v4",
    # "jit_base_cache_5_v4",
    # "jit_base_cache_parallel_5_v4",
    # "jit_base_10_v4",
    # "jit_base_cache_10_v4",
    # "jit_base_cache_parallel_10_v4",
    # "jit_basenocompute_cache_5",
    # "jit_basenocompute_cache_10",
    # "compiler_base",
    # "compiler_base_cache",
    # "compiler_base_cache_parallel",
    # "manual",
    # "interpreter_base",
    # "interpreter_base_cache",
    # "interpreter_base_json",
    # # "interpreter_base_cache_json",
    # "jit_base_5",
    # "jit_base_cache_5",
    # "jit_base_nodiscard_5",
    # "jit_base_nodiscard_cache_5",
    # # "jit_base_10",
    # # "jit_base_nodiscard_10",
    # # "jit_base_nodiscard_cache_10",
    # # v1
    # "jit_base_5_v1",
    # "jit_base_cache_5_v1",
    # # v2
    # "jit_base_cache_5_v2",
    # # "jit_base_json_1",
    # "jit_base_json_5",
    # # "jit_base_json_10",
    # # "jit_base_json_nodiscard_5",
    # # "jit_base_json_nodiscard_10",
    # # "jit_base_cache_json_1",
    # "jit_base_cache_json_5",
    # # "jit_base_cache_json_10",
    # "compiler_base_json",
    # "Compiler - Python Code",
    # "Compiler* - Python",
    # "Interpreter - Byte-Code Effects",
    # "Interpreter - Python-Code Effects",
    # "Interpreter - Python-Code Effects (Eager)",
    # "Interpreter - Python-Code (VAR) Effects (Eager)",
    # "Interpreter - JIT Python-Code Effects (Eager)",
    # "Interpreter - JIT Python-Code Effects (Eager)  (1 iter)",
]


MODEL_MAPPING = {
    "openai/gpt-4.1-2025-04-14": "GPT-4.1",
    "vllm/gpt-oss-20b": "GPT-OSS 20B",
    "anthropic/claude-sonnet-4-20250514": "Sonnet 4",
    "anthropic/claude-sonnet-4-5-20250929": "Sonnet 4.5",
}

MODEL_COLORS = {
    "openai/gpt-4.1-2025-04-14": "#DE6059",
    "anthropic/claude-sonnet-4-20250514": "#4971AC",
    "anthropic/claude-sonnet-4-5-20250929": "#9879a4",
    "vllm/gpt-oss-20b": "#84BE6D",
}

METHOD_MAPPING = {
    "interpreter_base": "Interpreter - Base",
    "jit_base_5_v4": "JIT - Base 5",
    "jit_base_cache_5_v4": "JIT - Base 5 (Cache)",
    "jit_base_cache_parallel_5_v4": "JIT - Base 5 (Cache) (Parallel)",
    "jit_base_10_v4": "JIT - Base 10",
    "jit_base_cache_10_v4": "JIT - Base 10 (Cache)",
    "jit_base_cache_parallel_10_v4": "JIT - Base 10 (Cache) (Parallel)",
    "jit_basenocompute_cache_5": "JIT - Base-NoCompute 5 (Cache) (Parallel)",
    "jit_basenocompute_cache_10": "JIT - Base-NoCompute 10 (Cache) (Parallel)",
    "compiler_base": "Compiler - Base",
    "compiler_python": "Compiler - Python",
    "manual": "Manual",
    "manual_code_t1p0": "Manual (Code Interpreter)",
    "manual_t0": "Manual (t=0)",
    # "interpreter_base": "Interpreter - Base",
    # "interpreter_base_cache": "Interpreter - Base (Cache-Enabled)",
    # "jit_base_json_1": "JIT - Base JSON (1 Steps Ahead)",
    # "jit_base_5": "JIT - Base (5 Steps Ahead)",
    # "jit_base_cache_5": "JIT - Base (5 Steps Ahead) (Cache-Enabled)",
    # "jit_base_nodiscard_5": "JIT - Base (5 Steps Ahead/No Discard)",
    # "jit_base_nodiscard_cache_5": "JIT - Base (5 Steps Ahead/No Discard) (Cache-Enabled)",
    # "jit_base_10": "JIT - Base (10 Steps Ahead)",
    # "jit_base_nodiscard_10": "JIT - Base (10 Steps Ahead/No Discard)",
    # "jit_base_nodiscard_cache_10": "JIT - Base (10 Steps Ahead/No Discard) (Cache-Enabled)",
    # # v1
    # "jit_base_5_v1": "JIT - Base V1 (5 Steps Ahead)",
    # "jit_base_cache_5_v1": "JIT - Base V1 (5 Steps Ahead) (Cache-Enabled)",
    # # v2
    # "jit_base_cache_5_v2": "JIT - Base V2 (5 Steps Ahead) (Cache-Enabled)",
    # # v3
    # "jit_base_cache_5_v3": "JIT - Base V3 (5 Steps Ahead) (Cache-Enabled)",
    # # no compute
    # "jit_basenocompute_cache_5": "JIT - Base (No Compute) (5 Steps Ahead) (Cache-Enabled)",
    # # json
    # "interpreter_base_json": "Interpreter - Base JSON",
    "interpreter_python_json": "Interpreter - Python JSON",
    # "interpreter_python_eager_cache_json": "Interpreter - Python-Optimized JSON",
    # "interpreter_base_noreg_json": "Interpreter - Base (No Reg) JSON",
    "interpreter_python_cache_json": "Interpreter - Python-Caching JSON",
    "interpreter_python_eager_cache_json": "Nightjar",
    "interpreter_base_noreg_json": "Nightjar (Naive)",
    # "interpreter_base_cache_json": "Interpreter - Base JSON (Cache-Enabled)",
    # "jit_base_json_5": "JIT - Base JSON (5 Steps Ahead)",
    # "jit_base_json_10": "JIT - Base JSON (10 Steps Ahead)",
    # "jit_base_json_20": "JIT - Base JSON (20 Steps Ahead)",
    # "jit_base_json_nodiscard_5": "JIT - Base JSON (5 Steps Ahead/No Discard)",
    # "jit_base_json_nodiscard_10": "JIT - Base JSON (10 Steps Ahead/No Discard)",
    # "jit_base_cache_json_1": "JIT - Base JSON (1 Steps Ahead) (Cache-Enabled)",
    # "jit_base_cache_json_5": "JIT - Base JSON (5 Steps Ahead) (Cache-Enabled)",
    # "jit_base_cache_json_20": "JIT - Base JSON (20 Steps Ahead) (Cache-Enabled)",
    # "compiler_json_base": "Compiler - Base Effects",
    # "interpreter_bytecode": "Interpreter - Byte-Code Effects",
    # "interpreter_python": "Interpreter - Python-Code Effects",
    # "interpreter_python_eager": "Interpreter - Python-Code Effects (Eager)",
    # "interpreter_python_var_eager": "Interpreter - Python-Code (VAR) Effects (Eager)",
    # "interpreter_jit_python_eager": "Interpreter - JIT Python-Code Effects (Eager)",
    # "interpreter_jit_python_eager_iter1": "Interpreter - JIT Python-Code Effects (Eager)  (1 iter)",
    # "interpreter_python_eager_effectcount_300": "Interpreter - Python-Code Effects (Eager) (Effect Count)",
    # "interpreter_python_eager_effectcount_20": "Interpreter - Python-Code Effects (Eager) (Effect Count 20)",
    # "interpreter_bytecode_eager": "Interpreter - Byte-Code Effects (Eager)",
    # "compiler_aot": "Compiler - Python Code",
    # "compiler_aot_source": "Compiler* - Python",
}

METHOD_COLORS = {
    "interpreter_base": "#D33B33",
    "jit_base_5_v4": "#EF8558",
    "jit_base_cache_5_v4": "#DAB244",
    "jit_base_cache_parallel_5_v4": "#A2B938",
    "jit_base_10_v4": "#4F7F35",
    "jit_base_cache_10_v4": "#1A5F2C",
    "jit_base_cache_parallel_10_v4": "#24B2A4",
    "jit_basenocompute_cache_5": "#306E95",
    "jit_basenocompute_cache_10": "#2A3A7B",
    "compiler_base": "#6D4ECA",
    "manual": "#3C3C3C",
    "manual_code_t1p0": "#EBB339",
    # "compiler_python",
    "interpreter_base_noreg_json": "#C95454",
    "interpreter_python_base_isolated_json_t1p0": "#DAB244",
    "interpreter_python_base_json_t1p0": "#4F7F35",
    # "interpreter_python_json",
    # "interpreter_python_cache_json",
    "interpreter_python_eager_cache_json": "#306E95",
    # "manual": "#DE6059",
    # "interpreter_base": "#4971AC",
    # "interpreter_base_cache": "#344797",
    # "jit_base_5": "#D570E4",
    # "jit_base_cache_5": "#A7389E",
    # "jit_base_nodiscard_5": "#8E224A",
    # "jit_base_nodiscard_cache_5": "#49272A",
    # "jit_base_10": "#5EE0B0",
    # "jit_base_nodiscard_10": "#42C1C5",
    # "jit_base_nodiscard_cache_10": "#467E8F",
    # # v1
    # "jit_base_5_v1": "#E0C65E",
    # "jit_base_cache_5_v1": "#C28F3D",
    # "jit_base_cache_5_v2": "#2D7E3E",
    # # json
    # "interpreter_base_json": "#9baa20",
    # "interpreter_base_cache_json": "#57992B",
    # "jit_base_json_1": "#9baa20",
    # "jit_base_json_5": "#57992B",
    # "jit_base_json_10": "#2D7E3E",
    # "jit_base_json_nodiscard_5": "#B73677",
    # "jit_base_json_nodiscard_10": "#9B2246",
    # "jit_base_cache_json_1": "#D852D6",
    # "jit_base_cache_json_5": "#B73677",
    # "jit_base_cache_json_10": "#9B2246",
    # "compiler_base_json": "#fbc75a",
    # "Compiler - Python Code": "#fbc75a",
    # "Interpreter - Byte-Code Effects": "#4971AC",
    # "Interpreter - Python-Code Effects": "#9879a4",
    # "Interpreter - Python-Code Effects (Eager)": "#4FC4C8",
    # "Interpreter - Python-Code (VAR) Effects (Eager)": "#5A286E",
}


# Define shapes for different methods (only those in PLOT_METHODS)
MODEL_MARKERS = {
    "openai/gpt-4.1-2025-04-14": "o",
    "anthropic/claude-sonnet-4-20250514": "v",
    "anthropic/claude-sonnet-4-5-20250929": "s",
    "vllm/gpt-oss-20b": "s",
}


def load_metadata(benchmark_suite: str = "interop") -> Tuple[Dict, str]:
    # Determine metadata filename based on benchmark suite
    if benchmark_suite == "cpython":
        metadata_filename = "benchmarks_cpython/metadata.json"
        results_column = "test_results"
    elif benchmark_suite == "example":
        metadata_filename = "benchmarks/metadata.json"  # Use interop metadata for example
        results_column = "hard_eval"
    else:  # interop
        metadata_filename = "benchmarks/metadata.json"
        results_column = "hard_eval"

    # Try multiple possible paths to support running from root or scripts/ directory
    possible_paths = [
        Path(metadata_filename),  # From root directory
        Path("..") / metadata_filename,  # From scripts/ directory
    ]

    metadata_path = None
    for path in possible_paths:
        if path.exists():
            metadata_path = path
            break

    if metadata_path is None:
        raise FileNotFoundError(f"Metadata file not found. Tried: {', '.join(str(p) for p in possible_paths)}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata, results_column


def discover_models(base_results_dir: Path) -> List[Tuple[str, str, Path]]:
    """Discover all models in the nested directory structure.

    Args:
        base_results_dir: Base directory containing results (e.g., benchmarks/results/final)

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

            # Check if this directory contains files (methods) or subdirectories (models)
            subdirs = [subdir for subdir in item.iterdir() if subdir.is_dir()]
            files = [f for f in item.iterdir() if f.is_file() and f.name.endswith(".jsonl")]

            # If it has files, it's a method directory - skip it
            # If it has subdirectories but no files, it's a model directory
            has_files = len(files) > 0

            print(files)

            if has_files:
                # This is a method directory (contains files) - skip it
                continue
            elif subdirs:
                # Check if any of the subdirectories contain method directories
                has_method_subdirs = False
                for subdir in subdirs:
                    subdir_contents = [subsubdir for subsubdir in subdir.iterdir() if subsubdir.is_dir()]
                    subdir_files = [f for f in subdir.iterdir() if f.is_file() and f.name.endswith(".jsonl")]
                    if subdir_files:  # If subdir has files, it's a method directory
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
            else:
                # This is an empty directory - skip it
                continue

        return found_models

    # Start the recursive search
    models = find_innermost_models(base_results_dir)
    return models


def apply_common_plot_styling(alpha: float = 0.3, axis: Literal["x", "y", "both"] = "both", fontsize: int = 12):
    """Apply common styling to plots including grid and spine visibility."""
    plt.grid(True, alpha=alpha, axis=axis)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tick_params(axis="both", which="major", labelsize=fontsize)


def apply_subplot_styling(ax, alpha: float = 0.3, fontsize: int = 13):
    """Apply common styling to subplots including grid and spine visibility."""
    ax.grid(True, alpha=alpha)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)


def save_plot(output_path: Path):
    """Save the current plot to the specified path with proper directory creation."""
    os.makedirs(output_path.parent, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def get_model_display_name(provider: str, model_name: str) -> str:
    """Get a display name for the model based on provider and model name."""
    full_name = f"{provider}/{model_name}"
    return MODEL_MAPPING.get(full_name, model_name)


def darken_color(hex_color: str, factor: float = 0.7) -> str:
    """Darken a hex color by multiplying RGB values by a factor.

    Args:
        hex_color: Hex color string (e.g., "#FF0000")
        factor: Darkening factor (0.0 = black, 1.0 = original color)

    Returns:
        Darkened hex color string
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip("#")

    # Convert to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Darken by multiplying by factor
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)

    # Convert back to hex
    return f"#{r:02x}{g:02x}{b:02x}"


def pass_at_k(n: int, c: float, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def compute_time_stats(results: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Compute runtime statistics for the given results.

    Args:
        results: DataFrame containing test results with runtime field

    Returns:
        Tuple of (mean_runtime, min_runtime, max_runtime, std_runtime) in seconds
    """
    # Get all runtimes across all benchmarks and runs (using runtime, not total_time)
    if "runtime" not in results.columns:
        return np.nan, np.nan, np.nan, np.nan

    # Filter out null times
    results = results[results["runtime"].notna()]  # type: ignore

    if results.empty:
        return np.nan, np.nan, np.nan, np.nan

    # First compute average runtime per question across runs
    question_avg_runtimes = results.groupby("file_name")["runtime"].mean()

    # Then compute statistics across the question averages
    mean_runtime = np.mean(question_avg_runtimes)
    std_runtime = np.std(question_avg_runtimes)

    # global min and max
    min_runtime = np.min(results["runtime"])
    max_runtime = np.max(results["runtime"])
    return float(mean_runtime), float(min_runtime), float(max_runtime), float(std_runtime)


def compute_tool_call_stats(results: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Compute tool call statistics for the given results.

    Args:
        results: DataFrame containing test results with n_tool_calls field

    Returns:
        Tuple of (mean_tool_calls, min_tool_calls, max_tool_calls, std_tool_calls)
    """
    # Get all tool calls across all benchmarks and runs
    if "n_tool_calls" not in results.columns:
        return np.nan, np.nan, np.nan, np.nan

    # Filter out null tool calls
    results = results[results["n_tool_calls"].notna()]  # type: ignore

    if results.empty:
        return np.nan, np.nan, np.nan, np.nan

    # First compute average tool calls per question across runs
    question_avg_tool_calls = results.groupby("file_name")["n_tool_calls"].mean()

    # Then compute statistics across the question averages
    mean_tool_calls = np.mean(question_avg_tool_calls)
    std_tool_calls = np.std(question_avg_tool_calls)

    # global min and max
    min_tool_calls = np.min(results["n_tool_calls"])
    max_tool_calls = np.max(results["n_tool_calls"])
    return float(mean_tool_calls), float(min_tool_calls), float(max_tool_calls), float(std_tool_calls)


def compute_pass_at_k(results: pd.DataFrame, k: int, metadata: Dict, results_column: str) -> Tuple[float, float, int]:
    """Compute the pass@k metric for the given results."""
    # Check if we have enough runs
    n = int(results.groupby("file_name")["run"].nunique().max())
    min_n = results.groupby("file_name")["run"].nunique().min()
    if n < k or min_n < k:
        return np.nan, np.nan, n

    pass_at_k_data = []
    partial_pass_at_k_data = []

    # Only process programs that are actually present in the results data
    program_list = results["file_name"].unique().tolist()

    for program in program_list:
        for test in metadata["tests"][program]:
            c_pc = 0
            for run_i in range(n):
                row: pd.DataFrame = results[(results["file_name"] == program) & (results["run"] == run_i)]  # type: ignore
                if len(row) == 0:
                    if k == n:
                        print(f"Warning: no results for program `{program}` on run `{run_i}`")
                    return np.nan, np.nan, n
                if test not in row[results_column].values[0].keys():
                    # if k == n:
                    #     print(f"Warning: test `{test}` not found in results for program `{program}` on run `{run_i}`")
                    continue
                if row[results_column].values[0][test]:
                    c_pc += 1
            partial_pass_at_k_data.append(pass_at_k(n, c_pc, k))

        c = 0
        for run_i in range(n):
            row = results[(results["file_name"] == program) & (results["run"] == run_i)]  # type: ignore
            if len(row) == 0:
                if k == n:
                    print(f"Warning: no results for program `{program}` on run `{run_i}`")
                return np.nan, np.nan, n

            all_correct = True
            for test in metadata["tests"][program]:
                if test not in row[results_column].values[0].keys():
                    # if k == n:
                    #     print(f"Warning: test `{test}` not found in results for program `{program}` on run `{run_i}`")
                    all_correct = False
                    break
                if not row[results_column].values[0][test]:
                    all_correct = False
                    break
            if all_correct:
                c += 1
        pass_at_k_data.append(pass_at_k(n, c, k))

    partial_pass_rate = np.mean(partial_pass_at_k_data)
    pass_rate = np.mean(pass_at_k_data)

    return float(pass_rate), float(partial_pass_rate), n


def avg_score(results: pd.DataFrame, metadata: Dict, results_column: str) -> pd.DataFrame:
    n_runs = int(results.groupby("file_name")["run"].nunique().max())

    program_list = results["file_name"].unique().tolist()

    scores = []

    for program in program_list:
        for run_i in range(n_runs):
            row = results[(results["file_name"] == program) & (results["run"] == run_i)]
            assert isinstance(row, pd.DataFrame)
            if len(row) == 0:
                print(f"Warning: no results for program `{program}` on run `{run_i}`")
                continue
            test_scores = []
            for test in metadata["tests"][program]:
                if test not in row[results_column].values[0].keys():
                    test_scores.append(0)
                    continue
                if row[results_column].values[0][test]:
                    test_scores.append(1)
                else:
                    test_scores.append(0)

            row_val = {
                "file_name": program,
                "run": run_i,
                "pass_rate": sum(test_scores) / len(test_scores),
            }
            scores.append(row_val)

    scores_df = pd.DataFrame(scores)
    return scores_df


def get_method_from_dir(dir_name: str) -> str:
    """Extract method information from directory name.

    Returns:
        Method name (e.g. "Zero", "Few")
    """

    return METHOD_MAPPING.get(dir_name, "Default")


def get_model_name(model_name: str) -> str:
    """Get model name from directory name."""
    return MODEL_MAPPING.get(model_name, model_name)


def clean_test_name(name: Union[str, Scalar]) -> str:
    """Clean test name for better display."""
    # Remove common prefixes and numbers
    return str(name).replace("test_", "")


def compute_metrics(data: pd.DataFrame) -> pd.DataFrame:
    data["total_time"] = data["runtime"] + data["compile_time"]
    data["runtime_per_tool"] = data["runtime"] / data["n_tool_calls"]

    def get_token_count(x, key):
        """Get token count handling both nested and flat structures."""
        if x is None or not isinstance(x, dict):
            return pd.NA
        # Check if it has nested structure (compile/runtime keys)
        if "compile" in x or "runtime" in x:
            compile_val = x.get("compile", {}).get(key, 0) or 0
            runtime_val = x.get("runtime", {}).get(key, 0) or 0
            return compile_val + runtime_val
        # Otherwise use flat structure
        return x.get(key, pd.NA)

    def get_cached_tokens(x):
        """Get cached token count, handling different field names."""
        if x is None or not isinstance(x, dict):
            return 0
        # Check nested structure first
        if "compile" in x or "runtime" in x:
            compile_cached = x.get("compile", {}).get("cached_input_tokens", 0) or 0
            runtime_cached = x.get("runtime", {}).get("cached_input_tokens", 0) or 0
            return compile_cached + runtime_cached
        # Check for flat structure with cached_token_reads
        if "cached_token_reads" in x:
            return x.get("cached_token_reads", 0) or 0
        # Check for cached_input_tokens in flat structure
        return x.get("cached_input_tokens", 0) or 0

    data["input_tokens"] = data["token_count"].apply(lambda x: get_token_count(x, "input_tokens"))  # type: ignore
    data["uncached_input_tokens"] = data["token_count"].apply(lambda x: get_token_count(x, "input_tokens") - get_cached_tokens(x) if not pd.isna(get_token_count(x, "input_tokens")) else pd.NA)  # type: ignore
    data["output_tokens"] = data["token_count"].apply(lambda x: get_token_count(x, "output_tokens"))  # type: ignore

    return data


def plot_benchmark_line_plots(
    raw_data: List[pd.DataFrame],
    method_order: List[str],
    model: str,
    output_path: Path,
    metadata: Dict,
    results_column: str,
    benchmark_suite: str,
    custom_title: Optional[str] = None,
):
    """Create line plots showing average pass rate for each method across benchmarks.

    Args:
        raw_data: List of DataFrames containing raw results
        method_order: List of methods in desired order
        model: Model name to filter results for
        output_path: Path to save the plot
        metadata: Metadata dictionary
        results_column: Name of the results column
        benchmark_suite: Which benchmark suite this is for
        custom_title: Optional custom title for the plot
    """
    # Get list of benchmarks
    if benchmark_suite == "example":
        benchmark_list = ["graph"]
    else:
        benchmark_list = list(metadata["tests"].keys())

    # Filter data for the specific model
    model_data_list = []
    for data in raw_data:
        if "model" in data.columns and not data.empty and data["model"].iloc[0] == model:
            model_data_list.append(data)

    if not model_data_list:
        print(f"Warning: No data found for model {model}")
        return

    # Combine all data for this model
    combined_data = pd.concat(model_data_list, ignore_index=True)

    # Calculate scores for each benchmark and method
    plot_data = []
    for method in method_order:
        method_data = combined_data[combined_data["method"] == method]

        for benchmark in benchmark_list:
            benchmark_data = method_data[method_data["file_name"] == benchmark]
            assert isinstance(benchmark_data, pd.DataFrame)
            if benchmark_data.empty:
                # No data for this benchmark-method combination, add with 0 score
                plot_data.append(
                    {
                        "Benchmark": clean_test_name(str(benchmark)),
                        "Method": method,
                        "Score": 0.0,
                        "Min": 0.0,
                        "Max": 0.0,
                        "Runtime": 0.0,
                        "Runtime_Min": 0.0,
                        "Runtime_Max": 0.0,
                        "Benchmark_Index": benchmark_list.index(benchmark),
                    }
                )
                continue

            # Calculate scores for each run
            scores = []
            n_runs = int(benchmark_data["run"].nunique())

            for run_i in range(n_runs):
                run_data = benchmark_data[benchmark_data["run"] == run_i]
                if run_data.empty:
                    continue

                # Calculate score for this run
                test_scores = []
                for test in metadata["tests"][benchmark]:
                    if test not in run_data[results_column].values[0].keys():  # type: ignore
                        test_scores.append(0)
                        continue
                    if run_data[results_column].values[0][test]:  # type: ignore
                        test_scores.append(1)
                    else:
                        test_scores.append(0)

                if test_scores:  # Only add score if we have test results
                    scores.append(sum(test_scores) / len(test_scores))

            # Always add data for this benchmark-method combination
            # If no valid scores, use 0 as default
            if scores:
                mean_score = np.mean(scores)
                min_score = np.min(scores)
                max_score = np.max(scores)
            else:
                mean_score = 0.0
                min_score = 0.0
                max_score = 0.0

            # Calculate runtime statistics
            if not benchmark_data.empty and "total_time" in benchmark_data.columns:
                # Calculate total_time if not already present
                if "total_time" not in benchmark_data.columns:
                    benchmark_data["total_time"] = benchmark_data["compile_time"] + benchmark_data["runtime"]

                runtimes = benchmark_data["total_time"].dropna()
                if not runtimes.empty:
                    mean_runtime = np.mean(runtimes)
                    min_runtime = np.min(runtimes)
                    max_runtime = np.max(runtimes)
                else:
                    mean_runtime = 0.0
                    min_runtime = 0.0
                    max_runtime = 0.0
            else:
                mean_runtime = 0.0
                min_runtime = 0.0
                max_runtime = 0.0

            plot_data.append(
                {
                    "Benchmark": clean_test_name(str(benchmark)),
                    "Method": method,
                    "Score": mean_score,
                    "Min": min_score,
                    "Max": max_score,
                    "Runtime": mean_runtime,
                    "Runtime_Min": min_runtime,
                    "Runtime_Max": max_runtime,
                    "Benchmark_Index": benchmark_list.index(benchmark),
                }
            )

    if not plot_data:
        print(f"Warning: No plot data found for model {model}")
        return

    plot_df = pd.DataFrame(plot_data)

    # Create subplots with 2 rows: pass rate and runtime
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Get unique methods for plotting
    methods = plot_df["Method"].unique().tolist()

    # Create line plots for each method on both subplots
    for method in methods:
        method_data = plot_df[plot_df["Method"] == method]
        assert isinstance(method_data, pd.DataFrame)
        if method_data.empty:
            continue

        # Sort by benchmark index to ensure proper line ordering
        method_data = method_data.sort_values("Benchmark_Index")

        color = METHOD_COLORS.get(method, "#666666")

        # Extract data for plotting
        x = method_data["Benchmark_Index"]

        # Plot 1: Pass Rate
        y_score = method_data["Score"]
        y_min = method_data["Min"]
        y_max = method_data["Max"]

        ax1.plot(x, y_score, color=color, linewidth=2, marker="o", markersize=4, label=method)
        # ax1.fill_between(x, y_min, y_max, color=color, alpha=0.2)

        # Plot 2: Runtime
        y_runtime = method_data["Runtime"]
        y_runtime_min = method_data["Runtime_Min"]
        y_runtime_max = method_data["Runtime_Max"]

        ax2.plot(x, y_runtime, color=color, linewidth=2, marker="o", markersize=4, label=method)
        # ax2.fill_between(x, y_runtime_min, y_runtime_max, color=color, alpha=0.2)

    # Customize the first subplot (Pass Rate)
    ax1.set_xlabel("Benchmark Index", fontsize=14)
    ax1.set_ylabel("Average Pass Rate", fontsize=14)
    ax1.set_xticks(range(len(benchmark_list)))
    ax1.set_xticklabels([clean_test_name(b) for b in benchmark_list], rotation=45, ha="right")
    ax1.set_ylim(0, 1.1)
    apply_subplot_styling(ax1, alpha=0.3, fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=11)

    # Customize the second subplot (Runtime)
    ax2.set_xlabel("Benchmark Index", fontsize=14)
    ax2.set_ylabel("Average Runtime (seconds)", fontsize=14)
    ax2.set_xticks(range(len(benchmark_list)))
    ax2.set_xticklabels([clean_test_name(b) for b in benchmark_list], rotation=45, ha="right")
    apply_subplot_styling(ax2, alpha=0.3, fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=11)

    # Set overall title
    if custom_title:
        fig.suptitle(custom_title, fontsize=16)
    else:
        suite_title = "InteropBench" if benchmark_suite == "interop" else "PythonTestBench"
        fig.suptitle(f"Performance by Method Across Benchmarks - {model} - {suite_title}", fontsize=16)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    save_plot(output_path)


def plot_time_vs_pass_rate(
    data: pd.DataFrame,
    output_path: Path,
):
    """Create scatter plots showing runtime/compile time/total time vs average pass rate for each method and model.

    Args:
        data: DataFrame containing all results with runtime and pass rate data
        output_path: Path to save the plot
        benchmark_suite: Which benchmark suite this is for
        custom_title: Optional custom title for the plot
    """

    plot_data = []
    for model in data["model"].unique():
        for method in PLOT_METHODS:
            method_data = data[(data["method"] == method) & (data["model"] == model)]

            pass_rate = method_data.groupby(["run"])["pass_rate"].mean().mean()
            runtime = method_data.groupby(["file_name"])["runtime"].mean().mean()
            compile_time = method_data.groupby(["file_name"])["compile_time"].mean().mean()
            total_time = method_data.groupby(["file_name"])["total_time"].mean().mean()

            plot_data.append(
                {
                    "model": model,
                    "method": method,
                    "pass_rate": pass_rate,
                    "runtime": runtime,
                    "compile_time": compile_time,
                    "total_time": total_time,
                }
            )

    plot_data_df = pd.DataFrame(plot_data).dropna()

    if plot_data_df.empty:
        print("Warning: No valid numeric data for runtime vs pass rate plot")
        return

    # Create subplots: runtime, compile time, total time
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 2))

    # Create legend handles for each model-method combination
    legend_handles = []
    added_combinations = set()

    # Helper function to plot data on a subplot
    def plot_subplot(ax, x_col, title, xlabel, show_ylabel=False):
        ax.clear()

        # Plot each model-method combination
        for model in data["model"].unique():
            marker = MODEL_MARKERS.get(model, "o")

            for method in PLOT_METHODS:
                method_data = plot_data_df[(plot_data_df["method"] == method) & (plot_data_df["model"] == model)]
                if method_data.empty:
                    continue

                color = METHOD_COLORS.get(method, "#666666")
                x = method_data[x_col].iloc[0]  # type: ignore
                y = method_data["pass_rate"].iloc[0]  # type: ignore

                # Create scatter plot point
                ax.scatter(x, y, c=color, marker=marker, s=100, alpha=0.8, linewidth=0.5)

                # Add to legend handles (only once per combination)
                combination = f"{MODEL_MAPPING.get(model, "N/A")} - {METHOD_MAPPING.get(method, "N/A")}"
                if combination not in added_combinations:
                    legend_handles.append(
                        plt.Line2D(  # pyright: ignore[reportPrivateImportUsage]
                            [0], [0], marker=marker, color=color, linestyle="None", markersize=8, label=combination
                        )
                    )
                    added_combinations.add(combination)

        # Customize the subplot
        ax.set_xlabel(xlabel, fontsize=12)
        if show_ylabel:
            ax.set_ylabel("Average Pass Rate", fontsize=12)
        ax.set_title(title, fontsize=14)

        # Set axis limits with some padding (filter out NaN values for this column)
        valid_x = plot_data_df[x_col]
        if not valid_x.empty:
            x_min, x_max = valid_x.min(), valid_x.max()
            x_range = x_max - x_min
            ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)

        # y_min, y_max = plot_data_df["pass_rate"].min(), plot_data_df["pass_rate"].max()
        # y_range = y_max - y_min
        ax.set_ylim(0, 1)

        # Apply common styling
        apply_subplot_styling(ax, alpha=0.3, fontsize=10)

    # Plot three subplots
    plot_subplot(ax1, "runtime", "Runtime", "Average Runtime (s)", show_ylabel=True)
    plot_subplot(ax2, "compile_time", "Compile Time", "Average Compile Time (s)", show_ylabel=False)
    plot_subplot(ax3, "total_time", "Total Time", "Average Total Time (s)", show_ylabel=False)

    # Set overall title
    # if custom_title:
    #     fig.suptitle(custom_title, fontsize=16)
    # else:
    # suite_title = "SPSBench" if benchmark_suite == "interop" else "PythonTestBench"
    # fig.suptitle(f"Time vs Pass Rate Analysis - {suite_title}", fontsize=16)

    # Add legend with better positioning for more entries
    fig.legend(handles=legend_handles, bbox_to_anchor=(0.5, -0.15), loc="upper center", fontsize=9, ncol=2)

    # Adjust layout to prevent label cutoff
    # plt.tight_layout()

    # Save the plot
    save_plot(output_path)


def plot_benchmark_bar_graph(
    raw_data: pd.DataFrame,
    method_order: List[str],
    model: str,
    output_path: Path,
    metadata: Dict,
    benchmark_suite: str,
    custom_title: Optional[str] = None,
):
    """Create bar graphs showing average pass rate with std deviation for each method across benchmarks.

    Args:
        raw_data: List of DataFrames containing raw results
        method_order: List of methods in desired order
        model: Model name to filter results for
        output_path: Path to save the plot
        metadata: Metadata dictionary
        results_column: Name of the results column
        benchmark_suite: Which benchmark suite this is for
        custom_title: Optional custom title for the plot
    """
    # Get list of benchmarks
    if benchmark_suite == "example":
        benchmark_list = ["graph"]
    else:
        benchmark_list = list(metadata["tests"].keys())
    # Calculate scores for each benchmark and method
    plot_data = []
    for method in method_order:
        method_data = raw_data[raw_data["method"] == method]

        for benchmark in benchmark_list:
            benchmark_data = method_data[method_data["file_name"] == benchmark]
            assert isinstance(benchmark_data, pd.DataFrame)
            if benchmark_data.empty:
                # No data for this benchmark-method combination
                plot_data.append(
                    {
                        "Benchmark": clean_test_name(str(benchmark)),
                        "Method": method,
                        "Score": 0.0,
                        "Std": 0.0,
                        "Benchmark_Index": benchmark_list.index(benchmark),
                    }
                )
                continue

            mean_score = benchmark_data["pass_rate"].mean()
            std_score = benchmark_data["pass_rate"].std()

            plot_data.append(
                {
                    "Benchmark": clean_test_name(str(benchmark)),
                    "Method": method,
                    "Score": mean_score,
                    "Std": std_score,
                    "Benchmark_Index": benchmark_list.index(benchmark),
                }
            )

    if not plot_data:
        print(f"Warning: No plot data found for model {model}")
        return

    plot_df = pd.DataFrame(plot_data)

    # Create figure with appropriate size based on number of benchmarks
    n_benchmarks = len(benchmark_list)
    fig, ax = plt.subplots(figsize=(max(8, n_benchmarks * 0.5), 4))

    # Get unique methods for plotting
    methods = [m for m in method_order if m in plot_df["Method"].unique()]

    # Set up bar positions
    x = np.arange(n_benchmarks)
    width = 0.8 / len(methods)  # width of bars

    # Create bars for each method
    for i, method in enumerate(methods):
        method_data = plot_df[plot_df["Method"] == method]
        assert isinstance(method_data, pd.DataFrame)
        if method_data.empty:
            continue

        # Sort by benchmark index to ensure proper ordering
        method_data = method_data.sort_values("Benchmark_Index")

        color = METHOD_COLORS.get(method, "#666666")
        darker_color = darken_color(color, factor=0.7)

        # Calculate x positions for this method's bars
        x_pos = x + (i - len(methods) / 2 + 0.5) * width

        # Calculate error bars, capping upper bound at 1.0
        scores = method_data["Score"].values
        stds = method_data["Std"].values

        # Lower error: min(std, score) to ensure we don't go below 0
        lower_errors = np.minimum(stds, scores)

        # Upper error: min(std, 1.0 - score) to ensure we don't exceed 1.0
        upper_errors = np.minimum(stds, 1.0 - scores)

        # Combine into asymmetric error array
        asymmetric_errors = np.array([lower_errors, upper_errors])

        # Plot bars with error bars
        ax.bar(
            x_pos,
            scores,
            width,
            yerr=asymmetric_errors,
            color=color,
            label=METHOD_MAPPING.get(method, method),
            capsize=width * 15,  # Make capsize proportional to bar width
            error_kw={"linewidth": 1, "elinewidth": 1, "ecolor": darker_color},
        )

    # Customize the plot
    ax.set_xlabel("Benchmark", fontsize=14)
    ax.set_ylabel("Average Pass Rate", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([clean_test_name(b) for b in benchmark_list], rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    apply_subplot_styling(ax, alpha=0.3, fontsize=12)
    ax.legend(bbox_to_anchor=(0.5, -0.7), loc="upper center", fontsize=11, ncol=2)

    # Set overall title
    if custom_title:
        fig.suptitle(custom_title, fontsize=16)
    else:
        suite_title = "InteropBench" if benchmark_suite == "interop" else "PythonTestBench"
        fig.suptitle(f"{MODEL_MAPPING.get(model, '')}", fontsize=16)

    # Adjust layout to prevent label cutoff
    # plt.tight_layout()

    # Save the plot
    save_plot(output_path)


def plot_tool_calls_vs_runtime(
    data: pd.DataFrame,
    output_path: Path,
):
    """Create scatter plot showing number of tool calls vs runtime for each method and model.
    Each point represents one run of one program.

    Args:
        data: DataFrame containing all results with n_tool_calls and runtime data
        output_path: Path to save the plot
    """

    # Exclude compiler and manual methods
    methods_to_plot = [m for m in PLOT_METHODS if m not in ["aotpynatzeroshotsource", "manual"]]

    # Filter data to only include relevant methods
    plot_data = data[data["method"].isin(methods_to_plot)].copy()

    # Remove any rows with missing n_tool_calls or runtime
    plot_data = plot_data.dropna(subset=["n_tool_calls", "runtime"])

    if plot_data.empty:
        print("Warning: No valid numeric data for tool calls vs runtime plot")
        return

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Create legend handles for each model-method combination
    legend_handles = []
    added_combinations = set()

    # Plot each model-method combination
    for model in plot_data["model"].unique():
        marker = MODEL_MARKERS.get(model, "o")

        for method in methods_to_plot:
            method_data = plot_data[(plot_data["method"] == method) & (plot_data["model"] == model)]
            if method_data.empty:
                continue

            color = MODEL_COLORS.get(model, "#666666")

            # Plot all individual data points
            ax.scatter(
                method_data["n_tool_calls"],
                method_data["runtime"],
                c=color,
                marker=marker,
                s=50,
                alpha=0.6,
                linewidth=0.5,
                edgecolors="black",
            )

            # Add to legend handles (only once per combination)
            combination = f"{MODEL_MAPPING.get(model, "N/A")} - {METHOD_MAPPING.get(method, "N/A")}"
            if combination not in added_combinations:
                legend_handles.append(
                    plt.Line2D(  # pyright: ignore[reportPrivateImportUsage]
                        [0], [0], marker=marker, color=color, linestyle="None", markersize=8, label=combination
                    )
                )
                added_combinations.add(combination)

    # Customize the plot
    ax.set_xlabel("Number of Effects", fontsize=12)
    ax.set_ylabel("Runtime (s)", fontsize=12)
    # ax.set_title("Tool Calls vs Runtime", fontsize=14)

    # Set axis limits with some padding
    valid_x = plot_data["n_tool_calls"]
    valid_y = plot_data["runtime"]
    if not valid_x.empty and not valid_y.empty:
        x_min, x_max = valid_x.min(), valid_x.max()
        x_range = x_max - x_min
        ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)

        y_min, y_max = valid_y.min(), valid_y.max()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    # Apply common styling
    apply_subplot_styling(ax, alpha=0.3, fontsize=10)

    # Add legend
    fig.legend(handles=legend_handles, bbox_to_anchor=(0.5, -0.05), loc="upper center", fontsize=9, ncol=2)

    # Save the plot
    save_plot(output_path)


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

    # Calculate unaccounted time
    recorded_total = (
        times["compile_time"]
        + times["assistant_successful"]
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
        combined = pd.DataFrame(results.apply(parse_trace_times, axis=1))

        row: Dict[str, Any] = {
            "model": f"{provider}/{model_name}",
            "method": method,
        }
        for col in [
            "compile_time",
            "assistant_successful",
            "assistant_discarded",
            "tool_noncompute",
            "tool_compute",
            "unaccounted",
        ]:
            row[col] = combined.groupby(["file_name"])[col].mean().mean()

        # Calculate pass rate
        row["avg_pass_rate"], row["std_pass_rate"] = calculate_pass_rate(results=results)

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

    # Sort methods alphabetically for consistent ordering
    data = data.sort_values("method")

    methods = data["method"].tolist()

    # Set up the figure
    fig, ax = plt.subplots(figsize=(5, 4))

    # Define colors for each segment
    colors = {
        "assistant_successful": "#4971AC",  # Blue
        "assistant_discarded": "#DE6059",  # Red
        "tool_noncompute": "#9879a4",  # Purple
        "tool_compute": "#fbc75a",  # Yellow/Gold
        "compile_time": "#7CC775",  # Yellow/Gold
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

    # Add pass rate labels on top of bars
    avg_pass_rates = data["avg_pass_rate"].tolist()
    std_pass_rates = data["std_pass_rate"].tolist()
    for i, (pos, total_height, avg_pass_rate, std_pass_rate) in enumerate(
        zip(x, bottom, avg_pass_rates, std_pass_rates)
    ):
        ax.text(
            int(pos),
            total_height + 0.5,
            f"{avg_pass_rate:.1%}±{std_pass_rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

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
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


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

    if args.results_dir is None:
        if args.benchmark_suite == "interop":
            results_dirname = "benchmarks/results/final"
        elif args.benchmark_suite == "cpython":
            results_dirname = "benchmarks_cpython/results/final"
        elif args.benchmark_suite == "example":
            results_dirname = "benchmarks/results/final"
        else:
            raise ValueError(f"Invalid benchmark suite: {args.benchmark_suite}")

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

    # Load metadata for the benchmark suite
    try:
        metadata, results_column = load_metadata(args.benchmark_suite)
        print(f"Loaded metadata for {args.benchmark_suite} benchmark")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        return

    # Discover all models in the directory structure
    models = discover_models(args.results_dir)

    if not models:
        print(f"No models found in {args.results_dir}")
        return

    # Sort models to have gpt-4.1 first before sonnet 4
    def model_sort_key(model_tuple):
        provider, model_name, model_path = model_tuple
        full_name = f"{provider}/{model_name}"

        # Define priority order: gpt-4.1 first, then sonnet 4, then others
        if "gpt-4.1" in full_name:
            return (1, full_name)  # Highest priority
        elif "sonnet-4" in full_name:
            return (0, full_name)  # Second priority
        else:
            return (2, full_name)  # All others

    models = sorted(models, key=model_sort_key)

    print(f"Found {len(models)} models:")
    for provider, model_name, model_path in models:
        # Process results for each model
        display_name = get_model_display_name(provider, model_name)
        print(f"  - {display_name} ({provider}/{model_name})")

    # Group models by provider for better organization
    models_by_provider = {}
    for provider, model_name, model_path in models:
        if provider not in models_by_provider:
            models_by_provider[provider] = []
        models_by_provider[provider].append((model_name, model_path))

    all_data = []

    # Collect all data first, then create combined tables
    for provider in sorted(models_by_provider.keys()):
        for model_name, model_path in models_by_provider[provider]:
            for method_dir in os.listdir(model_path):
                # Read in results
                results_file = model_path / method_dir / f"results_{method_dir}.jsonl"
                if not results_file.exists():
                    continue

                with open(results_file, "r") as f:
                    results = pd.read_json(f, lines=True)

                # Normalize model identifier to match provider/model_name used elsewhere,
                # so nested paths like "vllm/openai/gpt-oss-20b" are grouped correctly.
                results["model"] = f"{provider}/{model_name}"

                # Debug: print columns to see what's available
                print(f"File Path: {results_file}")
                print(f"  -> Number of rows: {len(results)}")

                results = pd.merge(
                    left=results,
                    right=avg_score(results, metadata, results_column),
                    how="left",
                    left_on=["file_name", "run"],
                    right_on=["file_name", "run"],
                )
                results["method"] = method_dir

                all_data.append(results)

    all_data_df = pd.concat(all_data)
    all_data_df.drop(["hard_eval", "errors"], inplace=True, axis=1)
    all_data_df = compute_metrics(all_data_df)

    print(f"\n{'='*60}")
    print(f"SPSBench" + ("-- Example" if args.benchmark_suite == "example" else ""))
    print(f"{'='*60}")

    python_feat_set: set[str] = set(metadata.get("python_features", []))

    def append_summary_row(data: pd.DataFrame, provider: str, model_name: str, method: str, summary_list: List[Dict]):
        if data.empty:
            return

        incomplete = False
        for program in metadata["tests"].keys():
            runs = data[data["file_name"] == program]["run"]
            if len(runs) < 5:
                incomplete = True
                break
        inc = "*" if incomplete else ""

        avg_pass_rate_per_run = data.groupby(["run"])["pass_rate"].mean()
        avg_pass_rate = f"{avg_pass_rate_per_run.mean():.2f}" or "N/A"
        std_pass_rate = f"{avg_pass_rate_per_run.std():.2f}" or "N/A"

        runtime_per_program = data.groupby(["file_name"])["runtime"].mean()
        runtime_avg = f"{runtime_per_program.mean():.1f}" or "N/A"
        runtime_min = f"{data['runtime'].min():.1f}" or "N/A"
        runtime_max = f"{data['runtime'].max():.1f}" or "N/A"

        compiletime_per_program = data.groupby(["file_name"])["compile_time"].mean()
        compiletime_avg = f"{compiletime_per_program.mean():.1f}" or "N/A"
        compiletime_min = f"{data['compile_time'].min():.1f}" or "N/A"
        compiletime_max = f"{data['compile_time'].max():.1f}" or "N/A"

        totaltime_per_program = data.groupby(["file_name"])["total_time"].mean()
        totaltime_avg = f"{totaltime_per_program.mean():.1f}" or "N/A"
        totaltime_min = f"{data['total_time'].min():.1f}" or "N/A"
        totaltime_max = f"{data['total_time'].max():.1f}" or "N/A"

        tool_calls_per_program = data.groupby(["file_name"])["n_tool_calls"].mean()
        tool_calls_avg = f"{tool_calls_per_program.mean():.1f}" or "N/A"
        tool_calls_min = f"{data['n_tool_calls'].min():.1f}" or "N/A"
        tool_calls_max = f"{data['n_tool_calls'].max():.1f}" or "N/A"

        runtime_per_tool_per_program = data.groupby(["file_name"])["runtime_per_tool"].mean()
        runtime_per_tool_avg = f"{runtime_per_tool_per_program.mean():.1f}" or "N/A"
        runtime_per_tool_min = f"{data['runtime_per_tool'].min():.1f}" or "N/A"
        runtime_per_tool_max = f"{data['runtime_per_tool'].max():.1f}" or "N/A"

        input_tokens_per_program = data.groupby(["file_name"])["input_tokens"].mean()
        input_tokens_tool_avg = f"{input_tokens_per_program.mean():.1f}" or "N/A"
        input_tokens_tool_min = f"{data['input_tokens'].min():.1f}" or "N/A"
        input_tokens_tool_max = f"{data['input_tokens'].max():.1f}" or "N/A"

        uncached_input_tokens_per_program = data.groupby(["file_name"])["uncached_input_tokens"].mean()
        uncached_input_tokens_tool_avg = f"{uncached_input_tokens_per_program.mean():.1f}" or "N/A"
        uncached_input_tokens_tool_min = f"{data['uncached_input_tokens'].min():.1f}" or "N/A"
        uncached_input_tokens_tool_max = f"{data['uncached_input_tokens'].max():.1f}" or "N/A"

        output_tokens_per_program = data.groupby(["file_name"])["output_tokens"].mean()
        output_tokens_tool_avg = f"{output_tokens_per_program.mean():.1f}" or "N/A"
        output_tokens_tool_min = f"{data['output_tokens'].min():.1f}" or "N/A"
        output_tokens_tool_max = f"{data['output_tokens'].max():.1f}" or "N/A"

        row = {
            "Model": MODEL_MAPPING.get(f"{provider}/{model_name}", f"{provider}/{model_name}"),
            "Method": f"{inc}{METHOD_MAPPING.get(method, method)}",
            "Pass Rate": f"{avg_pass_rate}±{std_pass_rate}",
            "Execution Time": f"{runtime_avg} ({runtime_min}-{runtime_max})",
            "Compile Time": f"{compiletime_avg} ({compiletime_min}-{compiletime_max})",
            "Total Time": f"{totaltime_avg} ({totaltime_min}-{totaltime_max})",
            "Tool Calls": f"{tool_calls_avg} ({tool_calls_min}-{tool_calls_max})",
            # "Runtime / Tool Call": f"{runtime_per_tool_avg} ({runtime_per_tool_min}-{runtime_per_tool_max})",
            "Input Tokens": f"{input_tokens_tool_avg} ({input_tokens_tool_min}-{input_tokens_tool_max})",
            "Uncached Input Tokens": f"{uncached_input_tokens_tool_avg} ({uncached_input_tokens_tool_min}-{uncached_input_tokens_tool_max})",
            "Output Tokens": f"{output_tokens_tool_avg} ({output_tokens_tool_min}-{output_tokens_tool_max})",
        }

        summary_list.append(row)

    summary_data: List[Dict] = []
    summary_data_no_python_feat: List[Dict] = []
    summary_data_only_python_feat: List[Dict] = []

    for provider in sorted(models_by_provider.keys()):
        for model_name, model_path in models_by_provider[provider]:
            for method in TABLE_METHODS:
                data = all_data_df[
                    (all_data_df["model"] == f"{provider}/{model_name}") & (all_data_df["method"] == method)
                ].copy()
                if args.benchmark_suite == "example":
                    data = data[data["file_name"] == "graph"]

                data = cast(pd.DataFrame, data)

                print(f"Parsing results for {f'{provider}/{model_name}'} with {method}")
                for program in metadata["tests"].keys():
                    if len(data[data["file_name"] == program]) < 5:
                        missing_runs = 5 - len(data[data["file_name"] == program])
                        # print(f"  !! Missing {missing_runs} runs for {program}")

                if data.empty:
                    continue

                append_summary_row(
                    data=data,
                    provider=provider,
                    model_name=model_name,
                    method=method,
                    summary_list=summary_data,
                )

                if python_feat_set:
                    data_no_python_feat = data[~data["file_name"].isin(list(python_feat_set))].copy()
                    data_only_python_feat = data[data["file_name"].isin(list(python_feat_set))].copy()

                    append_summary_row(
                        data=cast(pd.DataFrame, data_no_python_feat),
                        provider=provider,
                        model_name=model_name,
                        method=method,
                        summary_list=summary_data_no_python_feat,
                    )

                    append_summary_row(
                        data=cast(pd.DataFrame, data_only_python_feat),
                        provider=provider,
                        model_name=model_name,
                        method=method,
                        summary_list=summary_data_only_python_feat,
                    )

            summary_data.append({})
            summary_data_no_python_feat.append({})
            summary_data_only_python_feat.append({})

    print(
        tabulate(
            summary_data,
            headers="keys",
            tablefmt="latex" if args.latex else "github",
            showindex=False,
            floatfmt=".1f",
        )
    )

    if summary_data_no_python_feat:
        print("\nNo Python-Specific Features\n")
        print(
            tabulate(
                summary_data_no_python_feat,
                headers="keys",
                tablefmt="latex" if args.latex else "github",
                showindex=False,
                floatfmt=".1f",
            )
        )

    # if summary_data_only_python_feat:
    #     print("\nOnly Python-Specific Features\n")
    #     print(
    #         tabulate(
    #             summary_data_only_python_feat,
    #             headers="keys",
    #             tablefmt="latex" if args.latex else "github",
    #             showindex=False,
    #             floatfmt=".1f",
    #         )
    #     )

    # Parse and aggregate trace data
    # data = aggregate_trace_data(trace_files=trace_files)

    # if data.empty:
    #     print("No data to process")
    #     return

    # # Print summary statistics
    # print_summary_statistics(data=data)

    # Generate plots if output_plot is specified
    if args.output_plot is not None and args.benchmark_suite != "example":
        print(f"\n{'='*60}")
        print(f"Generating plots to {args.output_plot}")
        print(f"{'='*60}")

        os.makedirs(args.output_plot, exist_ok=True)

        # Plot time vs pass rate
        time_plot_path = args.output_plot / f"time_vs_pass_rate_{args.benchmark_suite}.pdf"
        print(f"Creating time vs pass rate plot: {time_plot_path}")
        plot_time_vs_pass_rate(all_data_df, time_plot_path)

        # Time breakdown
        # print(f"Creating time breakdown plot: {time_plot_path}")
        # time_plot_path = args.output_plot / "time_breakdown_{args.benchmark_suite}.pdf"
        # plot_stacked_bar_chart(data=data, output_path=args.output_plot)

        # Plot tool calls vs runtime
        tool_calls_plot_path = args.output_plot / f"tool_calls_vs_runtime_{args.benchmark_suite}.pdf"
        print(f"Creating tool calls vs runtime plot: {tool_calls_plot_path}")
        plot_tool_calls_vs_runtime(all_data_df, tool_calls_plot_path)

        # Plot benchmark line plots for each model
        for provider in sorted(models_by_provider.keys()):
            for model_name, model_path in models_by_provider[provider]:
                model_display_name = get_model_display_name(provider, model_name)

                data = all_data_df[(all_data_df["model"] == f"{provider}/{model_name}")].copy()
                if args.benchmark_suite == "example":
                    data = data[data["file_name"] == "graph"]

                data = cast(pd.DataFrame, data)

                if not data.empty:
                    # line_plot_path = (
                    #     args.output_plot / f"benchmark_line_{args.benchmark_suite}_{provider}_{model_name}.pdf"
                    # )
                    # print(f"Creating benchmark line plot for {model_display_name}: {line_plot_path}")
                    # plot_benchmark_line_plots(
                    #     raw_data=model_raw_data,
                    #     method_order=TABLE_METHODS,
                    #     model=f"{provider}/{model_name}",
                    #     output_path=line_plot_path,
                    #     metadata=metadata,
                    #     results_column=results_column,
                    #     benchmark_suite=args.benchmark_suite,
                    # )

                    bar_plot_path = (
                        args.output_plot / f"benchmark_bar_{args.benchmark_suite}_{provider}_{model_name}.pdf"
                    )
                    print(f"Creating benchmark bar plot for {model_display_name}: {bar_plot_path}")
                    plot_benchmark_bar_graph(
                        raw_data=data,
                        method_order=PLOT_METHODS,
                        model=f"{provider}/{model_name}",
                        output_path=bar_plot_path,
                        metadata=metadata,
                        benchmark_suite=args.benchmark_suite,
                    )

        print(f"\nPlots saved to {args.output_plot}")


if __name__ == "__main__":
    main()

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tabulate import tabulate
from tap import Tap

TABLE_METHODS = [
    "interpreter_python_json",
]

MODEL_MAPPING = {
    "openai/gpt-4.1-2025-04-14": "GPT-4.1",
    "anthropic/claude-sonnet-4-20250514": "Sonnet 4",
}

METHOD_MAPPING = {
    "interpreter_python_json": "Nightjar (Ours)",
}


class ArgumentParser(Tap):
    results_dir: Path  # Path to the results directory (e.g., benchmarks/results/final/)
    output_file: Path  # Optional output file for the summary (defaults to stdout)

    def configure(self):
        self.add_argument("results_dir", type=Path, help="Path to results directory")
        self.add_argument(
            "--output_file",
            type=Path,
            help="Optional output file for the summary",
            default=None,
        )


def load_analysis_data(analysis_file: Path) -> List[Dict]:
    """Load analysis data from JSONL file."""
    data = []
    with open(analysis_file, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                data.append(entry)
            except json.JSONDecodeError:
                continue
    return data


def generate_summary(data: List[Dict], total_runs_from_results: int = None) -> Dict[str, pd.DataFrame]:
    """Generate summary statistics from analysis data."""

    # Count failure reasons (run-level)
    failure_reasons = defaultdict(int)

    # Count error categories and recovery methods (error-level)
    error_categories = defaultdict(int)
    recovery_methods = defaultdict(int)

    total_labeled_runs = len(data)
    total_errors = 0

    # Expected total runs
    EXPECTED_RUNS = 125

    # Use total_runs_from_results if provided, otherwise use labeled runs
    total_runs = total_runs_from_results if total_runs_from_results is not None else total_labeled_runs

    for entry in data:
        # Count failure reason
        failure_reason = entry.get("failure_reason")
        if failure_reason is None:
            failure_reasons["null (100% pass)"] += 1
        else:
            failure_reasons[failure_reason] += 1

        # Count error-level labels
        for error in entry.get("errors", []):
            total_errors += 1

            category = error.get("category")
            if category is None:
                error_categories["null"] += 1
            else:
                error_categories[category] += 1

            recovery = error.get("recovery_method")
            if recovery is None:
                recovery_methods["null"] += 1
            else:
                recovery_methods[recovery] += 1

    # Add "missing" if we have fewer than expected runs
    if total_runs < EXPECTED_RUNS:
        missing_count = EXPECTED_RUNS - total_runs
        failure_reasons["missing"] = missing_count

    # Create DataFrames

    # Failure Reasons (use EXPECTED_RUNS as denominator for percentage)
    failure_df = pd.DataFrame(
        [
            {"Failure Reason": reason, "Count": count, "Percentage": f"{count / EXPECTED_RUNS * 100:.1f}%"}
            for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True)
        ]
    )

    # Error Categories
    if total_errors > 0:
        categories_df = pd.DataFrame(
            [
                {"Error Category": category, "Count": count, "Percentage": f"{count / total_errors * 100:.1f}%"}
                for category, count in sorted(error_categories.items(), key=lambda x: x[1], reverse=True)
            ]
        )
    else:
        categories_df = pd.DataFrame(columns=["Error Category", "Count", "Percentage"])

    # Recovery Methods
    if total_errors > 0:
        recovery_df = pd.DataFrame(
            [
                {"Recovery Method": method, "Count": count, "Percentage": f"{count / total_errors * 100:.1f}%"}
                for method, count in sorted(recovery_methods.items(), key=lambda x: x[1], reverse=True)
            ]
        )
    else:
        recovery_df = pd.DataFrame(columns=["Recovery Method", "Count", "Percentage"])

    return {
        "failure_reasons": failure_df,
        "error_categories": categories_df,
        "recovery_methods": recovery_df,
        "stats": {
            "total_runs": total_runs,
            "total_errors": total_errors,
        },
    }


def find_analysis_files(results_dir: Path) -> List[Dict[str, str]]:
    """Find all analysis files for the specified methods."""
    analysis_files = []

    # Scan through provider/model/method structure
    for provider_dir in results_dir.iterdir():
        if not provider_dir.is_dir():
            continue

        for model_dir in provider_dir.iterdir():
            if not model_dir.is_dir():
                continue

            for method in TABLE_METHODS:
                method_dir = model_dir / method
                if not method_dir.exists():
                    continue

                # Look for analysis_traces_{method}.jsonl and results_{method}.jsonl
                analysis_file = method_dir / f"analysis_traces_{method}.jsonl"
                results_file = method_dir / f"results_{method}.jsonl"

                if analysis_file.exists():
                    analysis_files.append(
                        {
                            "provider": provider_dir.name,
                            "model": model_dir.name,
                            "method": method,
                            "analysis_path": analysis_file,
                            "results_path": results_file if results_file.exists() else None,
                        }
                    )

    return analysis_files


def create_summary_statistics_table(all_summaries: List[Dict]) -> pd.DataFrame:
    """Create a summary table with total runs, total errors, and errors per run for each configuration."""

    rows = []
    for item in all_summaries:
        provider = item["provider"]
        model = item["model"]
        method = item["method"]
        summary = item["summary"]

        # Get display names from mappings
        model_key = f"{provider}/{model}"
        model_name = MODEL_MAPPING.get(model_key, model_key)
        method_name = METHOD_MAPPING.get(method, method)

        total_runs = summary["stats"]["total_runs"]
        total_errors = summary["stats"]["total_errors"]
        errors_per_run = total_errors / total_runs if total_runs > 0 else 0

        rows.append(
            {
                "Model": model_name,
                "Method": method_name,
                "Total Runs": total_runs,
                "Total Errors": total_errors,
                "Errors per Run": f"{errors_per_run:.2f}",
            }
        )

    df = pd.DataFrame(rows)
    return df


def create_consolidated_tables(all_summaries: List[Dict]) -> Dict[str, pd.DataFrame]:
    """Create three consolidated tables (failure reasons, error categories, error recovery) for all model-method combinations."""

    # Prepare data for all model-method combinations
    combinations_data = []
    for item in all_summaries:
        provider = item["provider"]
        model = item["model"]
        method = item["method"]
        summary = item["summary"]

        # Get display names from mappings
        model_key = f"{provider}/{model}"
        model_name = MODEL_MAPPING.get(model_key, model_key)
        method_name = METHOD_MAPPING.get(method, method)

        # Create combined column name
        column_name = f"{model_name} - {method_name}"

        combinations_data.append(
            {
                "column_name": column_name,
                "summary": summary,
            }
        )

    # Collect all unique categories across all combinations
    all_failure_reasons = set()
    all_error_categories = set()
    all_recovery_methods = set()

    for combo_data in combinations_data:
        summary = combo_data["summary"]
        all_failure_reasons.update(summary["failure_reasons"]["Failure Reason"].tolist())
        if not summary["error_categories"].empty:
            all_error_categories.update(summary["error_categories"]["Error Category"].tolist())
        if not summary["recovery_methods"].empty:
            all_recovery_methods.update(summary["recovery_methods"]["Recovery Method"].tolist())

    # Create column names (all model-method combinations)
    column_names = [combo_data["column_name"] for combo_data in combinations_data]

    # Build failure reasons table
    failure_data = {}
    for reason in sorted(all_failure_reasons):
        failure_data[reason] = []
        for combo_data in combinations_data:
            summary = combo_data["summary"]
            df = summary["failure_reasons"]
            matching = df[df["Failure Reason"] == reason]
            if len(matching) > 0:
                count = matching.iloc[0]["Count"]
                percentage = matching.iloc[0]["Percentage"]
                failure_data[reason].append(f"{count} ({percentage})")
            else:
                failure_data[reason].append("0 (0.0%)")

    failure_df = pd.DataFrame(failure_data, index=column_names).T
    failure_df.index.name = "Failure Reason"

    # Build error categories table
    error_cat_data = {}
    for category in sorted(all_error_categories):
        error_cat_data[category] = []
        for combo_data in combinations_data:
            summary = combo_data["summary"]
            df = summary["error_categories"]
            if df.empty:
                error_cat_data[category].append("0 (0.0%)")
            else:
                matching = df[df["Error Category"] == category]
                if len(matching) > 0:
                    count = matching.iloc[0]["Count"]
                    percentage = matching.iloc[0]["Percentage"]
                    error_cat_data[category].append(f"{count} ({percentage})")
                else:
                    error_cat_data[category].append("0 (0.0%)")

    if error_cat_data:
        error_cat_df = pd.DataFrame(error_cat_data, index=column_names).T
        error_cat_df.index.name = "Error Category"
    else:
        error_cat_df = pd.DataFrame()

    # Build recovery methods table
    recovery_data = {}
    for method in sorted(all_recovery_methods):
        recovery_data[method] = []
        for combo_data in combinations_data:
            summary = combo_data["summary"]
            df = summary["recovery_methods"]
            if df.empty:
                recovery_data[method].append("0 (0.0%)")
            else:
                matching = df[df["Recovery Method"] == method]
                if len(matching) > 0:
                    count = matching.iloc[0]["Count"]
                    percentage = matching.iloc[0]["Percentage"]
                    recovery_data[method].append(f"{count} ({percentage})")
                else:
                    recovery_data[method].append("0 (0.0%)")

    if recovery_data:
        recovery_df = pd.DataFrame(recovery_data, index=column_names).T
        recovery_df.index.name = "Recovery Method"
    else:
        recovery_df = pd.DataFrame()

    return {
        "failure_reasons": failure_df,
        "error_categories": error_cat_df,
        "recovery_methods": recovery_df,
    }


def main():
    args = ArgumentParser().parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return

    print(f"Scanning for analysis files in: {args.results_dir}")
    analysis_files = find_analysis_files(results_dir=args.results_dir)

    if not analysis_files:
        print("No analysis files found.")
        return

    print(f"Found {len(analysis_files)} analysis files\n")

    # Collect all summaries
    all_summaries = []
    for file_info in sorted(analysis_files, key=lambda x: (x["provider"], x["model"], x["method"])):
        provider = file_info["provider"]
        model = file_info["model"]
        method = file_info["method"]
        analysis_file = file_info["analysis_path"]
        results_file = file_info["results_path"]

        print(f"Processing: {provider}/{model}/{method}")

        # Load analysis data
        data = load_analysis_data(analysis_file=analysis_file)

        if not data:
            print(f"  Warning: No data found in {analysis_file}")
            continue

        # Load results data to get total runs count
        total_runs_from_results = None
        if results_file and results_file.exists():
            results_data = load_analysis_data(analysis_file=results_file)
            total_runs_from_results = len(results_data)
            print(f"  Total runs from results: {total_runs_from_results}")
        else:
            print(f"  Warning: Results file not found, using labeled runs count")

        # Generate summary
        summary = generate_summary(data=data, total_runs_from_results=total_runs_from_results)
        all_summaries.append(
            {
                "provider": provider,
                "model": model,
                "method": method,
                "summary": summary,
            }
        )

    if not all_summaries:
        print("No valid summaries generated.")
        return

    print("\nGenerating consolidated tables...\n")

    # Create summary statistics table
    summary_stats_table = create_summary_statistics_table(all_summaries=all_summaries)

    # Create consolidated tables for all model-method combinations
    consolidated_tables = create_consolidated_tables(all_summaries=all_summaries)

    # Prepare output
    output_lines = []
    output_lines.append("=" * 120)
    output_lines.append("LABEL SUMMARY STATISTICS - ALL MODEL-METHODS")
    output_lines.append("=" * 120)
    output_lines.append(f"Results Directory: {args.results_dir}")
    output_lines.append(f"Methods: {', '.join(TABLE_METHODS)}")
    output_lines.append(f"Total Configurations: {len(all_summaries)}")
    output_lines.append("")

    # Summary Statistics Table
    output_lines.append("\n" + "=" * 120)
    output_lines.append("SUMMARY: TOTAL RUNS, ERRORS, AND ERRORS PER RUN")
    output_lines.append("=" * 120)
    output_lines.append(tabulate(summary_stats_table, headers="keys", tablefmt="github", showindex=False))
    output_lines.append("")

    # Failure Reasons Table (all model-methods)
    output_lines.append("\n" + "=" * 120)
    output_lines.append("RUN-LEVEL FAILURE REASONS (ALL MODEL-METHODS)")
    output_lines.append("=" * 120)
    output_lines.append(tabulate(consolidated_tables["failure_reasons"], headers="keys", tablefmt="github"))
    output_lines.append("")

    # Error Categories Table (all model-methods)
    output_lines.append("\n" + "=" * 120)
    output_lines.append("ERROR-LEVEL CATEGORIES (ALL MODEL-METHODS)")
    output_lines.append("=" * 120)
    if not consolidated_tables["error_categories"].empty:
        output_lines.append(tabulate(consolidated_tables["error_categories"], headers="keys", tablefmt="github"))
    else:
        output_lines.append("No errors found.")
    output_lines.append("")

    # Recovery Methods Table (all model-methods)
    output_lines.append("\n" + "=" * 120)
    output_lines.append("ERROR-LEVEL RECOVERY METHODS (ALL MODEL-METHODS)")
    output_lines.append("=" * 120)
    if not consolidated_tables["recovery_methods"].empty:
        output_lines.append(tabulate(consolidated_tables["recovery_methods"], headers="keys", tablefmt="github"))
    else:
        output_lines.append("No errors found.")
    output_lines.append("")

    output_text = "\n".join(output_lines)

    # Output
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output_text)
        print(f"Summary saved to: {args.output_file}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()

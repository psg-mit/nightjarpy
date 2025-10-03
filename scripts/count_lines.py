#!/usr/bin/env python3
"""
Script to count non-empty lines before "#### Tests ####" marker in .nj and _manual.py files.
"""

import glob
import os
from typing import Dict, List, Tuple

from tabulate import tabulate


def count_lines_before_tests_marker(file_path: str) -> int:
    """
    Count non-empty lines before the "#### Tests ####" marker in a file.

    Args:
        file_path: Path to the file to analyze

    Returns:
        Number of non-empty lines before the marker, or -1 if marker not found
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        count = 0
        for line in lines:
            line = line.strip()
            if line == "#### Tests ####":
                return count
            elif line and not line.startswith("#"):  # Non-empty line and not comments
                count += 1

        # Marker not found
        return -1

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return -1


def get_file_pairs(programs_dir: str) -> List[Tuple[str, str, str]]:
    """
    Get pairs of .nj and _manual.py files for each program.

    Args:
        programs_dir: Path to the programs directory

    Returns:
        List of tuples (program_name, nj_file_path, manual_file_path)
    """
    pairs = []

    # Get all subdirectories in programs
    for subdir in os.listdir(programs_dir):
        subdir_path = os.path.join(programs_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        benchmark_name = os.path.basename(subdir_path)
        # Look for nightjar benchmark file
        nj_files = glob.glob(os.path.join(subdir_path, f"{benchmark_name}.py"))
        manual_files = glob.glob(os.path.join(subdir_path, f"{benchmark_name}_manual.py"))

        if nj_files and manual_files:
            nj_file = nj_files[0]
            manual_file = manual_files[0]
            pairs.append((subdir, nj_file, manual_file))

    return sorted(pairs)


def main():
    """Main function to count lines and display results."""
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    programs_dir = os.path.join(script_dir, "benchmarks", "programs")

    if not os.path.exists(programs_dir):
        print(f"Error: Programs directory not found at {programs_dir}")
        return

    # Get file pairs
    file_pairs = get_file_pairs(programs_dir)

    if not file_pairs:
        print("No matching file pairs found.")
        return

    # Count lines for each file pair
    results = []

    for program_name, nj_file, manual_file in file_pairs:
        nj_lines = count_lines_before_tests_marker(nj_file)
        manual_lines = count_lines_before_tests_marker(manual_file)

        # Calculate difference (manual - nightjar)
        if nj_lines >= 0 and manual_lines >= 0:
            difference = manual_lines - nj_lines
        else:
            difference = "N/A"

        # Calculate percentage decrease ((manual - nightjar) / manual * 100)
        if nj_lines >= 0 and manual_lines >= 0 and manual_lines > 0:
            percent_decrease = ((manual_lines - nj_lines) / manual_lines) * 100
        else:
            percent_decrease = "N/A"

        results.append(
            {
                "program": program_name,
                "nightjar": nj_lines if nj_lines >= 0 else "Marker not found",
                "manual": manual_lines if manual_lines >= 0 else "Marker not found",
                "difference": difference,
                "percent_decrease": percent_decrease,
            }
        )

    # Display results in a table
    headers = ["Program", "Nightjar ", "Manual", "Difference", "% Decrease"]
    table_data = [[r["program"], r["nightjar"], r["manual"], r["difference"], r["percent_decrease"]] for r in results]

    print("Non-empty lines before '#### Tests ####' marker:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Summary statistics
    valid_nj_counts = [r["nightjar"] for r in results if isinstance(r["nightjar"], int)]
    valid_manual_counts = [r["manual"] for r in results if isinstance(r["manual"], int)]
    valid_differences = [r["difference"] for r in results if isinstance(r["difference"], int)]
    valid_percent_decreases = [
        r["percent_decrease"] for r in results if isinstance(r["percent_decrease"], (int, float))
    ]

    if valid_nj_counts:
        print(f"\nSummary for Nightjar files:")
        print(f"  Total programs: {len(valid_nj_counts)}")
        print(f"  Average lines: {sum(valid_nj_counts) / len(valid_nj_counts):.1f}")
        print(f"  Min lines: {min(valid_nj_counts)}")
        print(f"  Max lines: {max(valid_nj_counts)}")

    if valid_manual_counts:
        print(f"\nSummary for Manual files:")
        print(f"  Total programs: {len(valid_manual_counts)}")
        print(f"  Average lines: {sum(valid_manual_counts) / len(valid_manual_counts):.1f}")
        print(f"  Min lines: {min(valid_manual_counts)}")
        print(f"  Max lines: {max(valid_manual_counts)}")

    if valid_differences:
        print(f"\nSummary for Differences (Manual - Nightjar):")
        print(f"  Total programs: {len(valid_differences)}")
        print(f"  Average difference: {sum(valid_differences) / len(valid_differences):.1f}")
        print(f"  Min difference: {min(valid_differences)}")
        print(f"  Max difference: {max(valid_differences)}")
        positive_diff = [d for d in valid_differences if d > 0]
        negative_diff = [d for d in valid_differences if d < 0]
        print(f"  Programs where manual > nightjar: {len(positive_diff)}")
        print(f"  Programs where manual < nightjar: {len(negative_diff)}")

    if valid_percent_decreases:
        print(f"\nSummary for Percentage Decrease:")
        print(f"  Total programs: {len(valid_percent_decreases)}")
        print(f"  Average % decrease: {sum(valid_percent_decreases) / len(valid_percent_decreases):.1f}%")
        print(f"  Min % decrease: {min(valid_percent_decreases):.1f}%")
        print(f"  Max % decrease: {max(valid_percent_decreases):.1f}%")
        positive_decrease = [d for d in valid_percent_decreases if d > 0]
        negative_decrease = [d for d in valid_percent_decreases if d < 0]
        print(f"  Programs with positive % decrease (nightjar shorter): {len(positive_decrease)}")
        print(f"  Programs with negative % decrease (nightjar longer): {len(negative_decrease)}")


if __name__ == "__main__":
    main()

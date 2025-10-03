#!/usr/bin/env python3
"""
Streamlit UI for labeling run-level failure reasons and tool errors in traces.
"""

import json
from collections import defaultdict
from inspect import Parameter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd
import streamlit as st
from streamlit.column_config import CheckboxColumn, TextColumn

from nightjar.nlfi.interface import Func, Immutable, JsonType, Obj, Ref
from nightjar.utils import is_subtype


def load_results_structure(results_file: Path) -> Tuple[Dict[str, int], Dict]:
    """Load results file to determine the number of runs per program and get eval results."""
    program_runs = defaultdict(set)
    results_data = {}  # Maps (program, run) -> full result entry

    with open(results_file, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                program_name = data.get("file_name", "unknown")
                run_num = data.get("run", 0)
                program_runs[program_name].add(run_num)
                results_data[(program_name, run_num)] = data
            except json.JSONDecodeError:
                continue

    return {prog: len(runs) for prog, runs in program_runs.items()}, results_data


def parse_traces(traces_file: Path, results_file: Path) -> pd.DataFrame:
    """Parse traces and extract tool errors, grouping by run."""
    # Load results structure if provided
    results_data = pd.read_json(results_file, lines=True)

    traces_data = pd.read_json(traces_file, lines=True).set_index(["file_name", "run", "model"])
    data = traces_data.join(results_data.set_index(["file_name", "run", "model"])).reset_index()

    runs = []

    for row_i, row in data.iterrows():
        # Extract errors from this trace
        trace = cast(List[Dict], row["trace"])
        errors_in_run = []

        for i, turn in enumerate(trace):
            if turn.get("role") == "tool":
                content = turn.get("content", "")
                tool_call_id = turn.get("tool_call_id", "")

                if isinstance(content, str) and content.startswith("Error:"):
                    # Find the corresponding tool call
                    tool_call_info = None
                    for j in range(i - 1, -1, -1):
                        prev_turn = trace[j]
                        if prev_turn.get("role") == "assistant" and "tool_calls" in prev_turn:
                            for tc in prev_turn["tool_calls"]:
                                if tc.get("id") == tool_call_id:
                                    tool_call_info = tc
                                    break
                        if tool_call_info:
                            break

                    error_entry = {
                        "error_tool_call_id": tool_call_id,
                        "error_tool_call": tool_call_info,
                        "error_content": content,
                        "error_turn_index": i,
                    }
                    errors_in_run.append(error_entry)

        # Calculate pass rate
        hard_eval = row.get("hard_eval", {})
        if hard_eval is None:
            hard_eval = {}
        total_tests = len(hard_eval)
        passed_tests = sum(1 for v in hard_eval.values() if v is True)
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        run_entry = {
            "file_name": row["file_name"],
            "run": row["run"],
            "model": row["model"],
            "trace": trace,
            "hard_eval": hard_eval,
            "pass_rate": pass_rate,
            "errors": errors_in_run,
        }
        runs.append(run_entry)

    return pd.DataFrame(runs)


def deserialize(
    name: str, json: JsonType
) -> Immutable | List[Tuple[Optional[str], Ref]] | List[Parameter] | List[Ref[Obj]] | str:
    if json is None:
        return None
    elif is_subtype(type(json), Immutable):
        return json  # type: ignore
    elif name == "arguments" or name == "kwds":
        return Func.arguments_from_json(json)
    elif name == "parameters":
        return Func.parameters_from_json(json)
    elif name == "base_classes":
        if not isinstance(json, list):
            raise ValueError(f"Expected list of base classes, but got type {type(json)}")
        return [Ref.from_json(base_class) for base_class in json]
    else:
        return Ref.from_json(json)


def format_trace_for_display(trace: List[Dict], errors: List[Dict]) -> pd.DataFrame:
    """Format a trace with error context into a DataFrame for display."""
    # Get all error turn indices and map them to error numbers
    error_turn_index_to_number = {}
    for idx, error in enumerate(errors):
        error_turn_index_to_number[error["error_turn_index"]] = idx + 1

    rows = []
    start_idx = 0
    end_idx = len(trace)

    for i in range(start_idx, end_idx):
        turn = trace[i]
        role = turn.get("role", "unknown")

        # Determine error number for this turn
        error_num = error_turn_index_to_number.get(i, "")

        if role == "user":
            rows.append(
                {
                    # "Turn": i,
                    "Role": "User",
                    "Content": turn.get("content", ""),
                    "Tool": "",
                    "Tool Args": "",
                    "Result": "",
                    "Error #": error_num,
                }
            )
        elif role == "assistant":
            tool_calls = turn.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    if func.get("name", "") == "eval":
                        args = json.loads(func.get("arguments", "{}")).get("expression", "")
                    elif func.get("name", "") == "exec":
                        args = json.loads(func.get("arguments", "{}")).get("code", "")
                    else:
                        args = func.get("arguments", "")
                    rows.append(
                        {
                            # "Turn": i,
                            "Role": "Assistant",
                            "Content": turn.get("content", ""),
                            "Tool": func.get("name", ""),
                            "Tool Args": args,
                            "Result": "",
                            "Error #": "",
                        }
                    )
            else:
                rows.append(
                    {
                        # "Turn": i,
                        "Role": "Assistant",
                        "Content": turn.get("content", ""),
                        "Tool": "",
                        "Tool Args": "",
                        "Result": "",
                        "Error #": "",
                    }
                )
        elif role == "tool":
            content = turn.get("content", "")
            rows.append(
                {
                    # "Turn": i,
                    "Role": "Tool",
                    "Content": "",
                    "Tool": "",
                    "Tool Args": "",
                    "Result": deserialize(name="res", json=content),
                    "Error #": error_num,
                }
            )

    return pd.DataFrame(rows)


def load_existing_labels(output_file: Path) -> Dict:
    """Load existing labels from file."""
    labeled_runs = {}
    if output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    key = (entry["file_name"], entry["run"])
                    labeled_runs[key] = entry
                except json.JSONDecodeError:
                    continue
    return labeled_runs


def save_label(label_entry: Dict, output_file: Path):
    """Append a label to the output file."""
    print(label_entry)
    with open(output_file, "a") as f:
        f.write(json.dumps(label_entry) + "\n")


def is_run_fully_labeled(run: pd.Series, labeled_runs: Dict) -> bool:
    """Check if a run is fully labeled (failure reason and all error labels)."""
    run_key = (run["file_name"], run["run"])

    # Check if run exists in labeled_runs
    if run_key not in labeled_runs:
        return False

    existing_label = labeled_runs[run_key]

    # For runs with pass_rate < 100%, failure_reason must be labeled
    if run["pass_rate"] < 100.0:
        if existing_label.get("failure_reason") is None:
            return False

    # Check if all errors have category and recovery_method labeled
    if run["errors"]:
        existing_errors = existing_label.get("errors", [])

        # Build a map of error_tool_call_id to existing error label
        error_label_map = {e.get("error_tool_call_id"): e for e in existing_errors}

        # Check each error in the run
        for error in run["errors"]:
            error_id = error["error_tool_call_id"]

            # If error not found in labels, it's not fully labeled
            if error_id not in error_label_map:
                return False

            error_label = error_label_map[error_id]

            # Check if category and recovery_method are labeled
            if error_label.get("category") is None or error_label.get("recovery_method") is None:
                return False

    return True


def main():

    st.set_page_config(page_title="Run Analysis Labeling", page_icon="🏷️", layout="wide")

    st.title("🏷️ Run Analysis Labeling Interface")
    st.markdown("Label run-level failure reasons and individual tool errors")

    # Sidebar for file selection
    st.sidebar.header("File Selection")

    # Discover subdirectories in results/final/
    script_dir = Path(__file__).parent
    results_base_dir = script_dir / "results" / "final"

    if not results_base_dir.exists():
        st.sidebar.error(f"❌ Directory not found: {results_base_dir}")
        st.info("Please ensure the 'results/final/' directory exists.")
        return

    # Find all subdirectories that contain both results_*.jsonl and traces_*.jsonl files
    valid_dirs = []
    for subdir in results_base_dir.rglob("*"):
        if subdir.is_dir():
            results_files = list(subdir.glob("results_*.jsonl"))
            traces_files = list(subdir.glob("traces_*.jsonl"))
            if results_files and traces_files:
                # Store relative path from results/final/
                rel_path = subdir.relative_to(results_base_dir)
                valid_dirs.append((str(rel_path), subdir))

    if not valid_dirs:
        st.sidebar.error("❌ No valid directories found with results_*.jsonl and traces_*.jsonl files")
        st.info(f"Please ensure subdirectories under '{results_base_dir}' contain both files.")
        return

    # Sort by path for consistent ordering
    valid_dirs.sort(key=lambda x: x[0])

    # Create dropdown for directory selection
    dir_options = [d[0] for d in valid_dirs]
    selected_dir_name = st.sidebar.selectbox(
        "Select results directory:", dir_options, help="Choose a directory containing results and traces files"
    )

    # Get the full path for the selected directory
    dir_path = next(d[1] for d in valid_dirs if d[0] == selected_dir_name)

    # Look for results and traces files in the directory
    results_files = list(dir_path.glob("results_*.jsonl"))
    traces_files = list(dir_path.glob("traces_*.jsonl"))

    # If multiple files found, let user select
    if len(results_files) > 1:
        results_file_names = [f.name for f in results_files]
        selected_results = st.sidebar.selectbox("Select results file:", results_file_names)
        results_path = dir_path / selected_results
    else:
        results_path = results_files[0]

    if len(traces_files) > 1:
        traces_file_names = [f.name for f in traces_files]
        selected_traces = st.sidebar.selectbox("Select traces file:", traces_file_names)
        traces_path = dir_path / selected_traces
    else:
        traces_path = traces_files[0]

    # Set output path in the same directory
    output_path = dir_path / f"analysis_{traces_path.stem}.jsonl"

    st.sidebar.success(f"✓ Directory: {selected_dir_name}")
    st.sidebar.success(f"✓ Results: {results_path.name}")
    st.sidebar.success(f"✓ Traces: {traces_path.name}")
    st.sidebar.info(f"📝 Output: {output_path.name}")

    # Load data - automatically reload when directory changes
    current_dir_key = f"{selected_dir_name}_{results_path.name}_{traces_path.name}"

    # Check if we need to reload (first time or directory changed)
    need_reload = (
        "runs" not in st.session_state
        or "current_dir_key" not in st.session_state
        or st.session_state.current_dir_key != current_dir_key
        or st.sidebar.button("Reload Data")
    )

    if need_reload:
        with st.spinner("Loading traces and parsing errors..."):
            runs = parse_traces(traces_file=traces_path, results_file=results_path)
            st.session_state.runs = runs
            st.session_state.output_path = output_path
            st.session_state.labeled_runs = load_existing_labels(output_path)
            st.session_state.current_dir_key = current_dir_key

            # Find first unlabeled run
            first_unlabeled_idx = 0
            for idx, (_, run) in enumerate(runs.iterrows()):
                if not is_run_fully_labeled(run=run, labeled_runs=st.session_state.labeled_runs):
                    first_unlabeled_idx = idx
                    break

            # Navigate to first unlabeled run
            st.query_params["run_idx"] = str(first_unlabeled_idx)

    runs = st.session_state.runs
    output_path = st.session_state.output_path
    labeled_runs = st.session_state.labeled_runs

    # Display statistics
    st.header("Run Statistics")

    total_runs = len(runs)
    total_errors = sum(len(run["errors"]) for _, run in runs.iterrows())
    avg_errors_per_run = total_errors / total_runs if total_runs > 0 else 0
    avg_pass_rate = runs["pass_rate"].mean()

    # Count fully labeled runs
    fully_labeled_count = sum(
        1 for _, run in runs.iterrows() if is_run_fully_labeled(run=run, labeled_runs=labeled_runs)
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Runs", total_runs)
    with col2:
        st.metric("Total Errors", total_errors)
    with col3:
        st.metric("Avg Errors/Run", f"{avg_errors_per_run:.2f}")
    with col4:
        st.metric("Labeled Runs", f"{fully_labeled_count} / {total_runs}")

    # Show program breakdown
    st.subheader("📊 Stats by Program")
    program_stats = []
    for file_name in sorted(runs["file_name"].unique()):
        program_runs = runs[runs["file_name"] == file_name]
        avg_errors = sum(len(run["errors"]) for _, run in program_runs.iterrows()) / len(program_runs)
        avg_pass = program_runs["pass_rate"].mean()
        program_stats.append(
            {
                "Program": file_name,
                "Runs": len(program_runs),
                "Avg Errors/Run": f"{avg_errors:.2f}",
                "Avg Pass Rate": f"{avg_pass:.1f}%",
            }
        )

    st.dataframe(
        pd.DataFrame(program_stats),
        width="stretch",
        hide_index=True,
        height=300,
    )

    # Labeling interface
    st.header("Label Runs")

    if total_runs == 0:
        st.info("No runs found in traces.")
        return

    program_list = sorted(list(runs["file_name"].unique()))
    # Filter options
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        show_filter = st.selectbox("Show", ["All Runs", "Unlabeled Only", "Labeled Only"])
    with col2:
        program_filter = st.selectbox("Filter by Program", ["All Programs"] + program_list)

    # Filter runs
    filtered_runs = runs.copy()

    # Filter by program
    if program_filter != "All Programs":
        filtered_runs = filtered_runs[filtered_runs["file_name"] == program_filter]

    # Filter by label status
    if show_filter == "Unlabeled Only":
        filtered_runs = filtered_runs[
            ~filtered_runs.apply(lambda row: is_run_fully_labeled(run=row, labeled_runs=labeled_runs), axis=1)
        ]
    elif show_filter == "Labeled Only":
        filtered_runs = filtered_runs[
            filtered_runs.apply(lambda row: is_run_fully_labeled(run=row, labeled_runs=labeled_runs), axis=1)
        ]

    filtered_runs = filtered_runs.reset_index(drop=True)

    if len(filtered_runs) == 0:
        st.info("No runs match the current filters.")
        return

    st.markdown(f"**Showing {len(filtered_runs)} runs**")

    # Run navigation
    # Check if run_idx is in query params (from prev/next buttons)
    default_run_idx = 1
    if "run_idx" in st.query_params:
        try:
            default_run_idx = int(st.query_params["run_idx"]) + 1
            # Clamp to valid range
            default_run_idx = max(1, min(default_run_idx, len(filtered_runs)))
        except (ValueError, TypeError):
            default_run_idx = 1

    run_idx = st.number_input("Run #", min_value=1, max_value=len(filtered_runs), value=default_run_idx, step=1) - 1

    # Update query params to match current run_idx
    st.query_params["run_idx"] = str(run_idx)

    run = filtered_runs.iloc[run_idx]
    run_key = (run["file_name"], run["run"])

    # Display run details
    st.subheader(f"Run {run_idx + 1} of {len(filtered_runs)}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Program:** {run['file_name']}")
    with col2:
        st.markdown(f"**Run:** {run['run']}")
    with col3:
        st.markdown(f"**Num Errors:** {len(run['errors'])}")

    st.subheader("Eval Results")

    # Calculate pass rate
    total_tests = len(run["hard_eval"])
    passed_tests = sum(1 for v in run["hard_eval"].values() if v is True)
    pass_rate = run["pass_rate"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Passed", f"{passed_tests}/{total_tests}")
    with col2:
        st.metric("Pass Rate", f"{pass_rate:.1f}%")

    eval_res = pd.DataFrame([run["hard_eval"]]).T.reset_index()
    eval_res.columns = ["test", "passed"]
    st.dataframe(
        eval_res,
        column_config={
            "test": TextColumn(label="Test", width="large"),
            "passed": CheckboxColumn(label="Passed", width="small"),
        },
        hide_index=True,
    )

    # Show errors in this run
    if run["errors"]:
        st.markdown("#### Errors in This Run")
        for i, error in enumerate(run["errors"]):
            with st.expander(f"Error {i+1}: {error['error_content'][:100]}...", expanded=True):
                st.error(error["error_content"])

                # Show tool call that errored
                if error["error_tool_call"]:
                    func = error["error_tool_call"].get("function", {})
                    st.code(f"{func.get('name', 'unknown')}({func.get('arguments', '')})", language="python")

    # Show trace context
    st.markdown("#### Full Trace Context")
    trace_df = format_trace_for_display(run["trace"], run["errors"])

    # Style the dataframe to highlight error row
    def highlight_error(row):
        # Highlight rows where Error # is not empty
        if row["Error #"] != "":
            return ["background-color: #7E3939"] * len(row)
        return [""] * len(row)

    styled_df = trace_df.style.apply(highlight_error, axis=1)

    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Role": TextColumn(width="small"),
            "Content": TextColumn(width="small"),
            "Tool": TextColumn(width="small"),
            "Tool Args": TextColumn(width="large"),
            "Result": TextColumn(width="large"),
            "Error #": TextColumn(width="small"),
        },
    )

    # Labeling form
    st.markdown("---")
    st.markdown("#### Label This Run")

    # Check if already labeled
    existing_label = labeled_runs.get(run_key)

    # Check if this is a 100% pass rate run
    is_perfect_run = bool(run["pass_rate"] == 100.0)

    if is_perfect_run:
        st.info(
            "✓ This run achieved 100% pass rate. Failure reason will be saved as null, but you can still label errors if present."
        )

    # Run-level failure reason
    st.markdown("**Run-level failure reason:**")
    failure_reason_options = ["give_up", "reasoning", "hallucination", "state_operation_fail", "api_error", "other"]
    default_failure_idx = 0
    if existing_label and existing_label.get("failure_reason"):
        try:
            default_failure_idx = failure_reason_options.index(existing_label["failure_reason"])
        except ValueError:
            default_failure_idx = 0

    failure_reason = st.selectbox(
        "Why didn't this run achieve 100% pass rate?",
        failure_reason_options,
        index=default_failure_idx,
        key=f"failure_reason_{run_idx}",
        disabled=is_perfect_run,
    )

    # Error-level labels
    error_labels = []
    if run["errors"]:
        st.markdown("---")
        st.markdown("**Label each error:**")

        for i, error in enumerate(run["errors"]):
            st.markdown(f"**Error {i+1}:**")
            col1, col2 = st.columns(2)

            # Get existing error label if it exists
            existing_error_label = None
            if existing_label and existing_label.get("errors"):
                for e in existing_label["errors"]:
                    if e.get("error_tool_call_id") == error["error_tool_call_id"]:
                        existing_error_label = e
                        break

            category_options = [
                "forbidden_var",
                "missing_var",
                "state_op_error",
                "name_error",
                "type_error",
                "func_call_error",
                "syntax_error",
                "api_error",
                "logic_error",
                "index_error",
                "other",
            ]
            default_category_idx = 0
            if existing_error_label and existing_error_label.get("category"):
                try:
                    default_category_idx = category_options.index(existing_error_label["category"])
                except ValueError:
                    default_category_idx = 0

            with col1:
                category = st.selectbox(
                    "Category",
                    category_options,
                    index=default_category_idx,
                    key=f"category_{run_idx}_{i}",
                )

            recovery_options = ["retry", "workaround", "correction", "ignore", "none", "not_recovered"]
            default_recovery_idx = 0
            if existing_error_label and existing_error_label.get("recovery_method"):
                try:
                    default_recovery_idx = recovery_options.index(existing_error_label["recovery_method"])
                except ValueError:
                    default_recovery_idx = 0

            with col2:
                recovery_method = st.selectbox(
                    "Recovery Method",
                    recovery_options,
                    index=default_recovery_idx,
                    key=f"recovery_{run_idx}_{i}",
                )

            error_labels.append(
                {
                    "error_tool_call_id": error["error_tool_call_id"],
                    "error_tool_call": error["error_tool_call"],
                    "error_content": error["error_content"],
                    "category": category,
                    "recovery_method": recovery_method,
                }
            )

    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
    with col1:
        if st.button("💾 Save Label", type="primary", use_container_width=True):
            label_entry = {
                "file_name": run["file_name"],
                "run": int(run["run"]),
                "pass_rate": float(run["pass_rate"]),
                "failure_reason": None if is_perfect_run else failure_reason,
                "errors": error_labels,
            }

            # Save to file
            if run_key not in labeled_runs:
                save_label(label_entry, output_path)
            else:
                # Update existing label - rewrite entire file
                labeled_runs[run_key] = label_entry
                with open(output_path, "w") as f:
                    for entry in labeled_runs.values():
                        f.write(json.dumps(entry) + "\n")

            st.session_state.labeled_runs[run_key] = label_entry
            st.success("✓ Label saved!")
            st.rerun()

    with col2:
        if is_run_fully_labeled(run=run, labeled_runs=labeled_runs):
            st.info("✓ Fully labeled")
        elif existing_label:
            st.warning("⚠️ Partially labeled")

    with col3:
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅️ Prev", use_container_width=True, disabled=(run_idx == 0)):
                st.query_params["run_idx"] = str(run_idx - 1)
                st.rerun()
        with col_next:
            if st.button("Next ➡️", use_container_width=True, disabled=(run_idx >= len(filtered_runs) - 1)):
                st.query_params["run_idx"] = str(run_idx + 1)
                st.rerun()

    # Show label summary
    if labeled_runs:
        st.subheader("📈 Label Summary")
        categories = defaultdict(int)
        recovery_methods = defaultdict(int)
        failure_reasons = defaultdict(int)

        for entry in labeled_runs.values():
            # Count run-level failure reasons
            if entry.get("failure_reason"):
                failure_reasons[entry["failure_reason"]] += 1
            else:
                failure_reasons["null (100% pass)"] += 1

            # Count error-level labels
            for error in entry.get("errors", []):
                if error.get("category"):
                    categories[error["category"]] += 1
                if error.get("recovery_method"):
                    recovery_methods[error["recovery_method"]] += 1

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Run Failure Reasons**")
            for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
                st.text(f"{reason}: {count}")

        with col2:
            st.markdown("**Error Categories**")
            if categories:
                for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                    st.text(f"{cat}: {count}")
            else:
                st.text("No errors labeled yet")

        with col3:
            st.markdown("**Error Recovery Methods**")
            if recovery_methods:
                for method, count in sorted(recovery_methods.items(), key=lambda x: x[1], reverse=True):
                    st.text(f"{method}: {count}")
            else:
                st.text("No errors labeled yet")


if __name__ == "__main__":
    main()

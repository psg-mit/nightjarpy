import json
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def load_traces_from_file(file_path: str) -> List[List[Dict[str, Any]]]:
    """Load traces from a JSONL file."""
    traces = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    trace = json.loads(line)
                    traces.append(trace)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return []
    return traces


def parse_trace(trace: List[Dict[str, Any]]) -> pd.DataFrame:
    """Parse a single trace into a structured DataFrame."""
    rows = []
    step = 0

    for item in trace:
        if item["role"] == "user":
            rows.append(
                {
                    "role": "User",
                    "content": item["content"],
                    "tool_name": None,
                    "tool_args": None,
                    "tool_id": None,
                    "tool_result": None,
                }
            )
            step += 1

        elif item["role"] == "assistant":
            if item.get("tool_calls"):
                for tool_call in item["tool_calls"]:
                    rows.append(
                        {
                            "role": "Assistant",
                            "content": item.get("content", ""),
                            "tool_name": tool_call["name"],
                            "tool_args": tool_call["args"],
                            "tool_id": tool_call["id"],
                            "tool_result": None,
                        }
                    )
                    step += 1
            else:
                rows.append(
                    {
                        "role": "Assistant",
                        "content": item.get("content", ""),
                        "tool_name": None,
                        "tool_args": None,
                        "tool_id": None,
                        "tool_result": None,
                    }
                )
                step += 1

        elif item["role"] == "tool":
            # Find the corresponding tool call and update it
            for i, row in enumerate(rows):
                if row["tool_id"] == item["tool_call_id"]:
                    rows[i]["tool_result"] = item["content"]
                    break

    return pd.DataFrame(rows)


def create_trace_summary(traces: List[List[Dict[str, Any]]]) -> pd.DataFrame:
    """Create a summary table of all traces."""
    summary_data = []

    for i, trace in enumerate(traces):
        df = parse_trace(trace)
        tool_calls = df[df["tool_name"].notna()]

        summary_data.append(
            {
                # "Trace #": i + 1,
                # "Total Steps": len(df),
                "Tool Calls": len(tool_calls),
                "Unique Tools": tool_calls["tool_name"].nunique() if len(tool_calls) > 0 else 0,
                "Tools Used": ", ".join(tool_calls["tool_name"].unique()) if len(tool_calls) > 0 else "None",
            }
        )

    return pd.DataFrame(summary_data)


def main():
    st.set_page_config(page_title="Tool Call Trace Visualizer", page_icon="🔍", layout="wide")

    st.title("🔍 Tool Call Trace Visualizer")
    st.markdown("Visualize and explore tool call traces from AI assistants")

    # Sidebar for input options
    st.sidebar.header("Input Options")

    input_method = st.sidebar.radio("Choose input method:", ["Upload JSONL file", "Manual input", "Use sample data"])

    traces = []

    if input_method == "Upload JSONL file":
        uploaded_file = st.sidebar.file_uploader(
            "Upload a JSONL file",
            type=["jsonl", "txt"],
            help="Upload a file containing tool call traces in JSONL format",
        )

        if uploaded_file is not None:
            try:
                content = uploaded_file.read().decode("utf-8")
                for line in content.split("\n"):
                    line = line.strip()
                    if line:
                        trace = json.loads(line)
                        traces.append(trace)
                st.sidebar.success(f"Loaded {len(traces)} traces from file")
            except Exception as e:
                st.sidebar.error(f"Error parsing file: {e}")

    elif input_method == "Manual input":
        st.sidebar.markdown("### Manual Input")
        manual_input = st.sidebar.text_area(
            "Paste your trace data (JSON format):", height=200, help="Paste a JSON array containing trace data"
        )

        if manual_input:
            try:
                trace_data = json.loads(manual_input)
                if isinstance(trace_data, list):
                    if isinstance(trace_data[0], list):
                        traces = trace_data
                    else:
                        traces = [trace_data]
                else:
                    st.sidebar.error("Input must be a list of traces")
            except Exception as e:
                st.sidebar.error(f"Error parsing JSON: {e}")

    else:  # Use sample data
        # Load from the trace.jsonl file in the current directory
        try:
            traces = load_traces_from_file("trace.jsonl")
            st.sidebar.success(f"Loaded {len(traces)} traces from trace.jsonl")
        except Exception as e:
            st.sidebar.error(f"Error loading sample data: {e}")

    if not traces:
        st.info("Please upload a file, enter manual data, or use sample data to get started.")
        return

    # Trace Summary and Statistics section
    st.header("Trace Summary & Statistics")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Trace Summary")
        # Show summary table
        summary_df = create_trace_summary(traces)
        st.dataframe(summary_df, use_container_width=True)

    with col2:
        st.subheader("Trace Statistics")

        col2_1, col2_2, col2_3 = st.columns(3)

        with col2_1:
            st.metric("Total Traces", len(traces))

        with col2_2:
            total_tool_calls = sum(
                len([item for item in trace if item.get("role") == "assistant" and item.get("tool_calls")])
                for trace in traces
            )
            avg_tool_calls = total_tool_calls / len(traces) if traces else 0
            st.metric("Average Tool Calls", f"{avg_tool_calls:.1f}")

        with col2_3:
            all_tools = set()
            for trace in traces:
                for item in trace:
                    if item.get("role") == "assistant" and item.get("tool_calls"):
                        for tool_call in item["tool_calls"]:
                            all_tools.add(tool_call["name"])
            st.metric("Unique Tools", len(all_tools))

        # Show all tools used
        if all_tools:
            st.subheader("Tools Used Across All Traces")
            st.write(", ".join(sorted(all_tools)))

    # Main content area
    # col1, col2 = st.columns([2, 1])

    # with col1:
    # Show detailed trace data
    st.header("Detailed Trace Data")
    if traces:
        df = parse_trace(
            traces[0]
            if len(traces) == 1
            else traces[st.selectbox("Select trace:", range(len(traces)), format_func=lambda x: f"Trace #{x + 1}")]
        )
        # Hide the tool_id column
        df_display = df.drop(columns=["tool_id"], errors="ignore")
        st.dataframe(df_display, use_container_width=True)


if __name__ == "__main__":
    main()

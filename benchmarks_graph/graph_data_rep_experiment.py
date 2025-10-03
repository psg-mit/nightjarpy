"""
Graph Data Representation Experiment

Usage examples:
    # Run all methods with 1 run each
    python graph_data_rep_experiment.py

    # Run only Nightjar method with 3 runs
    python benchmarks_graph/graph_data_rep_experiment.py --model "anthropic/claude-sonnet-4-20250514" --methods interpreter_base_noreg_json --runs 1 --verbose --output_file "benchmarks_graph/graph_results_paper2.jsonl"

    # Run both methods with 5 runs each
    python graph_data_rep_experiment.py --methods Nightjar Oracle --runs 5

    # Test with specific node counts
    python graph_data_rep_experiment.py --nodes 50 100 500 --runs 3

    # Test with a single node count
    python graph_data_rep_experiment.py --nodes 1000 --methods Oracle

    # Generate plots from specific results file
    python benchmarks_graph/graph_data_rep_experiment.py --mode plot --output_file "benchmarks_graph/graph_results_paper.jsonl"
"""

import csv
import inspect
import json
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from func_timeout import func_set_timeout
from matplotlib.lines import Line2D
from openai import OpenAI
from pydantic import BaseModel, Field, create_model, model_serializer
from tabulate import tabulate
from tap import Tap
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import nightjarpy as nj
from nightjarpy import NJ_TELEMETRY, nj_llm_factory
from nightjarpy.configs import (
    INTERPRETER_BASE_NOREG_JSON_CONFIG,
    INTERPRETER_PYTHON_EAGER_CACHE_JSON_CONFIG,
    INTERPRETER_PYTHON_JSON_CONFIG,
    LLMConfig,
)
from nightjarpy.utils.utils import NJ_CACHE


class Args(Tap):
    mode: str = "run"  # Mode: 'run' or 'plot'
    methods: List[str] = [
        "interpreter_python_json",
        "interpreter_python_eager_cache_json",
        "interpreter_base_noreg_json",
        "Oracle",
    ]  # Methods to run
    runs: int = 1  # Number of runs per method
    nodes: List[int] = [25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000]  # Number of nodes to test
    output_file: str = "graph_results_paper.jsonl"  # Output file for results
    plot_path: str = "graph_example.pdf"
    model: str = "anthropic/claude-sonnet-4-20250514"  # Model to use
    nonce: bool = True  # Use nonce
    verbose: bool = False  # Verbose output


PLOT_METHODS = [
    "Oracle",
    # "interpreter_python_json",
    "interpreter_python_eager_cache_json",
    "interpreter_base_noreg_json",
]

METHOD_MAPPING = {
    "interpreter_base_noreg_json": "Nightjar (Baseline)",
    "interpreter_python_json": "Nightjar (Python)",
    "interpreter_python_eager_cache_json": "Nightjar",
    "Oracle": "Pass-by-Copy",
}

args = Args().parse_args()

MODEL = args.model
NONCE = args.nonce
VERBOSE = args.verbose

configs = []
for method in args.methods:
    if method == "interpreter_python_json":
        config = INTERPRETER_PYTHON_JSON_CONFIG
    elif method == "interpreter_python_eager_cache_json":
        config = INTERPRETER_PYTHON_EAGER_CACHE_JSON_CONFIG
    elif method == "interpreter_base_noreg_json":
        config = INTERPRETER_BASE_NOREG_JSON_CONFIG
    else:
        config = INTERPRETER_PYTHON_JSON_CONFIG

    config.with_llm_updates(model=MODEL).with_interpreter_updates(max_effects=100, use_nonce=True)
    configs.append(config)

# nj_llm, oracle_usage = nj_llm_factory(MODEL, max_calls=300)

nj_llm = nj_llm_factory(LLMConfig(model=MODEL), filename="graph_experiment", funcname="main_oracle", max_calls=300)

### Experiment setup


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cached_tokens: Optional[int]


### Method setup


class Graph:
    """A directed graph. Nodes are represented by a set of node values. Edges are represented by a dictionary of source node value to a set of target node values."""

    nodes: set[int]
    edges: dict[int, set[int]]

    def __init__(self, nodes: set[int], edges: dict[int, set[int]]):
        self.nodes = nodes
        self.edges = edges

    def __str__(self):
        return f"Graph(nodes={self.nodes}, edges={self.edges})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other: "Graph"):
        for src in self.edges:
            if src not in other.edges:
                return False

            if not self.edges[src] == other.edges[src]:
                return False

        for src in other.edges:
            if src not in self.edges:
                return False

            if not self.edges[src] == other.edges[src]:
                return False

        return self.nodes == other.nodes

    def __hash__(self):
        return hash((self.nodes, self.edges))

    def __len__(self):
        return len(self.nodes)

    def __contains__(self, node):
        return node in self.nodes


def main_oracle(queries: List[str], graph: Graph) -> Tuple[List[str], Graph]:
    NJ_TELEMETRY.reset()

    def serialize(g: Graph) -> str:
        s = {
            "nodes": list(g.nodes),
            "edges": [{"src": src, "tgts": list(tgts)} for src, tgts in g.edges.items()],
        }
        return json.dumps(s)

    def reify(enc_g: Dict) -> Graph:
        nodes = set(enc_g["nodes"])
        edges = {e["src"]: set(e["tgts"]) for e in enc_g["edges"]}
        return Graph(nodes=nodes, edges=edges)

    responses = []
    for query in queries:
        type_out = nj_llm(
            f"""What is the expected response type of <query>?
<query>{query}</query>""",
            output_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "expected_type",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"t": {"enum": ["integer", "string", "boolean"]}},
                        "required": ["t"],
                        "additionalProperties": False,
                    },
                },
            },
        )

        # if type_out["t"] == "None":
        #     expected_type_schema = {"const": None, "type": "null"}
        # else:
        expected_type_schema = {"type": [type_out["t"], "null"]}

        q_out = nj_llm(
            f"""Perform the <query> with respect to <graph>,
where nodes are paper IDs and edges point from a
cited paper to a set of papers that cite it.
Return `break` as True if the <query> indicates termination.
Else, return a `response`.
If the <graph> was updated, return it as `graph`.
`response` should contain only the value, no prefix or suffix.
<query>{query}</query>
<graph>{serialize(graph)}</graph>""",
            output_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "response": expected_type_schema,
                            "break": {"type": "boolean"},
                            "graph": {
                                "anyOf": [
                                    {
                                        "properties": {
                                            "nodes": {"items": {"type": "integer"}, "type": "array"},
                                            "edges": {
                                                "items": {
                                                    "properties": {
                                                        "src": {"type": "integer"},
                                                        "tgts": {
                                                            "items": {"type": "integer"},
                                                            "type": "array",
                                                        },
                                                    },
                                                    "required": ["src", "tgts"],
                                                    "additionalProperties": False,
                                                    "type": "object",
                                                },
                                                "type": "array",
                                            },
                                        },
                                        "required": ["nodes", "edges"],
                                        "additionalProperties": False,
                                        "type": "object",
                                    },
                                    {"type": "null"},
                                ]
                            },
                        },
                        "required": ["response", "break", "graph"],
                        "additionalProperties": False,
                    },
                    # "schema": OutputSchema.model_json_schema(),
                },
            },
        )

        if q_out["break"]:
            break
        response = q_out["response"]
        if q_out["graph"]:
            graph = reify(q_out["graph"])

        tqdm.write(f"A: {response}")
        responses.append(response)
    oracle_usage = NJ_TELEMETRY.total_llm_usage()
    tokens = TokenUsage(
        input_tokens=oracle_usage.input_tokens,
        output_tokens=oracle_usage.output_tokens,
        cached_tokens=oracle_usage.cached_input_tokens,
    )
    return responses, graph, tokens


def main_nightjar(queries: List[str], graph: Graph) -> Tuple[List[str], Graph]:
    NJ_TELEMETRY.reset()
    responses = []
    for query in queries:
        """natural
        Perform the <query> with respect to <graph>,
        where nodes are paper IDs and edges point
        from a cited paper to a set of papers that cite it.
        Break if the <query> indicates termination.
        Else, save a <:response> and update <graph> to answer <query>.
        <:response> should contain only the value, no prefix or suffix.
        """
        tqdm.write(f"A: {response}")
        responses.append(response)

    usage = NJ_TELEMETRY.total_llm_usage()
    tokens = TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_tokens=usage.cached_input_tokens,
    )
    return responses, graph, tokens


### Experiment setup


Answer = TypeVar("Answer")


@dataclass
class Example(Generic[Answer]):
    query: str
    args: dict[str, Any]
    is_correct: Callable[[Graph, Answer], bool]
    ground_truth: Optional[Answer] = None
    output_type: Any = Optional[str]


@dataclass
class EvalTemplate:
    query_template: str
    example_generator: Callable[[Graph, str], Example]

    def gen_example(self, graph: Graph) -> Example:
        return self.example_generator(graph, self.query_template)


eval_templates: dict[str, EvalTemplate] = {}


@overload
def register_eval_template(
    example_generator: Callable[[Graph, str], Example],
    *,
    query_template: str,
) -> Callable[[Graph], Example]: ...


@overload
def register_eval_template(
    *,
    query_template: str,
) -> Callable[
    [Callable[[Graph, str], Example]],
    Callable[[Graph], Example],
]: ...


def register_eval_template(
    example_generator: Callable[[Graph, str], Example] | None = None,
    *,
    query_template: str,
) -> Callable[[Graph], Example] | Callable[[Callable[[Graph, str], Example]], Callable[[Graph], Example]]:
    def decorator(func: Callable[[Graph, str], Example]) -> Callable[[Graph], Example]:
        temp = EvalTemplate(query_template, func)
        eval_templates[func.__name__] = temp
        return temp.gen_example

    if example_generator is None:
        return decorator
    return decorator(example_generator)


@register_eval_template(query_template="Give the number of papers that cite paper {x}.")
def node_neighbors_gen(graph: Graph, query_template: str) -> Example:
    node_id = random.choice(list(graph.nodes))

    node_edges = graph.edges.get(node_id, set())
    out_degree = len(node_edges)

    return Example(
        query=query_template.format(x=node_id),
        args={"x": node_id},
        output_type=int,
        is_correct=lambda graph, preds: preds[0] == out_degree,
        ground_truth=out_degree,
    )


@register_eval_template(query_template="Does paper {x} directly/indirectly get cited by paper {y}?")
def directed_path_gen(graph: Graph, query_template: str) -> Example:
    node_x = random.choice(list(graph.nodes))
    node_y = random.choice([x for x in graph.nodes if x != node_x])

    def check_path_exists(graph: Graph, nodes: tuple[int, int]) -> bool:
        a, b = nodes
        visited = set()
        stack = [a]
        while stack:
            current = stack.pop()
            if current == b:
                return True
            if current not in visited:
                visited.add(current)
                stack.extend(graph.edges.get(current, set()))  # Follow only outgoing edges
        return False

    ground_truth = check_path_exists(graph, (node_x, node_y))

    return Example(
        query=query_template.format(x=node_x, y=node_y),
        args={"x": node_x, "y": node_y},
        output_type=bool,
        is_correct=lambda graph, preds: preds[0] == ground_truth,
        ground_truth=ground_truth,
    )


@register_eval_template(query_template="How many papers cite both paper {x} and paper {y}?")
def intersection(graph: Graph, query_template: str) -> Example:
    node_x = random.choice(list(graph.nodes))
    node_y = random.choice([x for x in graph.nodes if x != node_x])
    cites_x = graph.edges.get(node_x, set())
    cites_y = graph.edges.get(node_y, set())

    n_cites_both = len(cites_x.intersection(cites_y))

    return Example(
        query=query_template.format(x=node_x, y=node_y),
        args={"x": node_x, "y": node_y},
        output_type=int,
        is_correct=lambda graph, preds: preds[0] == n_cites_both,
        ground_truth=n_cites_both,
    )


@register_eval_template(query_template="Update the graph so paper {x} cites paper {y}.")
def add_edge_gen(graph: Graph, query_template: str) -> Example:
    node_x = random.choice(list(graph.nodes))
    node_y = random.choice([x for x in graph.nodes if x != node_x])

    correct_graph = deepcopy(graph)
    if node_y not in correct_graph.edges:
        correct_graph.edges[node_y] = set()
    correct_graph.edges[node_y].add(node_x)

    def is_correct(graph: Graph, preds: list) -> bool:
        return correct_graph == graph

    return Example(
        query=query_template.format(x=node_x, y=node_y),
        args={"x": node_x, "y": node_y},
        is_correct=is_correct,
        ground_truth=None,
        output_type=str,
    )


@register_eval_template(query_template="Remove paper {x} from the graph completely.")
def remove_node_gen(graph: Graph, query_template: str) -> Example:
    node_x = random.choice(list(graph.nodes))

    correct_graph = deepcopy(graph)
    correct_graph.nodes.remove(node_x)
    for src, targets in correct_graph.edges.items():
        if node_x in targets:
            targets.remove(node_x)
    if node_x in correct_graph.edges:
        del correct_graph.edges[node_x]

    def is_correct(graph: Graph, preds: list) -> bool:
        return correct_graph == graph

    return Example(
        query=query_template.format(x=node_x),
        args={"x": node_x},
        is_correct=is_correct,
        ground_truth=None,
        output_type=str,
    )


@register_eval_template(query_template="Exit, please.")
def invalid_query(graph: Graph, query_template: str) -> Example:

    return Example(
        query=query_template,
        args={},
        is_correct=lambda g, preds: len(preds) == 0,
        ground_truth=None,
        output_type=Optional[str],
    )


def make_graph(n_nodes: int = 10, edge_density: float = 0.5):
    graph = Graph(nodes=set(range(n_nodes)), edges={})
    nodes = list(range(n_nodes))
    random.shuffle(nodes)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < edge_density:
                if nodes[i] not in graph.edges:
                    graph.edges[nodes[i]] = set()
                graph.edges[nodes[i]].add(nodes[j])
    return graph


@dataclass
class Result:
    n_nodes: int
    query: str
    ground_truth: Optional[Any]
    output: Optional[int]
    is_correct: bool
    method: str
    run: int
    time: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    additional_metadata: Optional[Dict[str, Any]] = None

    def model_dump(self) -> dict:
        return {
            "n_nodes": self.n_nodes,
            "run": self.run,
            "method": self.method,
            "is_correct": self.is_correct,
            "query": self.query,
            "time": self.time,
            "ground_truth": self.ground_truth,
            "output": self.output,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "additional_metadata": self.additional_metadata,
        }


results_headers = [
    "N Nodes",
    "Query",
    "Ground Truth",
    "Response",
    "Method",
    "Run",
    "Time",
    "Is Correct",
    "Input Tokens",
    "Output Tokens",
    "Cached Tokens",
    "Additional Metadata",
]


def plot_mode(output_file: str, plot_path: str):
    """Generate plots from the results file."""
    import json

    # Read results from file
    results = []
    with open(output_file, "r") as f:
        for line in f:
            results.append(json.loads(line))

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Pricing configuration
    pricing = {
        "openai/gpt-4.1-2025-04-14": {
            "input": 2,
            "output": 8,
            "cached": 0.5,
        },
    }
    model = "openai/gpt-4.1-2025-04-14"

    def pass_at_k(n: int, c: float, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))

    score_df = (
        df.groupby(["method", "n_nodes", "run"])
        .agg(
            score=("is_correct", "mean"),
            # score_std=('is_correct', 'std'),
        )
        .reset_index()
        .groupby(["method", "n_nodes"])
        .agg(
            score_avg=("score", "mean"),
            score_std=("score", "std"),
            score_min=("score", "min"),
            score_max=("score", "max"),
        )
    )

    # raw_pass_at_k_df = df.groupby(['method', 'n_nodes', 'query']).agg(
    #     n_correct=('is_correct', 'sum'),
    # ).reset_index()

    # for k in [1, 3, 5]:
    #     raw_pass_at_k_df[f"pass_at_{k}"] = raw_pass_at_k_df.apply(lambda row: pass_at_k(runs, row["n_correct"], k), axis=1)

    # pass_at_k_df = raw_pass_at_k_df.groupby(['method', 'n_nodes']).agg(
    #     pass_at_1=('pass_at_1', 'mean'),
    #     pass_at_3=('pass_at_3', 'mean'),
    #     pass_at_5=('pass_at_5', 'mean'),
    # ).reset_index()

    # df[df["input_tokens"] > 0].copy()

    df = df.apply(
        lambda x: (
            {
                **x,
                "input_tokens": (
                    (
                        x["additional_metadata"]["execution_usage"]["input_tokens"]
                        + (
                            x["additional_metadata"]["compilation_usage"]["input_tokens"]
                            if x["query"] == "Give the number of papers that cite paper 19."
                            else 0
                        )
                    )
                    if x["input_tokens"] == -1
                    else x["input_tokens"]
                ),
                "output_tokens": (
                    (
                        x["additional_metadata"]["execution_usage"]["output_tokens"]
                        + (
                            x["additional_metadata"]["compilation_usage"]["output_tokens"]
                            if x["query"] == "Give the number of papers that cite paper 19."
                            else 0
                        )
                    )
                    if x["output_tokens"] == -1
                    else x["output_tokens"]
                ),
                "cached_tokens": (
                    (
                        x["additional_metadata"]["execution_usage"]["cached_tokens"]
                        + (
                            x["additional_metadata"]["compilation_usage"]["cached_tokens"]
                            if x["query"] == "Give the number of papers that cite paper 19."
                            else 0
                        )
                    )
                    if x["cached_tokens"] == -1
                    else x["cached_tokens"]
                ),
                "time": (
                    (
                        x["time"]
                        + (
                            x["additional_metadata"]["compile_time"]
                            if x["query"] == "Give the number of papers that cite paper 19."
                            else 0
                        )
                    )
                    if x["time"] == -1
                    else x["time"]
                ),
            }
            if x["method"] in ["Compilation", "Compilation w/ Source"]
            else x
        ),
        axis=1,
    )

    # Filter out rows where we still don't have valid token counts
    valid_df = df[df["input_tokens"] != -1].copy()

    # valid_df.head()
    valid_df = (
        valid_df.groupby(["method", "n_nodes", "run"])
        .agg(
            time=("time", "sum"),
            input_tokens=("input_tokens", lambda x: sum(x) if all(xx != -1 for xx in x) else -1),
            #   'sum'),
            output_tokens=("output_tokens", lambda x: sum(x) if all(xx != -1 for xx in x) else -1),
            cached_tokens=("cached_tokens", lambda x: sum(x) if all(xx != -1 for xx in x) else -1),
        )
        .reset_index()
    )

    valid_df["net_input_tokens"] = valid_df["input_tokens"] - valid_df["cached_tokens"]

    valid_df["cost"] = (
        valid_df["net_input_tokens"] * pricing[model]["input"] / 1000000
        + valid_df["output_tokens"] * pricing[model]["output"] / 1000000
        + valid_df["cached_tokens"] * pricing[model]["cached"] / 1000000
    )

    agg_df = (
        df.groupby(["method", "n_nodes"])
        .agg(
            min_time=("time", "min"),
            avg_time=("time", "mean"),
            max_time=("time", "max"),
        )
        .reset_index()
    )

    valid_agg_df = (
        valid_df.groupby(["method", "n_nodes"])
        .agg(
            # cached_tokens=('cached_tokens', 'min'),
            min_input_tokens=("input_tokens", "min"),
            avg_input_tokens=("input_tokens", "mean"),
            max_input_tokens=("input_tokens", "max"),
            min_output_tokens=("output_tokens", "min"),
            avg_output_tokens=("output_tokens", "mean"),
            max_output_tokens=("output_tokens", "max"),
            min_net_input_tokens=("net_input_tokens", "min"),
            avg_net_input_tokens=("net_input_tokens", "mean"),
            max_net_input_tokens=("net_input_tokens", "max"),
            min_cost=("cost", "min"),
            avg_cost=("cost", "mean"),
            max_cost=("cost", "max"),
        )
        .reset_index()
    )

    # valid_df
    # agg_df = pd.merge(pass_at_k_df, paper_paper_paper_paper_agg_df, on=['method', 'n_nodes'], how='left')
    agg_df = pd.merge(agg_df, score_df, on=["method", "n_nodes"], how="left")
    agg_df = pd.merge(agg_df, valid_agg_df, on=["method", "n_nodes"], how="left")

    agg_df = agg_df[agg_df["method"].isin(PLOT_METHODS)]

    agg_df.replace(
        METHOD_MAPPING,
        inplace=True,
    )

    paper_agg_df = agg_df
    # agg_df.head()

    color_palette = {
        "Nightjar": "#4F6C95",  # blue
        "Nightjar (Baseline)": "#52B7A6",  # blue
        "Nightjar (Pass-by-Reference)": "#4F6C95",  # blue
        "Pass-by-Copy": "#D78175",  # red
    }

    markers = {
        "Nightjar": "o",
        "Nightjar (Baseline)": "D",
        "Pass-by-Copy": "s",
        "Compilation": "D",
        "Compilation w/ Source": "X",
    }

    from matplotlib.lines import Line2D

    # Set Helvetica as the default font
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = ['Helvetica']
    # # If you're using math text, you might want:
    # plt.rcParams['mathtext.fontset'] = 'dejavusans'  # This gives a clean sans-serif look for math

    fig, ax = plt.subplots(2, 3, figsize=(8, 4))
    # sns.lineplot(data=paper_agg_df, x="n_nodes", y="pass_at_1", hue="method", ax=ax[0][0], marker="o", palette=color_palette)
    # sns.lineplot(data=paper_agg_df, x="n_nodes", y="pass_at_3", hue="method", ax=ax[0][1], marker="o", palette=color_palette)
    # sns.lineplot(data=paper_agg_df, x="n_nodes", y="pass_at_5", hue="method", ax=ax[0][2], marker="o", palette=color_palette)
    sns.lineplot(
        data=paper_agg_df,
        x="n_nodes",
        y="score_avg",
        hue="method",
        ax=ax[0][0],
        style="method",
        markers=markers,
        palette=color_palette,
    )
    for method in paper_agg_df["method"].unique():
        data = paper_agg_df[paper_agg_df["method"] == method]
        ax[0][0].fill_between(
            data["n_nodes"],
            data["score_min"],
            data["score_max"],
            alpha=0.2,
            color=color_palette[method],
            label="_nolegend_",
        )

    # ax[0][0].set_ylim(-0.05, 1.05)
    # ax[0][1].set_ylim(-0.05, 1.05)
    # ax[0][2].set_ylim(-0.05, 1.05)
    # print(paper_agg_df)
    paper_agg_df = paper_agg_df[paper_agg_df["min_input_tokens"] >= 1].copy()

    sns.lineplot(
        data=paper_agg_df,
        x="n_nodes",
        y="avg_time",
        hue="method",
        ax=ax[0][1],
        style="method",
        markers=markers,
        palette=color_palette,
    )
    sns.lineplot(
        data=paper_agg_df,
        x="n_nodes",
        y="avg_cost",
        hue="method",
        ax=ax[0][2],
        style="method",
        markers=markers,
        palette=color_palette,
    )
    sns.lineplot(
        data=paper_agg_df,
        x="n_nodes",
        y="avg_input_tokens",
        hue="method",
        ax=ax[1][0],
        style="method",
        markers=markers,
        palette=color_palette,
    )
    sns.lineplot(
        data=paper_agg_df,
        x="n_nodes",
        y="avg_output_tokens",
        hue="method",
        ax=ax[1][1],
        style="method",
        markers=markers,
        palette=color_palette,
    )
    sns.lineplot(
        data=paper_agg_df,
        x="n_nodes",
        y="avg_net_input_tokens",
        hue="method",
        ax=ax[1][2],
        style="method",
        markers=markers,
        palette=color_palette,
    )
    for method in paper_agg_df["method"].unique():
        data = paper_agg_df[paper_agg_df["method"] == method]
        ax[0][1].fill_between(
            data["n_nodes"],
            data["min_time"],
            data["max_time"],
            alpha=0.2,
            color=color_palette[method],
            label="_nolegend_",
        )
        ax[1][0].fill_between(
            data["n_nodes"],
            data["min_input_tokens"],
            data["max_input_tokens"],
            alpha=0.2,
            color=color_palette[method],
            label="_nolegend_",
        )
        ax[1][1].fill_between(
            data["n_nodes"],
            data["min_output_tokens"],
            data["max_output_tokens"],
            alpha=0.2,
            color=color_palette[method],
            label="_nolegend_",
        )
        ax[1][2].fill_between(
            data["n_nodes"],
            data["min_net_input_tokens"],
            data["max_net_input_tokens"],
            alpha=0.2,
            color=color_palette[method],
            label="_nolegend_",
        )
        ax[0][2].fill_between(
            data["n_nodes"],
            data["min_cost"],
            data["max_cost"],
            alpha=0.2,
            color=color_palette[method],
            label="_nolegend_",
        )

    for row in ax:
        for a in row:
            # turn off legend
            a.legend().set_visible(False)
            a.grid(axis="both", linestyle="--", alpha=0.3, color="gray")
            a.set_axisbelow(True)
            a.set_xlabel(None)

    CONTEXT_OUT_LINE = 1000

    for a in ax[0][:1]:
        ymin, ymax = a.get_ylim()
        ymin = -0.05
        ymax = 1.05
        a.set_ylim(ymin, ymax)
        a.vlines(CONTEXT_OUT_LINE, ymin, ymax, color="black", linestyle=":", zorder=0)
        a.set_xscale("log")

    ymin, ymax = ax[0][1].get_ylim()
    ymin = 0.1e0
    ymax = 1.5e3
    ax[0][1].set_yscale("log")
    ax[0][1].set_ylim(ymin, ymax)
    ax[0][1].vlines(CONTEXT_OUT_LINE, ymin, ymax, color="black", linestyle=":", zorder=0)
    ax[0][1].set_xscale("log")

    ymin, ymax = ax[1][0].get_ylim()
    ymin = 4e2
    ymax = 1e8 * 1.3
    ax[1][0].set_ylim(ymin, ymax)
    ax[1][0].vlines(CONTEXT_OUT_LINE, ymin, ymax, color="black", linestyle=":", zorder=0)
    ymin, ymax = ax[1][1].get_ylim()
    ymin = 1e2
    ymax = 1e5
    ax[1][1].set_ylim(ymin, ymax)
    ax[1][1].vlines(CONTEXT_OUT_LINE, ymin, ymax, color="black", linestyle=":", zorder=0)
    ymin, ymax = ax[1][2].get_ylim()
    ymin = 4e2
    ymax = 1e8 * 1.3
    ax[1][2].set_ylim(ymin, ymax)
    ax[1][2].vlines(CONTEXT_OUT_LINE, ymin, ymax, color="black", linestyle=":", zorder=0)

    # ax[0][2].set_yscale("linear")
    ymin, ymax = ax[0][2].get_ylim()
    ax[0][2].set_yscale("log")
    ymin = 1e-2
    ymax = 1e2
    ax[0][2].set_ylim(ymin, ymax)
    ax[0][2].vlines(CONTEXT_OUT_LINE, ymin, ymax, color="black", linestyle=":", zorder=0)
    ax[0][2].set_xscale("log")

    for a in ax[1][:]:
        a.set_xscale("log")
        a.set_yscale("log")

    for a in ax[1]:
        a.set_xlabel("Number of Nodes\n(log scale)")

    ax[0][0].set_title("Average Pass Rate")
    # ax[0][1].set_title("Pass@3")
    # ax[0][2].set_title("Pass@5")
    ax[0][0].set_ylabel("Pass Rate")
    # ax[0][1].set_ylabel("Pass@3")
    # ax[0][2].set_ylabel("Pass@5")
    ax[0][1].set_title("Execution Time")
    ax[0][1].set_ylabel("Time (s) (log scale)")
    ax[1][0].set_title("Input Tokens")
    ax[1][0].set_ylabel("Tokens (log scale)")
    ax[1][1].set_title("Output Tokens")
    ax[1][1].set_ylabel("Tokens (log scale)")
    ax[1][2].set_title("Uncached Input Tokens")
    ax[1][2].set_ylabel("Tokens (log scale)")
    ax[0][2].set_title("Cost")
    ax[0][2].set_ylabel("USD (log scale)")
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color_palette["Nightjar"],
            label="Nightjar",
            marker=markers["Nightjar"],
            markeredgecolor="white",
        ),
        Line2D(
            [0],
            [0],
            color=color_palette["Nightjar (Baseline)"],
            label="Nightjar (Baseline)",
            marker=markers["Nightjar (Baseline)"],
            markeredgecolor="white",
        ),
        Line2D(
            [0],
            [0],
            color=color_palette["Pass-by-Copy"],
            label="Manual Impl (Pass-by-Copy)",
            marker=markers["Pass-by-Copy"],
            markeredgecolor="white",
        ),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.subplots_adjust(wspace=0.5, hspace=0.4)
    # plt.tight_layout()
    # plt.savefig("../../nightjar-paper/assets/graph_example.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")  # Save copy in current directory


if __name__ == "__main__":
    if args.mode == "plot":
        print(f"Generating plots from: {args.output_file}")
        plot_mode(args.output_file, args.plot_path)
        print("Plots generated successfully!")
    else:
        print(f"Running experiment with:")
        print(f"  Methods: {args.methods}")
        print(f"  Runs per method: {args.runs}")
        print(f"  Node counts: {args.nodes}")
        print(f"  Output file: {args.output_file}")
        print(f"  Model: {args.model}")
        print(f"  Nonce: {args.nonce}")
        print(f"  Verbose: {args.verbose}")
        print()

        ALL_NODES = [25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000]

        n_nodes = args.nodes
        n_examples = 1
        graph_queries: dict[int, tuple[Graph, list[Example]]] = {}

        query_file = "benchmarks_graph/graph_queries_paper.pkl"

        if os.path.exists(query_file):
            with open(query_file, "rb") as f:
                graph_queries: dict[int, tuple[Graph, list[Example]]] = dill.load(f)
        else:
            random.seed(42)
            for n in ALL_NODES:
                graph = make_graph(n)
                examples = []
                for template in eval_templates.values():
                    for _ in range(n_examples):
                        ex = template.gen_example(graph)
                        while ex.query in [q.query for q in examples]:
                            ex = template.gen_example(graph)
                        examples.append(ex)
                graph_queries[n] = (graph, examples)

            random.seed()
            with open(query_file, "wb") as f:
                dill.dump(graph_queries, f)

        results = []
        already_done = set()
        output_file = args.output_file
        if os.path.exists(output_file):
            with open(output_file) as f:
                for line in f:
                    res = json.loads(line)
                    results.append(Result(**res))
                    already_done.add((res["n_nodes"], res["query"], res["method"], res["run"]))

        # Create method mapping
        method_functions = {
            "interpreter_python_json": main_nightjar,
            "interpreter_python_eager_cache_json": main_nightjar,
            "interpreter_base_noreg_json": main_nightjar,
            "Oracle": main_oracle,
        }

        # Filter methods based on command line arguments
        # settings = [(method, method_functions[method]) for method in args.methods if method in method_functions]

        # if not settings:
        #     print("Error: No valid methods specified. Available methods: Nightjar, Oracle")
        #     exit(1)

        with logging_redirect_tqdm():
            for method in tqdm(args.methods, desc="Methods"):
                if method == "Oracle":
                    fn = main_oracle
                else:
                    if method == "interpreter_python_json":
                        config = INTERPRETER_PYTHON_JSON_CONFIG
                    elif method == "interpreter_python_eager_cache_json":
                        config = INTERPRETER_PYTHON_EAGER_CACHE_JSON_CONFIG
                    elif method == "interpreter_base_noreg_json":
                        config = INTERPRETER_BASE_NOREG_JSON_CONFIG
                    else:
                        print("Error: unknown method")
                        continue

                    config.with_llm_updates(model=MODEL).with_interpreter_updates(max_effects=100)
                    fn = nj.fn(main_nightjar, config=config)

                @func_set_timeout(timeout=800)
                def fn_run(queries: List[str], graph: Graph):
                    return fn(queries, graph)

                for run in tqdm(range(args.runs), desc="Runs"):
                    NJ_CACHE.clear_cache()

                    for n, (og_graph, examples) in tqdm(graph_queries.items(), desc="Nodes"):
                        if n not in args.nodes:
                            continue

                        additional_metadata = {}

                        graph_done = True
                        for example in examples:
                            if (n, example.query, method, run) not in already_done:
                                graph_done = False

                        if graph_done:
                            continue

                        # if method in ["Naive Compilation", "Naive Compilation w/ Source"]:
                        #     compile_t1 = time.time()
                        #     # compile the code
                        #     fn_run, compilation_usage = fn()  # type: ignore
                        #     additional_metadata["compilation_usage"] = {
                        #         "input_tokens": compilation_usage["input_tokens"],
                        #         "output_tokens": compilation_usage["output_tokens"],
                        #         "cached_tokens": compilation_usage["cached_token_reads"],
                        #     }
                        #     compile_t2 = time.time()
                        #     additional_metadata["compile_time"] = compile_t2 - compile_t1
                        # else:

                        additional_metadata["compilation_usage"] = {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "cached_tokens": 0,
                        }

                        for eg_i, example in tqdm(enumerate(examples), desc="Queries"):
                            if (n, example.query, method, run) in already_done:
                                continue

                            tqdm.write(f"Running {method} for {n} nodes on run {run}")
                            tqdm.write(f"Query: {example.query}")
                            # reset graph
                            graph = deepcopy(og_graph)

                            t1 = time.time()
                            try:
                                output, graph, tokens = fn_run([example.query], graph)  # type: ignore

                            except Exception as e:
                                tqdm.write(f"Error: {e}")
                                output = None
                                tokens = TokenUsage(input_tokens=-1, output_tokens=-1, cached_tokens=-1)
                            t2 = time.time()

                            try:
                                correctness = example.is_correct(graph, output)
                            except Exception as e:
                                tqdm.write(f"Error: {e}")
                                correctness = False

                            if tokens.input_tokens == -1:
                                additional_metadata["execution_usage"] = {
                                    "input_tokens": -1,
                                    "output_tokens": -1,
                                    "cached_tokens": -1,
                                }
                            else:
                                additional_metadata["execution_usage"] = {
                                    "input_tokens": tokens.input_tokens,
                                    "output_tokens": tokens.output_tokens,
                                    "cached_tokens": tokens.cached_tokens,
                                }
                            if additional_metadata["execution_usage"]["input_tokens"] == -1:
                                res = Result(
                                    n_nodes=n,
                                    query=example.query,
                                    ground_truth=example.ground_truth,
                                    output=[str(x)[:1000] for x in output] if output is not None else None,
                                    is_correct=correctness,
                                    method=method,
                                    run=run,
                                    time=t2 - t1,
                                    input_tokens=-1,
                                    output_tokens=-1,
                                    cached_tokens=-1,
                                )
                            else:
                                res = Result(
                                    n_nodes=n,
                                    query=example.query,
                                    ground_truth=example.ground_truth,
                                    output=[str(x)[:1000] for x in output] if output is not None else None,
                                    is_correct=correctness,
                                    method=method,
                                    run=run,
                                    time=t2 - t1,
                                    input_tokens=additional_metadata["execution_usage"]["input_tokens"]
                                    + additional_metadata["compilation_usage"]["input_tokens"],
                                    output_tokens=additional_metadata["execution_usage"]["output_tokens"]
                                    + additional_metadata["compilation_usage"]["output_tokens"],
                                    cached_tokens=additional_metadata["execution_usage"]["cached_tokens"]
                                    + additional_metadata["compilation_usage"]["cached_tokens"],
                                )

                            res.additional_metadata = additional_metadata

                            with open(output_file, "a") as f:
                                f.write(json.dumps(res.model_dump()) + "\n")
                            results.append(res)
                            already_done.add((n, example.query, method, run))

        os.system("say 'experiment done'")

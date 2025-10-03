import ast
import json
import logging
import os
import runpy
import sys
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import datasets
import func_timeout
import numpy as np
from dspy.utils.parallelizer import ParallelExecutor
from func_timeout import func_set_timeout
from openai import OpenAI
from openai.types.container_create_params import ExpiresAfter
from pydantic import BaseModel
from tap import Tap
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import nightjarpy as nj
import wandb
from nightjarpy import NJ_TELEMETRY, nj_llm_factory
from nightjarpy.configs import (
    INTERPRETER_BASE_NOREG_JSON_CONFIG,
    INTERPRETER_PYTHON_BASE_ISOLATED_NOREG_JSON_CONFIG,
    INTERPRETER_PYTHON_BASE_NOREG_JSON_CONFIG,
    INTERPRETER_PYTHON_CACHE_JSON_CONFIG,
    INTERPRETER_PYTHON_EAGER_CACHE_JSON_CONFIG,
    INTERPRETER_PYTHON_JSON_CONFIG,
    LLMConfig,
)
from nightjarpy.llm.factory import create_llm
from nightjarpy.utils import NJ_CACHE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HANDLERS = Literal[
    "interprter_python_json",
    "interpreter_python_eager_cache_json",
    "manual",
    "codetool",
    "customtool",
]

HANDLER_CONFIG_MAPPING = {
    "interpreter_python_json": INTERPRETER_PYTHON_JSON_CONFIG,
    "interpreter_python_cache_json": INTERPRETER_PYTHON_CACHE_JSON_CONFIG,
    "interpreter_python_eager_cache_json": INTERPRETER_PYTHON_EAGER_CACHE_JSON_CONFIG,
    "interpreter_base_noreg_json": INTERPRETER_BASE_NOREG_JSON_CONFIG,
    "interpreter_python_base_isolated_json": INTERPRETER_PYTHON_BASE_ISOLATED_NOREG_JSON_CONFIG,
    "interpreter_python_base_json": INTERPRETER_PYTHON_BASE_NOREG_JSON_CONFIG,
}


class Options(Tap):
    output_file: str = "benchmarks_gsm8k/results/test/results.jsonl"  # File to save evaluation results (JSON Lines)
    trace_file: str = "benchmarks_gsm8k/results/test/traces.jsonl"  # File to save trace output
    runs: int = 1  # Number of times to run each program
    timeout: int = 1000  # Timeout in seconds for each program run
    model: str = "openai/gpt-4.1"  # Model name to use for natural language
    max_tool_calls: int = 300  # Maximum number of tool calls to make
    handler_name: HANDLERS = "interprter_python_json"  # Handler to use
    verbose: bool = False  # Whether to print verbose output
    system_prompt_path: Optional[str] = None  # Path to the system prompt file
    wandb_project: str = "nightjar"  # Wandb project name
    wandb_run_name: Optional[str] = None  # Wandb run name
    parallel: bool = False  # Whether to parallelize execution
    num_workers: int = 4  # Number of parallel workers (only used if --parallel is True)
    batch_size: int = 50  # Batch size for parallel execution (only used if --parallel is True)
    reuse_container: bool = False


class TimeoutException(Exception):
    def __str__(self) -> str:
        return "Timeout"


def load_namespace_from_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a Python script as a namespace by executing it.
    """
    if not os.path.exists(file_path):
        tqdm.write(f"File not found: {file_path}", file=sys.stderr)
        return None

    globals_: Dict[str, Any] = {"__name__": "__main__"}
    try:
        with logging_redirect_tqdm():
            namespace = runpy.run_path(file_path, globals_)
        return namespace
    except Exception as e:
        tqdm.write(f"Failed to run script {file_path}: {e}", file=sys.stderr)
        return None


class TraceEntry(BaseModel):
    question: str
    model: str
    run: int  # Run number
    trace: List[Dict[str, Any]]


@dataclass
class TestResult:
    question: str
    model: str
    run: int
    runtime: Optional[float]
    compile_time: Optional[float]
    result: Optional[float]
    expected_result: float
    eval_result: bool
    errors: Dict[str, Any]
    token_count: Optional[Dict[str, int]]
    n_tool_calls: int


def load_completed_runs(results_file: str) -> set:
    """
    Reads the results file if it exists and returns a set of (program_id, run) tuples
    that have already been completed.
    """
    completed = set()
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                for line in f:
                    try:
                        entry: dict = json.loads(line)
                        question = entry.get("question")
                        expected_result = entry.get("expected_result")
                        run = entry.get("run", 0)
                        if question is not None and expected_result is not None:
                            completed.add((question, expected_result, run))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            tqdm.write(f"Failed to load existing results: {e}", file=sys.stderr)
    else:
        dirpath = os.path.dirname(results_file)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
    return completed


def save_result(result: TestResult, output_file: str):
    """Save a single result to the output file in JSONL format."""
    errors_data = {}
    if result.errors is None:
        result.errors = {}
    for k, v in result.errors.items():
        try:
            errors_data[k] = f"{v.__class__.__name__}: {str(v)}" if v is not None else None
        except Exception as e:
            errors_data[k] = f"{v.__class__.__name__}"

    result_data = {
        "question": result.question,
        "model": result.model,
        "run": result.run,
        "runtime": result.runtime,
        "compile_time": result.compile_time,
        "result": result.result,
        "expected_result": result.expected_result,
        "eval_result": result.eval_result,
        "token_count": result.token_count,
        "errors": errors_data,
        "n_tool_calls": result.n_tool_calls,
    }

    tqdm.write(f"Saving result...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "a") as f:
        f.write(json.dumps(result_data) + "\n")
        f.flush()


def save_trace(result: TraceEntry, output_file: str):
    """Save a single trace to the output file in JSONL format."""
    trace_data = result.model_dump()
    tqdm.write(f"Saving trace...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "a") as f:
        f.write(json.dumps(trace_data) + "\n")
        f.flush()


def run_program(args) -> Tuple[TestResult, TraceEntry]:
    NJ_CACHE.clear_cache()
    NJ_TELEMETRY.reset()

    dest_path, run_index, question, expected_answer, container_id, opts = args
    opts: Options

    compile_time = None

    # Load the compiled program
    namespace = load_namespace_from_file(dest_path)
    if namespace is None:
        return (
            TestResult(
                question=question,
                model=opts.model,
                run=run_index,
                runtime=None,
                compile_time=compile_time,
                result=None,
                expected_result=expected_answer,
                eval_result=False,
                errors={"load": "Failed to load compiled program"},
                token_count=None,
                n_tool_calls=0,
            ),
            TraceEntry(
                question=question,
                model=opts.model,
                run=run_index,
                trace=[],
            ),
        )

    # Run the program on all test cases
    errors = {}
    eval_result = False
    answer = None

    if opts.handler_name == "manual":
        config = LLMConfig(model=opts.model)
        nj_llm = nj_llm_factory(
            config, filename=os.path.basename(dest_path), funcname="main", max_calls=opts.max_tool_calls
        )

        @func_set_timeout(opts.timeout)
        def run_test_case() -> float:
            return namespace["main"](question, nj_llm)

        def get_telemetry() -> Tuple[Dict[str, int], List[Dict[str, Any]], int]:
            usage = NJ_TELEMETRY.total_llm_usage()
            token_usage = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cached_token_reads": usage.cached_input_tokens or 0,
                "cached_token_writes": usage.cached_output_tokens or 0,
                "total_tokens": (
                    usage.input_tokens
                    + usage.output_tokens
                    + (usage.cached_input_tokens or 0)
                    + (usage.cached_output_tokens or 0)
                ),
            }
            return token_usage, [], 0

    elif opts.handler_name == "codetool":
        config = LLMConfig(model=opts.model)
        llm = create_llm(config)

        @func_set_timeout(opts.timeout)
        def run_test_case() -> float:
            return namespace["main"](
                problem=question, llm=llm, max_tool_calls=opts.max_tool_calls, container_id=container_id
            )

        def get_telemetry() -> Tuple[Dict[str, int], List[Dict[str, Any]], int]:
            usage = llm.get_usage()
            if not usage:
                token_usage = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_token_reads": 0,
                    "cached_token_writes": 0,
                    "total_tokens": 0,
                }
            else:
                token_usage = {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cached_token_reads": usage.cached_input_tokens or 0,
                    "cached_token_writes": usage.cached_output_tokens or 0,
                    "total_tokens": (
                        usage.input_tokens
                        + usage.output_tokens
                        + (usage.cached_input_tokens or 0)
                        + (usage.cached_output_tokens or 0)
                    ),
                }

            traces = deepcopy(NJ_TELEMETRY.trace) or []
            traces = [t.model_dump() for t in traces]
            n_tool_calls = NJ_TELEMETRY.n_tool_calls
            return token_usage, traces, n_tool_calls

    elif opts.handler_name == "customtool":
        config = LLMConfig(model=opts.model)
        llm = create_llm(config)

        @func_set_timeout(opts.timeout)
        def run_test_case() -> float:
            return namespace["main"](problem=question, llm=llm, max_tool_calls=opts.max_tool_calls)

        def get_telemetry() -> Tuple[Dict[str, int], List[Dict[str, Any]], int]:
            traces = llm.tool_trace
            n_tool_calls = llm.n_tool_calls
            token_usage = {
                "input_tokens": sum(usage.prompt_tokens for usage in llm.usage),
                "output_tokens": sum(usage.completion_tokens for usage in llm.usage),
                "cached_token_reads": sum(
                    usage.cached_token_reads if usage.cached_token_reads is not None else 0 for usage in llm.usage
                ),
                "cached_token_writes": sum(
                    usage.cached_token_writes if usage.cached_token_writes is not None else 0 for usage in llm.usage
                ),
                "total_tokens": sum(usage.prompt_tokens + usage.completion_tokens for usage in llm.usage),
            }
            return token_usage, traces, n_tool_calls

    else:

        @func_set_timeout(opts.timeout)
        def run_test_case() -> float:
            return namespace["main"](question)

        def get_telemetry() -> Tuple[Dict[str, int], List[Dict[str, Any]], int]:
            usage = NJ_TELEMETRY.total_llm_usage()
            token_usage = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cached_token_reads": usage.cached_input_tokens or 0,
                "cached_token_writes": usage.cached_output_tokens or 0,
                "total_tokens": (
                    usage.input_tokens
                    + usage.output_tokens
                    + (usage.cached_input_tokens or 0)
                    + (usage.cached_output_tokens or 0)
                ),
            }
            traces = deepcopy(NJ_TELEMETRY.trace) or []
            traces = [t.model_dump() for t in traces]
            n_tool_calls = NJ_TELEMETRY.n_tool_calls
            return token_usage, traces, n_tool_calls

    runtime = None
    start_time = time.time()
    try:
        # Run the main function
        answer = run_test_case()
        runtime = time.time() - start_time

        # Compare results
        eval_result = answer == expected_answer

    except func_timeout.exceptions.FunctionTimedOut as e:
        runtime = time.time() - start_time
        tqdm.write(f"Timed out", file=sys.stderr)
        errors[f"runtime"] = e
    except Exception as e:
        runtime = time.time() - start_time
        tqdm.write(f"Error: {e}", file=sys.stderr)
        errors[f"runtime"] = e

    token_usage, traces, n_tool_calls = get_telemetry()

    test_result = TestResult(
        question=question,
        model=opts.model,
        run=run_index,
        runtime=runtime,
        compile_time=None,
        result=answer,
        expected_result=expected_answer,
        eval_result=eval_result,
        errors=errors,
        token_count=token_usage,
        n_tool_calls=n_tool_calls,
    )

    trace_entry = TraceEntry(
        question=question,
        model=opts.model,
        run=run_index,
        trace=traces,
    )

    return test_result, trace_entry


def modify_nj_fn_decorators(src_code: str, config_json: str) -> str:
    """
    Use AST to modify @nj.fn decorators to include config parameter.
    """
    try:
        # Parse the source code into an AST
        tree = ast.parse(src_code)

        # Create a visitor to modify @nj.fn decorators
        class NJFnDecoratorModifier(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                # Check if the function has decorators
                if node.decorator_list:
                    new_decorators = []
                    for decorator in node.decorator_list:
                        # Check if this is a @nj.fn decorator (with or without parentheses)
                        is_nj_fn = False

                        # Case 1: @nj.fn (no parentheses)
                        if (
                            isinstance(decorator, ast.Attribute)
                            and isinstance(decorator.value, ast.Name)
                            and decorator.value.id == "nj"
                            and decorator.attr == "fn"
                        ):
                            is_nj_fn = True
                            # Convert to call with config
                            new_decorator = ast.Call(
                                func=ast.Attribute(value=ast.Name(id="nj", ctx=ast.Load()), attr="fn", ctx=ast.Load()),
                                args=[],
                                keywords=[
                                    ast.keyword(
                                        arg="config",
                                        value=ast.Call(
                                            func=ast.Attribute(
                                                value=ast.Attribute(
                                                    value=ast.Name(id="nj", ctx=ast.Load()),
                                                    attr="Config",
                                                    ctx=ast.Load(),
                                                ),
                                                attr="model_validate_json",
                                                ctx=ast.Load(),
                                            ),
                                            args=[ast.Constant(value=config_json)],
                                            keywords=[],
                                        ),
                                    )
                                ],
                            )

                        # Case 2: @nj.fn() (with parentheses)
                        elif (
                            isinstance(decorator, ast.Call)
                            and isinstance(decorator.func, ast.Attribute)
                            and isinstance(decorator.func.value, ast.Name)
                            and decorator.func.value.id == "nj"
                            and decorator.func.attr == "fn"
                        ):
                            is_nj_fn = True
                            # Add config to existing call
                            new_keywords = list(decorator.keywords) if decorator.keywords else []
                            # Check if config is already present
                            config_exists = any(kw.arg == "config" for kw in new_keywords)
                            if not config_exists:
                                new_keywords.append(
                                    ast.keyword(
                                        arg="config",
                                        value=ast.Call(
                                            func=ast.Attribute(
                                                value=ast.Attribute(
                                                    value=ast.Name(id="nj", ctx=ast.Load()),
                                                    attr="Config",
                                                    ctx=ast.Load(),
                                                ),
                                                attr="model_validate_json",
                                                ctx=ast.Load(),
                                            ),
                                            args=[ast.Constant(value=config_json)],
                                            keywords=[],
                                        ),
                                    )
                                )

                            new_decorator = ast.Call(func=decorator.func, args=decorator.args, keywords=new_keywords)

                        if is_nj_fn:
                            new_decorators.append(new_decorator)
                        else:
                            # Keep other decorators as-is
                            new_decorators.append(decorator)

                    # Update the function's decorators
                    node.decorator_list = new_decorators

                return self.generic_visit(node)

        # Apply the transformer
        modifier = NJFnDecoratorModifier()
        modified_tree = modifier.visit(tree)

        # Convert back to source code
        modified_src = ast.unparse(modified_tree)
        return modified_src

    except Exception as e:
        # If AST manipulation fails, fall back to string replacement
        tqdm.write(f"AST manipulation failed, falling back to string replacement: {e}", file=sys.stderr)
        return src_code.replace("@nj.fn", f'@nj.fn(config=nj.Config.model_validate_json("""{config_json}"""))')


def add_config(
    handler_name: HANDLERS,
    program_path: str,
    opts: Options,
) -> str:
    """If nightjar, replace @nj.fn with config"""
    with open(program_path) as f:
        src_code = f.read()

    config = HANDLER_CONFIG_MAPPING[handler_name]

    if config.interpreter_config is not None:
        config = config.with_interpreter_updates(max_effects=opts.max_tool_calls)
    if config.compiler_config is not None:
        config = config.with_compiler_updates(max_runtime_calls=opts.max_tool_calls)

    # Update LLM config
    config = config.with_llm_updates(model=opts.model, temperature=1.0)
    config_json = config.model_dump_json()

    # Use AST to modify the decorators
    src_code = modify_nj_fn_decorators(src_code, config_json)

    # Write the modified source code to a temporary file and return its path
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(src_code)
        tmp_file.flush()
        temp_path = tmp_file.name

    return temp_path


def main(opts: Options):
    """Main function to run the evaluation."""

    # Initialize wandb
    run_name = (
        opts.wandb_run_name or f"gsm8k-{opts.model}-{opts.handler_name}-mtc{opts.max_tool_calls}-{int(time.time())}"
    )
    wandb.init(
        project=opts.wandb_project,
        name=run_name,
        config={
            "model": opts.model,
            "handler_name": opts.handler_name,
            "runs": opts.runs,
            "timeout": opts.timeout,
            "max_tool_calls": opts.max_tool_calls,
            "parallel": opts.parallel,
            "num_workers": opts.num_workers,
            "batch_size": opts.batch_size,
        },
    )

    programs_dir = Path("/Users/ellieyhc/Documents/Research/nightjar-private/benchmarks_gsm8k/")

    try:
        # Load already completed runs
        completed_runs = load_completed_runs(opts.output_file)
        if completed_runs:
            tqdm.write(f"Found {len(completed_runs)} already completed runs, will skip them")

        data = datasets.load_dataset("openai/gsm8k", "main", split="test")

        # Load GSM8k problems

        # Calculate total pending work (program x runs)
        pending_work: List[Tuple[str, float, int]] = []
        for run_index in range(opts.runs):
            for example in data:
                question = example["question"]  # type: ignore
                answer = example["answer"]  # type: ignore
                answer = float(answer.split("####")[1].strip().replace(",", ""))
                if (question, answer, run_index) not in completed_runs:
                    pending_work.append((question, answer, run_index))

        if not pending_work:
            tqdm.write("All runs are already complete.")
            return

        n_questions = len(data["question"])  # type:ignore
        total_work = n_questions * opts.runs  # type: ignore
        tqdm.write(f"Found {n_questions} programs x {opts.runs} runs = {total_work} total, {len(pending_work)} pending")

        # Initialize tracking variables for wandb logging
        programs_completed = 0
        total_pass_rate = 0.0
        all_pass_rates = []
        all_runtimes = []

        container_id = None

        if opts.handler_name == "manual":
            dest_path = os.path.join(programs_dir, f"gsm8k_manual.py")
        elif opts.handler_name == "codetool":
            dest_path = os.path.join(programs_dir, f"gsm8k_codetool.py")
            if opts.model.split("/")[0] == "openai" and opts.reuse_container:
                client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                container = client.containers.create(name="test-container")
                container_id = container.id
                tqdm.write(f"Using container {container_id}")
        elif opts.handler_name == "customtool":
            dest_path = os.path.join(programs_dir, f"gsm8k_customtool.py")
        else:
            dest_path = os.path.join(programs_dir, f"gsm8k.py")
            dest_path = add_config(opts.handler_name, dest_path, opts)

        # Process pending work
        if opts.parallel:
            # Parallel execution mode with batching
            tqdm.write(f"Running in parallel mode with {opts.num_workers} workers, batch size {opts.batch_size}")

            # Split pending work into batches
            num_batches = (len(pending_work) + opts.batch_size - 1) // opts.batch_size
            tqdm.write(f"Processing {len(pending_work)} examples in {num_batches} batches")

            global_example_idx = 0
            for batch_idx in tqdm(range(num_batches), desc="Processing batches", total=num_batches):
                batch_start = batch_idx * opts.batch_size
                batch_end = min((batch_idx + 1) * opts.batch_size, len(pending_work))
                batch_work = pending_work[batch_start:batch_end]

                tqdm.write(f"\n=== Processing batch {batch_idx + 1}/{num_batches} ({len(batch_work)} examples) ===")

                # Prepare arguments for this batch
                batch_args = [
                    (dest_path, run_index, question, answer, container_id, opts)
                    for question, answer, run_index in batch_work
                ]

                # Use ParallelExecutor to parallelize execution for this batch
                executor = ParallelExecutor(
                    num_threads=opts.num_workers,
                    disable_progress_bar=False,
                    max_errors=1000,
                    provide_traceback=True,
                    compare_results=False,
                    timeout=opts.timeout,
                )

                batch_results = executor.execute(run_program, batch_args)

                # Process results for this batch
                for (question, answer, run_index), res in zip(batch_work, batch_results):
                    res = cast(Optional[Tuple[TestResult, TraceEntry]], res)

                    # Handle execution failure
                    if res is None:
                        test_result = TestResult(
                            question=question,
                            model=opts.model,
                            run=run_index,
                            runtime=None,
                            compile_time=None,
                            result=None,
                            expected_result=answer,
                            eval_result=False,
                            errors={"execution": "Program failed to execute"},
                            token_count=None,
                            n_tool_calls=0,
                        )
                        trace_entry = TraceEntry(
                            question=question,
                            model=opts.model,
                            run=run_index,
                            trace=[],
                        )
                    else:
                        test_result, trace_entry = res

                    # Save results immediately
                    save_result(test_result, opts.output_file)
                    if trace_entry is not None and opts.trace_file is not None:
                        save_trace(trace_entry, opts.trace_file)

                    # Calculate pass rate
                    all_pass_rates.append(test_result.eval_result)
                    total_pass_rate = sum(all_pass_rates) / len(all_pass_rates)

                    # Track metrics
                    programs_completed += 1
                    if test_result.runtime is not None:
                        all_runtimes.append(test_result.runtime)

                    # Calculate averages
                    average_runtime = sum(all_runtimes) / len(all_runtimes) if all_runtimes else 0.0

                    # Log to wandb
                    wandb.log(
                        {
                            "programs_completed": programs_completed,
                            "current_pass_rate": int(test_result.eval_result),
                            "average_pass_rate": total_pass_rate,
                            "average_runtime": average_runtime,
                            "example_id": global_example_idx,
                            "question": question,
                            "run_index": run_index,
                            "runtime": test_result.runtime or 0,
                            "n_tool_calls": test_result.n_tool_calls,
                            "batch_idx": batch_idx,
                        }
                    )
                    global_example_idx += 1

                tqdm.write(
                    f"Batch {batch_idx + 1}/{num_batches} complete. "
                    f"Overall progress: {programs_completed}/{len(pending_work)} "
                    f"({100 * programs_completed / len(pending_work):.1f}%), "
                    f"Pass rate: {100 * total_pass_rate:.1f}%"
                )
        else:
            # Sequential execution mode (use ParallelExecutor for isolation only)
            for e_i, (question, answer, run_index) in tqdm(
                enumerate(pending_work), desc="Evaluating examples", total=len(pending_work)
            ):
                tqdm.write(f"\n--- Processing problem {e_i} (run {run_index}) ---")

                # Use ParallelExecutor to isolate program execution
                executor = ParallelExecutor(
                    num_threads=1,
                    disable_progress_bar=True,
                    max_errors=1000,
                    provide_traceback=True,
                    compare_results=False,
                    timeout=opts.timeout,
                )
                results = executor.execute(
                    run_program,
                    [
                        (
                            dest_path,
                            run_index,
                            question,
                            answer,
                            container_id,
                            opts,
                        )
                    ],
                )
                res = cast(Optional[Tuple[TestResult, TraceEntry]], results[0])

                # Handle execution failure
                if res is None:
                    test_result = TestResult(
                        question=question,
                        model=opts.model,
                        run=run_index,
                        runtime=None,
                        compile_time=None,
                        result=None,
                        expected_result=answer,
                        eval_result=False,
                        errors={"execution": "Program failed to execute"},
                        token_count=None,
                        n_tool_calls=0,
                    )
                    trace_entry = TraceEntry(
                        question=question,
                        model=opts.model,
                        run=run_index,
                        trace=[],
                    )
                else:
                    test_result, trace_entry = res

                # Save results
                save_result(test_result, opts.output_file)
                if trace_entry is not None and opts.trace_file is not None:
                    save_trace(trace_entry, opts.trace_file)

                # Calculate pass rate for this program
                all_pass_rates.append(test_result.eval_result)
                total_pass_rate = sum(all_pass_rates) / len(all_pass_rates)
                tqdm.write(f"Passed: {test_result.eval_result}")

                # Track metrics
                programs_completed += 1
                if test_result.runtime is not None:
                    all_runtimes.append(test_result.runtime)

                # Calculate averages
                average_runtime = sum(all_runtimes) / len(all_runtimes) if all_runtimes else 0.0

                # Log to wandb
                wandb.log(
                    {
                        "programs_completed": programs_completed,
                        "current_pass_rate": int(test_result.eval_result),
                        "average_pass_rate": total_pass_rate,
                        "average_runtime": average_runtime,
                        "example_id": e_i,
                        "question": question,
                        "run_index": run_index,
                        "runtime": test_result.runtime or 0,
                        "n_tool_calls": test_result.n_tool_calls,
                    }
                )

        # Log final summary to wandb
        wandb.log(
            {
                "final_programs_completed": programs_completed,
                "final_average_pass_rate": total_pass_rate,
                "final_average_runtime": sum(all_runtimes) / len(all_runtimes) if all_runtimes else 0.0,
                "total_runs": len(pending_work),
            }
        )

        tqdm.write(f"\nEvaluation complete. Results saved to {opts.output_file}")

        # Create an artifact object
        artifact_name = f"gsm8k-{opts.model}-{opts.handler_name}-mtc{opts.max_tool_calls}".replace("/", "_").replace(
            ":", "_"
        )
        artifact_name += "-caching" if opts.caching else ""
        artifact_name += "-optimize" if opts.optimize else ""

        artifact = wandb.Artifact(name=artifact_name, type="results")
        artifact.add_file(local_path=opts.output_file)
        if opts.trace_file and os.path.exists(opts.trace_file):
            artifact.add_file(local_path=opts.trace_file)

        wandb.log_artifact(artifact)

    except KeyboardInterrupt:
        tqdm.write("\nInterrupted by user")

    finally:
        # Always finish wandb run, even on interrupt
        wandb.finish()


if __name__ == "__main__":
    opts = Options().parse_args()
    main(opts)

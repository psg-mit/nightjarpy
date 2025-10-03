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
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypeAlias, cast

import func_timeout
from dspy.utils.parallelizer import ParallelExecutor
from func_timeout import func_set_timeout
from pydantic import BaseModel
from tap import Tap
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import wandb
from nightjarpy import NJ_TELEMETRY
from nightjarpy.configs import (
    INTERPRETER_BASE_NOREG_JSON_CONFIG,
    INTERPRETER_PYTHON_BASE_ISOLATED_NOREG_JSON_CONFIG,
    INTERPRETER_PYTHON_BASE_NOREG_JSON_CONFIG,
    INTERPRETER_PYTHON_CACHE_JSON_CONFIG,
    INTERPRETER_PYTHON_EAGER_CACHE_JSON_CONFIG,
    INTERPRETER_PYTHON_JSON_CONFIG,
    LLMConfig,
)
from nightjarpy.runtime.run import nj_llm_factory
from nightjarpy.types import ChatMessage, LLMUsage
from nightjarpy.utils import NJ_CACHE, sum_usage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HANDLERS: TypeAlias = Literal[
    "manual",
    "manual_code",
    "compiler_base",
    "interpreter_python_json",
    "interpreter_python_cache_json",
    "interpreter_python_eager_cache_json",
    "interpreter_base_noreg_json",
    "interpreter_python_base_isolated_json",
    "interpreter_python_base_json",
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
    """
    Options for the benchmark evaluation test harness.
    """

    benchmark_dir: str = "benchmarks/programs"  # Path to the benchmarks directory
    output_file: str = "evaluation_results.jsonl"  # File to save evaluation results (JSON Lines)
    trace_file: Optional[str] = None  # File to dump execution trace (JSON Lines)
    runs: int = 1  # Number of times to run each program
    timeout: int = 300  # Timeout in seconds for each program run
    model: str = "openai/gpt-4.1"  # Model name to use for natural language
    max_tool_calls: int = 100  # Maximum number of tool calls to make
    handler_name: HANDLERS = "interpreter_python_json"  # Handler to use for natural language
    temperature: float = 1.0
    wandb_project: str = "nightjar"  # Wandb project name
    wandb_run_name: Optional[str] = None  # Wandb run name (auto-generated if None)


def load_completed_runs(results_file: str):
    """
    Reads the results file if it exists and returns a set of (program, run) tuples
    that have already been completed.
    """
    completed = set()
    existing_results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                for line in f:
                    try:
                        entry: dict = json.loads(line)
                        key = (entry.get("file_name"), entry.get("run"))
                        completed.add(key)
                        # Store existing result for potential merging
                        existing_results[key] = entry
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            tqdm.write(f"Failed to load existing results: {e}", file=sys.stderr)
    else:
        dirpath, filename = os.path.split(results_file)
        os.makedirs(dirpath, exist_ok=True)
    return completed, existing_results


def calculate_pass_rate(hard_eval: Dict[str, bool]) -> float:
    """
    Calculate the pass rate from hard_eval results.
    Returns the ratio of True values to total items.
    """
    if not hard_eval:
        return 0.0

    true_count = sum(1 for value in hard_eval.values() if value is True)
    total_count = len(hard_eval)

    return true_count / total_count if total_count > 0 else 0.0


def load_namespace_from_path(program_path: str):
    """
    Load file as a namespace
    """
    globals_: Dict[str, Any] = {}
    try:
        with logging_redirect_tqdm():
            program_name = os.path.basename(program_path)
            namespace = runpy.run_path(program_path, globals_)
        return namespace
    except Exception as e:
        tqdm.write(f"Failed to load {program_path}: {e}", file=sys.stderr)
        raise ValueError("Failed to load")


class TraceEntry(BaseModel):
    file_name: str
    model: str
    run: int  # Run number
    trace: List[ChatMessage]


class TestResult(BaseModel):
    file_name: str
    model: str
    run: int  # Run number
    compile_time: Optional[float]
    runtime: Optional[float]
    hard_eval: Dict[str, bool]  # Individual test results
    errors: Dict[str, Any]
    token_count: Optional[Dict[str, LLMUsage]]
    n_tool_calls: int


def save_trace(result: TraceEntry, output_file: str):
    """Save a single result to the output file in JSONL format."""
    trace_data = result.model_dump()
    trace_data["trace"] = [x.model_dump() for x in result.trace]
    tqdm.write(f"saving trace...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "a") as f:
        f.write(json.dumps(trace_data) + "\n")
        f.flush()


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

    result_data = result.model_dump()
    result_data["errors"] = errors_data

    tqdm.write(f"saving result...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "a") as f:
        f.write(json.dumps(result_data) + "\n")
        f.flush()


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
    config = config.with_llm_updates(model=opts.model, temperature=opts.temperature)
    config_json = config.model_dump_json()

    # Use AST to modify the decorators
    src_code = modify_nj_fn_decorators(src_code, config_json)

    # Write the modified source code to a temporary file and return its path
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(src_code)
        tmp_file.flush()
        temp_path = tmp_file.name

    return temp_path


def run_program(args) -> Tuple[TestResult, TraceEntry]:
    program_dir, program_name, run_index, opts = args
    opts: Options

    NJ_CACHE.clear_cache()
    NJ_TELEMETRY.reset()

    if opts.handler_name in ["manual", "manual_code"]:
        filename = f"{program_name}_manual.py"
        path = os.path.join(program_dir, filename)
    else:
        filename = f"{program_name}.py"
        path = os.path.join(program_dir, filename)
        # Preprocess file.
        path = add_config(opts.handler_name, path, opts)

    if not os.path.exists(path):
        tqdm.write(
            f"File not found: {path}",
            file=sys.stderr,
        )
        result = TestResult(
            file_name=program_name,
            model=opts.model,
            run=run_index,
            runtime=None,
            hard_eval={},
            errors={"error": "File not found"},
            token_count=None,
            n_tool_calls=0,
            compile_time=None,
        )
        entry = TraceEntry(
            file_name=program_name,
            model=opts.model,
            run=run_index,
            trace=[],
        )
        return result, entry

    # Load file, which triggers compilation
    try:
        t1 = time.time()
        namespace = load_namespace_from_path(path)
        t2 = time.time()
    except Exception as e:
        tqdm.write(f"Compilation failed: {e}", file=sys.stderr)
        result = TestResult(
            file_name=program_name,
            model=opts.model,
            run=run_index,
            runtime=None,
            hard_eval={},
            errors={"compilation": "Compilation failed"},
            token_count=None,
            n_tool_calls=0,
            compile_time=None,
        )
        entry = TraceEntry(
            file_name=program_name,
            model=opts.model,
            run=run_index,
            trace=[],
        )
        return result, entry

    compile_time = t2 - t1
    compile_usage = NJ_TELEMETRY.total_llm_usage()
    NJ_TELEMETRY.reset()

    @func_set_timeout(opts.timeout)
    def run_program_with_timeout(func: Callable):
        config = LLMConfig(model=opts.model, container=(opts.handler_name == "manual_code"))
        nj_llm = nj_llm_factory(
            config,
            filename=filename,
            funcname="main",
            max_calls=opts.max_tool_calls,
            code_interpreter=(opts.handler_name == "manual_code"),
        )
        # Run the program
        start_time = time.time()

        # Run the program
        try:
            with logging_redirect_tqdm():
                if opts.handler_name in ["manual", "manual_code"]:
                    outputs, errors, hard_results = func(nj_llm)
                else:
                    outputs, errors, hard_results = func()
        except Exception as e:
            runtime = time.time() - start_time
            return {}, {"run": e}, runtime

        runtime = time.time() - start_time
        return hard_results, errors, runtime

    # Execute the run function to get input/output pairs
    try:
        tqdm.write(f"running program...")
        hard_results, errors, runtime = run_program_with_timeout(namespace["run"])
        tqdm.write(f"results: {hard_results}")
        tqdm.write(f"program finished")
    except func_timeout.exceptions.FunctionTimedOut as e:
        tqdm.write("Run function timed out", file=sys.stderr)
        result = TestResult(
            file_name=program_name,
            model=opts.model,
            run=run_index,
            runtime=opts.timeout,
            hard_eval={},
            errors={"run": e},
            token_count={"compile": compile_usage},
            n_tool_calls=0,
            compile_time=compile_time,
        )
        entry = TraceEntry(
            file_name=program_name,
            model=opts.model,
            run=run_index,
            trace=[],
        )
        return result, entry

    runtime_usage = NJ_TELEMETRY.total_llm_usage()
    n_tool_calls = NJ_TELEMETRY.n_tool_calls
    traces = deepcopy(NJ_TELEMETRY.trace) or []
    NJ_TELEMETRY.reset()

    entry = TraceEntry(
        file_name=program_name,
        model=opts.model,
        run=run_index,
        trace=traces,
    )

    result = TestResult(
        file_name=program_name,
        model=opts.model,
        run=run_index,
        runtime=runtime,
        hard_eval=hard_results,
        errors=errors,
        token_count={
            "compile": compile_usage,
            "runtime": runtime_usage,
        },
        n_tool_calls=n_tool_calls,
        compile_time=compile_time,
    )
    return result, entry


def main(opts: Options):
    # Initialize wandb if enabled
    run_name = (
        opts.wandb_run_name or f"interop-{opts.model}-{opts.handler_name}-mtc{opts.max_tool_calls}-{int(time.time())}"
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
        },
    )

    try:
        # Load already completed runs
        completed_runs, existing_results = load_completed_runs(opts.output_file)

        # List all subdirectories in the specified benchmarks directory
        try:
            program_dirs = [
                os.path.join(opts.benchmark_dir, d)
                for d in os.listdir(opts.benchmark_dir)
                if os.path.isdir(os.path.join(opts.benchmark_dir, d))
            ]
        except Exception as e:
            tqdm.write(f"Failed to list benchmark directories: {e}", file=sys.stderr)
            sys.exit(1)

        # Sort program directories to ensure consistent order
        program_dirs.sort()

        # Calculate total pending runs and create a list of work to do
        pending_work = []
        for run_index in range(opts.runs):
            for program_dir in program_dirs:
                program_name = os.path.basename(program_dir)
                if (program_name, run_index) not in completed_runs:
                    pending_work.append((program_dir, program_name, run_index))

        if not pending_work:
            tqdm.write("All runs are already complete.")
            return

        # Initialize tracking variables for wandb logging
        programs_completed = 0
        pass_rate = None
        total_pass_rate = 0.0
        all_hard_evals = []
        all_runtimes = []

        pbar = tqdm(total=len(pending_work), desc="Processing Runs")

        for program_dir, program_name, run_index in pending_work:
            tqdm.write(f"\n--- Processing {program_name} ---")

            executor = ParallelExecutor(
                num_threads=1,
                disable_progress_bar=True,
                max_errors=1000,
                provide_traceback=True,
                compare_results=False,
                timeout=opts.timeout,
            )
            results = cast(
                List[Tuple[TestResult, TraceEntry]],
                executor.execute(
                    run_program,
                    [
                        (
                            program_dir,
                            program_name,
                            run_index,
                            opts,
                        )
                    ],
                ),
            )
            result, trace_entry = results[0]
            if result is None:
                result = TestResult(
                    file_name=program_name,
                    model=opts.model,
                    run=run_index,
                    runtime=None,
                    hard_eval={},
                    errors={"run": "Program failed to run"},
                    token_count=None,
                    n_tool_calls=0,
                    compile_time=None,
                )
            if trace_entry is None:
                trace_entry = TraceEntry(
                    file_name=program_name,
                    model=opts.model,
                    run=run_index,
                    trace=[],
                )

            # Save result immediately
            save_result(result, opts.output_file)
            if opts.trace_file is not None:
                save_trace(trace_entry, opts.trace_file)

            # Update tracking variables and log to wandb
            programs_completed += 1
            if result.hard_eval:
                pass_rate = calculate_pass_rate(result.hard_eval)
                all_hard_evals.append(result.hard_eval)
                total_pass_rate = sum(calculate_pass_rate(eval_dict) for eval_dict in all_hard_evals) / len(
                    all_hard_evals
                )

            all_runtimes.append(result.runtime or 0)
            average_runtime = sum(all_runtimes) / len(all_runtimes) if all_runtimes else 0.0

            wandb.log(
                {
                    "programs_completed": programs_completed,
                    "current_pass_rate": pass_rate,
                    "average_pass_rate": total_pass_rate,
                    "average_runtime": average_runtime,
                    "program_name": program_name,
                    "run_index": run_index,
                    "runtime": result.runtime or 0,
                    "n_tool_calls": result.n_tool_calls,
                    "compile_time": result.compile_time or 0,
                }
            )

            pbar.update(1)

        pbar.close()

        # Log final summary to wandb
        wandb.log(
            {
                "final_programs_completed": programs_completed,
                "final_average_pass_rate": total_pass_rate,
                "final_average_runtime": average_runtime,
                "total_programs": len(pending_work),
            }
        )

        tqdm.write(f"\nEvaluation complete. Results saved to {opts.output_file}")
        tqdm.write(f"Total programs completed: {programs_completed}")
        tqdm.write(f"Final average pass rate: {total_pass_rate:.3f}")
        tqdm.write(f"Final average runtime: {average_runtime:.3f}s")
    except KeyboardInterrupt:
        tqdm.write("\nInterrupted by user")
    finally:
        # Always finish wandb run, even on interrupt
        wandb.finish()


if __name__ == "__main__":
    opts = Options().parse_args()
    main(opts)

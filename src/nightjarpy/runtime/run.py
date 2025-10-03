import inspect
import json
import logging
import os
import random
import textwrap
import time
import types
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, cast

import dotenv
import shortuuid
from dspy.utils.parallelizer import ParallelExecutor
from pydantic import BaseModel

from nightjarpy.configs import (
    CompilerConfig,
    Config,
    ExecutionSubstrate,
    InterpreterConfig,
    LLMConfig,
)
from nightjarpy.context import Context
from nightjarpy.effects import Done, Effect
from nightjarpy.llm.factory import create_llm
from nightjarpy.types import (
    SUCCESS,
    AssistantMessage,
    ChatMessage,
    EffectCall,
    EffectError,
    EffectException,
    Label,
    Ref,
    ResponseFormat,
    Success,
    ToolCall,
    ToolMessage,
    UserMessage,
    Value,
    Variable,
    is_function,
)
from nightjarpy.utils import (
    NJ_TELEMETRY,
    LLMUsage,
    deserialize_json,
    enable_nj_logging,
    extract_label,
    extract_variable,
    parse_effect,
    serialize,
    serialize_json,
)
from nightjarpy.utils.utils import (
    MAX_SERIALIZE_LEN,
    extract_effects,
    parallelize_effects,
    serialize_effect,
    validate_args,
    validate_kwargs,
)

logger = logging.getLogger(__name__)

dotenv.load_dotenv()
TRACE_PATH = os.getenv("NJ_TRACE", None)


def _log_telemetry_and_trace(
    all_emitted_effects: List[ToolCall],
    emitted_effects_str: Optional[str],
    handled_effect_ids: List[str],
    handled_effects_results: List[ToolMessage],
    effect_gen_time: float,
    usage: Optional[LLMUsage],
    log_trace: bool = True,
) -> None:
    """Log telemetry messages and optionally dump trace."""
    handled_effects = [x for x in all_emitted_effects if x.id in handled_effect_ids]
    discarded_effects = [x for x in all_emitted_effects if x.id not in handled_effect_ids]
    NJ_TELEMETRY.log_messages(
        [
            AssistantMessage(
                tool_calls=handled_effects,
                discarded_tool_calls=discarded_effects,
                content=emitted_effects_str,
                time=effect_gen_time,
                usage=usage,
            )
        ]
        + handled_effects_results
    )
    if log_trace and TRACE_PATH:
        NJ_TELEMETRY.dump_trace(TRACE_PATH)


def _handle_effect(
    inp: Tuple[Context, ToolCall, bool, Dict[str, Effect]],
) -> Tuple[Optional[str], ToolCall, Optional[Value | Success | EffectError], float, Optional[Exception]]:
    context, effect, json_structured_output, effect_mapping = inp
    logger.info(f"Handling effect: {effect}")

    try:
        t1 = time.time()

        if not json_structured_output:
            # effect.name holds the entire string before parsing
            effect_str = effect.name

            parsed_effect = parse_effect(effect.name, effect_mapping=effect_mapping)
            kwargs = {k: v for k, v in parsed_effect.args}
            # Serialize in json format for storage
            effect.args = {k: json.loads(serialize_json(v)) for k, v in parsed_effect.args}
            effect.name = parsed_effect.name
        else:
            effect_str = None
            deserialized_kwargs = {k: deserialize_json(v, context) for k, v in effect.args.items()}
            kwargs = validate_kwargs(deserialized_kwargs, parameters=effect_mapping[effect.name].parameters)

        if effect.name not in effect_mapping:
            raise ValueError(f"Effect {effect.name} not found")

        logger.info(f"Arguments: {kwargs}")

        res: Value | Success | EffectError
        res = effect_mapping[effect.name](context=context, **kwargs)
        t2 = time.time()

        return effect_str, effect, res, t2 - t1, None
    except (Done, EffectException, Exception) as e:
        t2 = time.time()

        return effect_str, effect, None, t2 - t1, e


def execute(
    natural_code: str,
    python_frame: Optional[types.FrameType],
    config: Config,
    filename: str,
    funcname: str,
    valid_labels: Optional[set[str]],
) -> Dict[str, Any]:
    """
    Execute a natural language code block.

    This method takes natural language code and executes it using the configured
    LLM agent, providing access to the current execution context.

    Args:
        natural_code: str
            The natural language code block to execute.
        python_frame: Optional[types.FrameType]
            The current Python frame, used for variable context.
        config: Config
            Configuration object containing LLM and interpreter settings.
        filename: str
            The name of the file being executed.
        funcname: str
            The name of the function being executed.
        valid_labels: Optional[List[str]]
            List of available program labels (e.g., "break", "continue", "return").

    Returns:
        Dictionary of variable names to Python values

    Raises:
        RuntimeError: If execution fails or exceeds tool call limits
    """

    llm_model = create_llm(config.llm_config)
    interpreter_config = config.interpreter_config
    assert isinstance(interpreter_config, InterpreterConfig)

    if interpreter_config.seed is not None:
        random.seed(interpreter_config.seed)

    nonce = str(random.randint(0, 1000000)).zfill(7)
    nonce = f"Nonce:{nonce}\n" if interpreter_config.use_nonce else ""

    system_prompt_kwargs = {}
    system_prompt_template = interpreter_config.prompt_template.system
    if "{nonce}" in system_prompt_template:
        system_prompt_kwargs["nonce"] = nonce
    if "{max_tool_calls}" in system_prompt_template:
        system_prompt_kwargs["max_tool_calls"] = interpreter_config.max_effects
    if "{steps_ahead}" in system_prompt_template:
        system_prompt_kwargs["steps_ahead"] = interpreter_config.steps_ahead

    filled_in_system_prompt = interpreter_config.prompt_template.system.format(**system_prompt_kwargs)

    effect_set = ExecutionSubstrate.get_effect_set(interpreter_config.execution_substrate)

    effect_set.set_use_functions(config.use_functions)

    serialize_fun = serialize_json if config.llm_config.json_structured_output else serialize

    effect_mapping: Dict[str, Effect] = {e.name: e for e in effect_set.effects}

    natural_code = textwrap.dedent(natural_code).strip()
    output_vars, input_vars = extract_variable(natural_code)
    valid_vars = input_vars | output_vars.keys()

    if valid_labels is None:
        # Doesn't take labels in code, so use what's given by context
        valid_labels_parsed = extract_label(natural_code)
    else:
        valid_labels_parsed = set(Label(l) for l in valid_labels)

    valid_labels_parsed.add(Label("raise"))

    context = Context(
        temp_var_init=interpreter_config.var_generator_init,
        valid_vars=valid_vars,
        output_vars=set(output_vars.keys()),
        valid_labels=set(valid_labels_parsed) if valid_labels_parsed else set(),
        python_frame=python_frame,
        llm_config=config.llm_config,
        compute_prompt_template=interpreter_config.compute_prompt_template,
        use_functions=config.use_functions,
    )

    logger.info(f"input vars: {input_vars}")
    logger.info(f"output vars: {output_vars}")
    logger.info(f"labels: {valid_labels_parsed}")

    # Lookup input vars
    # Eagerly load input vars into context in the current frame in the Context handler
    # need this so deletion can find the variable in the current frame
    for var in input_vars:
        # Store into FL context for interface format
        try:
            val = context.lookup(var)
            # assign checks <:output_var> notation
            context.current_frame.insert(var, val)
        except ValueError:
            # if not in the context but in the output vars, that's ok
            if var not in output_vars:
                raise ValueError(f"Undefined variable `{var}`")

    content = natural_code

    if interpreter_config.eager_load:
        eager_load_str = ""
        if len(input_vars) > 0:
            eager_load_str += "\nType and Value (or Reference to Value) of Input Variables:\n"
        for var in input_vars:
            try:
                val = context.lookup(var)
            except ValueError:
                # if not in the context but in the output vars, that's ok
                if var not in output_vars:
                    raise ValueError(f"Undefined variable `{var}`")
            else:
                if not isinstance(val, Ref):
                    val_str = serialize(val)  # type: ignore
                    ty_str = type(val).__qualname__

                    if len(val_str) > MAX_SERIALIZE_LEN:
                        val_str = val_str[:MAX_SERIALIZE_LEN] + "[TRUNCATED FOR LENGTH]"
                    val_str = f"Value: {val_str}"
                    eager_load_str += f"{var} [type: {ty_str}]: {val_str}"
                else:
                    ty_str = "Obj"
                    val_str = f"Ref({val.addr})"
                    py_val = context.decode_and_sync_python_value(val, {})

                    attrs = dir(py_val)
                    attrs = filter(lambda x: not x.startswith("__"), attrs)
                    attrs = str(sorted(list([x for x in attrs])))
                    if len(attrs) > MAX_SERIALIZE_LEN:
                        attrs = attrs[:MAX_SERIALIZE_LEN] + "[TRUNCATED FOR LENGTH]"

                    eager_load_str += f"{var} [type: {ty_str}]: {val_str}"
                    eager_load_str += f" Attrs: {attrs}"

                py_val = context.decode_and_sync_python_value(val, {})
                if is_function(py_val):
                    try:
                        parameters = list(inspect.signature(py_val).parameters.values())
                    except:
                        parameters = None
                    eager_load_str += f" Signature: {parameters}"

                eager_load_str += "\n"
        content += eager_load_str

    if interpreter_config.show_effect_count:
        if interpreter_config.max_effects_per_execute is not None:
            content += f"\nNumber of tool calls left: {interpreter_config.max_effects_per_execute-1}\n"
        else:
            content += f"\nNumber of tool calls left: {interpreter_config.max_effects - NJ_TELEMETRY.n_tool_calls-1}\n"
        if interpreter_config.max_iters is not None:
            content += f"\nNumber of iterations left: {interpreter_config.max_iters-1}"

    logger.info(f"\n======== Natural Code ========\n{content}\n=================")

    messages: List[ChatMessage] = [UserMessage(content=content)]
    NJ_TELEMETRY.log_messages(messages)
    n_effects = 0
    max_iters = interpreter_config.max_iters or interpreter_config.max_effects

    for n_iter in range(max_iters):

        if interpreter_config.max_effects_per_execute is None:
            logger.info(f"n effects: {NJ_TELEMETRY.n_tool_calls}")
            if NJ_TELEMETRY.n_tool_calls >= interpreter_config.max_effects:
                if TRACE_PATH:
                    NJ_TELEMETRY.dump_trace(TRACE_PATH)
                raise TimeoutError(f"Max effects reached: {NJ_TELEMETRY.n_tool_calls}")
        else:
            logger.info(f"n effects: {n_effects}")
            if n_effects >= interpreter_config.max_effects_per_execute:
                if TRACE_PATH:
                    NJ_TELEMETRY.dump_trace(TRACE_PATH)
                raise TimeoutError(f"Max effects reached: {n_effects}")

        if config.llm_config.json_structured_output:
            emitted_effects_str = None
            t1 = time.time()
            all_emitted_effects = llm_model.gen_tool_calls(
                messages=messages,
                tools=list(effect_set.effects),
                tool_choice="required",
                system=filled_in_system_prompt,
                parallel_tool_calls=interpreter_config.jit,
            )
            t2 = time.time()
            logger.info(f"Generated Effects: {all_emitted_effects}")
        else:
            t1 = time.time()
            emitted_effects_str = llm_model.gen(
                messages=messages,
                system=filled_in_system_prompt,
            )
            t2 = time.time()
            logger.info(f"Raw generated effect str:\n{textwrap.indent(emitted_effects_str,prefix='  - ')}")
            # We want to lazily parse effects so that the LLM only needs to regen the malformed one, and the working effects before it can execute.
            # So we'll save the whole line as the name, and then parse it later
            all_emitted_effects = extract_effects(raw_str=emitted_effects_str)

        effect_gen_time = t2 - t1

        usage = llm_model.get_usage()
        if usage is None:
            logger.warning("No LLM usage to log")
        else:
            NJ_TELEMETRY.log_llm_usage(filename=filename, funcname=funcname, usage=usage)

        if len(all_emitted_effects) == 0:
            messages.extend(
                [
                    AssistantMessage(
                        content=emitted_effects_str or "",
                    ),
                    UserMessage(
                        content="Did not parse any tool calls. Please make sure all tool calls are in the expected output format"
                    ),
                ]
            )
            continue

        # if not not json structured output, need to track the parsed effects to return to LLM
        # all_emitted_effects is what is stored
        handled_effects: List[str] = []
        # This one is for the storage
        handled_effects_results_log: List[ToolMessage] = []
        # This one is for llm
        handled_effects_results: List[ToolMessage | UserMessage] = []
        handled_effects_ids: List[str] = []

        if interpreter_config.discard_steps_ahead:
            # Throw away more than n_steps ahead
            logger.info(f"Throwing away {max(0, len(all_emitted_effects) - interpreter_config.steps_ahead)} effects...")
            candidate_effects = all_emitted_effects[: interpreter_config.steps_ahead]
        else:
            candidate_effects = all_emitted_effects

        if config.llm_config.parallelize_effects:
            batched_emitted_effects = parallelize_effects(candidate_effects, effect_mapping=effect_mapping)
        else:
            batched_emitted_effects = [[x] for x in candidate_effects]

        logger.info(
            f"Effect batches:\n{'\n\n'.join(['\n'.join([textwrap.indent(xx.name + (str(xx.args) if len(xx.args) > 0 else ""), '  - ') for xx in x]) for x in batched_emitted_effects])}"
        )

        def _add_response(response: str, tool_id: str, t: float):
            # if len(response) > MAX_SERIALIZE_LEN:
            #     response = response[:MAX_SERIALIZE_LEN] + "[TRUNCATED]"

            if config.llm_config.json_structured_output:
                handled_effects_results.append(ToolMessage(content=response, tool_call_id=tool_id, time=t))
            else:
                handled_effects_results.append(UserMessage(content=response, time=t))

            handled_effects_results_log.append(ToolMessage(content=response, tool_call_id=tool_id, time=t))

        encountered_error = False

        n_effects_this_iter = 0
        # Processes effects in order until one of them jumps out
        for batch_i in range(len(batched_emitted_effects)):

            if config.llm_config.parallelize_effects:
                parallel_inputs = [
                    (
                        context,
                        effect,
                        config.llm_config.json_structured_output,
                        effect_mapping,
                    )
                    for effect in batched_emitted_effects[batch_i]
                ]
                executor = ParallelExecutor(
                    num_threads=len(parallel_inputs),
                    disable_progress_bar=True,
                    max_errors=0,
                    provide_traceback=True,
                    compare_results=False,
                    timeout=60,
                )

                results = executor.execute(_handle_effect, parallel_inputs)
            else:
                assert (
                    len(batched_emitted_effects[batch_i]) == 1
                ), "Parallel execution of effects is DISABLED, but received multiple effects"
                results = [
                    _handle_effect(
                        (
                            context,
                            batched_emitted_effects[batch_i][0],
                            config.llm_config.json_structured_output,
                            effect_mapping,
                        )
                    )
                ]

            # Post-process results
            for effect_i, res in enumerate(results):
                n_effects_this_iter += 1
                n_effects += 1
                NJ_TELEMETRY.n_tool_calls += 1

                if res is None:
                    error = ValueError("Failed to execute effect")
                    updated_effect = batched_emitted_effects[batch_i][effect_i]
                    handled_effects.append(updated_effect.name)
                    effect_res = None
                    effect_time = -1
                else:
                    effect_str, updated_effect, effect_res, effect_time, error = res

                    batched_emitted_effects[batch_i][effect_i] = updated_effect

                    if effect_str is not None:
                        handled_effects.append(effect_str)
                handled_effects_ids.append(updated_effect.id)

                if isinstance(error, Done):
                    logger.info(f"Exiting natural interpreter...")
                    _add_response(serialize_fun(SUCCESS), updated_effect.id, effect_time)
                    _log_telemetry_and_trace(
                        all_emitted_effects=all_emitted_effects,
                        emitted_effects_str=emitted_effects_str,
                        handled_effect_ids=handled_effects_ids,
                        effect_gen_time=effect_gen_time,
                        handled_effects_results=handled_effects_results_log,
                        usage=usage,
                        log_trace=True,
                    )
                    return error.outputs
                elif isinstance(error, EffectException):
                    logger.info(f"Handling abortive effect {error}")
                    _add_response(serialize_fun(SUCCESS), updated_effect.id, effect_time)
                    _log_telemetry_and_trace(
                        all_emitted_effects=all_emitted_effects,
                        emitted_effects_str=emitted_effects_str,
                        handled_effect_ids=handled_effects_ids,
                        effect_gen_time=effect_gen_time,
                        handled_effects_results=handled_effects_results_log,
                        usage=usage,
                        log_trace=True,
                    )
                    raise error
                elif isinstance(error, Exception):
                    effect_res = EffectError(f"Error during tool call: {error}")
                    logger.info(f"Encountered error {effect_res}... Continuing")
                    encountered_error = True

                logger.info(f"Effect resume with: {effect_res}")
                res_str = serialize_fun(effect_res)

                if interpreter_config.show_effect_count:
                    if interpreter_config.max_effects_per_execute is not None:
                        res_str += f"\nNumber of tool calls left: {interpreter_config.max_effects_per_execute - n_effects - 1}\n"
                    else:
                        res_str += f"\nNumber of tool calls left: {interpreter_config.max_effects - NJ_TELEMETRY.n_tool_calls - 1}\n"
                    if interpreter_config.max_iters is not None:
                        res_str += f"\nNumber of iterations left: {max_iters - n_iter - 1}"

                _add_response(res_str, updated_effect.id, effect_time)

            if encountered_error:
                break

        if not config.llm_config.json_structured_output:
            res_message = AssistantMessage(
                content="\n".join(handled_effects),
            )
        else:
            res_message = AssistantMessage(
                tool_calls=all_emitted_effects[:n_effects_this_iter],
            )
        messages.extend([res_message] + handled_effects_results)

        # Logging reads the json one, and stores the original generated string
        _log_telemetry_and_trace(
            all_emitted_effects=all_emitted_effects,
            emitted_effects_str=emitted_effects_str,
            handled_effect_ids=handled_effects_ids,
            handled_effects_results=handled_effects_results_log,
            effect_gen_time=effect_gen_time,
            usage=usage,
            log_trace=False,
        )

    if TRACE_PATH:
        NJ_TELEMETRY.dump_trace(TRACE_PATH)
    raise TimeoutError(f"Max iters reached: {n_iter + 1}")


T = TypeVar("T", bound=BaseModel)


def nj_llm_factory(
    config: LLMConfig, filename: str, funcname: str, max_calls: Optional[int] = None, code_interpreter: bool = False
):
    verbose = os.getenv("NJ_VERBOSE", False)
    if verbose:
        enable_nj_logging()

    n_calls = 0

    def nj_llm(prompt: str, output_format: Optional[Dict | type[T]] = None) -> str | Dict | T:
        nonlocal n_calls

        if max_calls is not None and n_calls >= max_calls:
            raise TimeoutError("Max LLM calls reached")

        schema = ResponseFormat(output_format) if output_format is not None else None

        llm_model = create_llm(config)

        logger.info(f"LLM call {n_calls}")

        if code_interpreter:
            res = llm_model.gen_code_interpreter(
                message=prompt,
                schema=schema,
                max_tool_calls=max_calls,
            )
        else:
            res = llm_model.gen([UserMessage(content=prompt)], schema=schema)

        usage = llm_model.get_usage()
        if usage is None:
            logger.warning("No LLM usage to log")
        else:
            NJ_TELEMETRY.log_llm_usage(filename, funcname, usage)

        n_calls += 1

        if res is None:
            raise ValueError("No response from model")

        return res

    return nj_llm

import ast
import logging
import random
import re
import textwrap
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, cast

import shortuuid
from pydantic import BaseModel, Field

from nightjarpy import __name__ as LIBNAME
from nightjarpy.compilation.utils import (
    LANGUAGE_IDENTIFIER,
    TRUNCATE_END,
    TRUNCATE_START,
)
from nightjarpy.configs import (
    CompilerConfig,
    Config,
    ExecutionStrategy,
    ExecutionSubstrate,
    InterpreterConfig,
)
from nightjarpy.llm.factory import create_llm
from nightjarpy.types import (
    NJ_VAR_PREFIX,
    EffectException,
    Label,
    ResponseFormat,
    ToolCall,
    UserMessage,
)
from nightjarpy.utils import NJ_TELEMETRY, VarGenerator, extract_variable
from nightjarpy.utils.utils import extract_effects, extract_label

logger = logging.getLogger(__name__)


class Compiler(ABC):
    """Base class for Nightjar compilers."""

    def __init__(self, config: Config, src_code: str, filename: str, funcname: str):
        self.config: Config = config
        self.variable_generator = VarGenerator()
        self.src_code = src_code
        self.filename = filename
        self.funcname = funcname

    @abstractmethod
    def compile_natural_block(self, natural_code: ast.Expr, kwargs: Dict[str, Any]) -> ast.AST:
        """
        Compile a natural language block.

        Args:
            natural_code: The natural language instruction code

        Returns:
            Compiled code as an AST expression
        """
        ...

    def add_preamble(self, program: ast.Module) -> ast.Module:
        # Add imports
        program.body = [
            ast.Import(names=[ast.alias(LIBNAME)]),
            ast.Import(names=[ast.alias("typing")]),
            ast.Import(names=[ast.alias("json")]),
            ast.Import(names=[ast.alias("inspect")]),
        ] + program.body

        return program


class AOTCompiler(Compiler):
    """Ahead-of-time compiler. Compiles natural code into Python code."""

    def __init__(self, config: Config, src_code: str, filename: str, funcname: str):
        super().__init__(config, src_code, filename, funcname)
        if config.compiler_config is None:
            raise ValueError("Compiler config must be provided in compiler mode")
        self.compiler_config = config.compiler_config
        self.llm = create_llm(config.llm_config)

    def add_preamble(self, program: ast.Module) -> ast.Module:
        if self.compiler_config.execution_substrate == ExecutionSubstrate.PYTHON:
            program.body = [
                ast.Assign(
                    targets=[ast.Name(id="nj_llm", ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=LIBNAME, ctx=ast.Load()), attr="nj_llm_factory", ctx=ast.Load()
                        ),
                        keywords=[
                            ast.keyword(
                                "config",
                                ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Attribute(
                                            value=ast.Name(id=LIBNAME, ctx=ast.Load()),
                                            attr="LLMConfig",
                                            ctx=ast.Load(),
                                        ),
                                        attr="model_validate_json",
                                        ctx=ast.Load(),
                                    ),
                                    args=[ast.Constant(value=self.config.llm_config.model_dump_json())],
                                    keywords=[],
                                ),
                            ),
                            ast.keyword("filename", ast.Constant(value=self.filename)),
                            ast.keyword("funcname", ast.Constant(value=self.funcname)),
                            # ast.keyword("max_calls", ast.Constant(value=self.compiler_config.max_runtime_calls)),
                        ],
                    ),
                )
            ] + program.body
        else:
            raise NotImplementedError()
        program = super().add_preamble(program)

        return program

    def compile_natural_block(self, node: ast.Expr, kwargs: Dict[str, Any]) -> ast.AST:
        """
        Compile natural language blocks ahead of time

        Args:
            natural_code: The natural language instruction code

        Returns:
            Compiled AST expression
        """

        if self.compiler_config.prompt_template is None:
            raise RuntimeError("LLM prompt template is required for compilation")

        if self.compiler_config.prompt_template.user is None:
            raise RuntimeError("User message prompt template is required for compilation")

        random.seed()
        nonce = str(random.randint(0, 1000000)).zfill(7)
        nonce = f"Nonce:{nonce}\n" if self.compiler_config.use_nonce else ""
        src_code = (
            f"Original Source Code:```\n{self.src_code}\n```\n\n" if self.compiler_config.include_source_code else ""
        )
        filled_in_system_prompt = self.compiler_config.prompt_template.system.format(nonce=nonce)

        if isinstance(node.value, ast.JoinedStr):
            # Natural block
            block = node.value.values
            expressions = []
            for val in block:
                if isinstance(val, ast.FormattedValue):
                    expressions.append(f"{{{ast.unparse(val.value)}}}")
                elif isinstance(val, ast.Constant):
                    expressions.append(val.value)

            natural_code = "".join(expressions)
        elif isinstance(node.value, ast.Constant):
            assert isinstance(node.value.value, str)
            natural_code = node.value.value

        natural_code = textwrap.dedent(natural_code).strip()

        logger.info(f"\n======== Natural Code ========\n{natural_code}\n=================")

        if self.compiler_config.execution_substrate == ExecutionSubstrate.PYTHON:

            class CodeSchema(BaseModel):
                code: str = Field(
                    description="The Python code to replace the natural language comment. Cannot contain `while True`."
                )

            MAX_RETRY = 3
            bad = True
            for retry_i in range(MAX_RETRY):
                # Query LLM to get Python code
                python_code = self.llm.gen_structured_output(
                    messages=[
                        UserMessage(
                            content=self.compiler_config.prompt_template.user.format(
                                natural_code=natural_code, source_code=src_code
                            )
                        )
                    ],
                    schema=ResponseFormat(res_schema=CodeSchema),
                    system=filled_in_system_prompt,
                ).code

                usage = self.llm.get_usage()
                if usage is None:
                    logger.warning("No LLM usage to log")
                else:
                    NJ_TELEMETRY.log_llm_usage(self.filename, self.funcname, usage)

                if "while" not in python_code.lower():
                    bad = False
                    break
            if bad:
                raise ValueError("Bad code")

            logger.info(f"\n=== Compiled Python Code ===\n{python_code}\n=================")

            python_ast = ast.parse(python_code)

            return python_ast
        else:
            raise NotImplementedError()


class InterpreterTransform(Compiler):
    """Source code transformation for interpreters. Replaces inline natural code with interpreter calls, and accompanying handlers."""

    def __init__(self, config: Config, src_code: str, filename: str, funcname: str):
        super().__init__(config, src_code, filename, funcname)
        if config.interpreter_config is None:
            raise ValueError("Interpreter config must be provided in interpreter mode")

    def compile_natural_block(self, node: ast.Expr, kwargs) -> ast.AST:
        """
        Compile natural language blocks just-in-time during execution.

        This compiler generates Python code that will make runtime calls
        to LLM agents to execute natural language instructions.

        Args:
            natural_code: The natural language instruction code

        Returns:
            Compiled Python code as a string that makes runtime LLM calls
        """

        natural_code = ast.unparse(node.value)
        output_vars, input_vars = extract_variable(natural_code)

        output_vars_dict_var = self.variable_generator()

        self.config = self.config.with_interpreter_updates(var_generator_init=self.variable_generator.current_id())

        valid_labels = set[str](["raise"])
        cases = [
            # Always have raise
            ast.match_case(
                pattern=ast.MatchValue(value=ast.Constant(value="raise")),
                body=[ast.Raise(exc=ast.Name(id=f"{NJ_VAR_PREFIX}effect_exc.value", ctx=ast.Load()))],
            )
        ]
        if kwargs["loop_depth"] > 0:
            cases.append(
                ast.match_case(
                    pattern=ast.MatchValue(value=ast.Constant(value="break")),
                    body=[ast.Break()],
                )
            )
            cases.append(
                ast.match_case(
                    pattern=ast.MatchValue(value=ast.Constant(value="continue")),
                    body=[ast.Continue()],
                )
            )

            valid_labels.update(["break", "continue"])

        if kwargs["function_depth"] > 0:
            cases.append(
                ast.match_case(
                    pattern=ast.MatchValue(value=ast.Constant(value="return")),
                    body=[ast.Return(value=ast.Name(id=f"{NJ_VAR_PREFIX}effect_exc.value", ctx=ast.Load()))],
                )
            )
            valid_labels.add("return")

        # Pass EffectException along
        cases.append(
            ast.match_case(
                pattern=ast.MatchAs(),
                body=[ast.Raise(exc=ast.Name(f"{NJ_VAR_PREFIX}effect_exc", ctx=ast.Load()))],
            )
        )

        compiled_ast: List[ast.stmt] = [
            ast.Assign(
                # output variables dictionary variable name
                targets=[ast.Name(id=output_vars_dict_var, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id=LIBNAME, ctx=ast.Load()),
                            attr="runtime",
                            ctx=ast.Load(),
                        ),
                        attr="execute",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[
                        ast.keyword("natural_code", node.value),
                        ast.keyword(
                            "python_frame",
                            ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="inspect", ctx=ast.Load()),
                                    attr="currentframe",
                                    ctx=ast.Load(),
                                ),
                                args=[],
                                keywords=[],
                            ),
                        ),
                        ast.keyword(
                            "config",
                            ast.Call(
                                func=ast.Attribute(
                                    value=ast.Attribute(
                                        value=ast.Name(id=LIBNAME, ctx=ast.Load()),
                                        attr="Config",
                                        ctx=ast.Load(),
                                    ),
                                    attr="model_validate_json",
                                    ctx=ast.Load(),
                                ),
                                args=[ast.Constant(value=self.config.model_dump_json())],
                                keywords=[],
                            ),
                        ),
                        ast.keyword("filename", ast.Constant(value=self.filename)),
                        ast.keyword("funcname", ast.Constant(value=self.funcname)),
                        ast.keyword(
                            "valid_labels",
                            ast.Set(
                                elts=[ast.Constant(value=label) for label in valid_labels],
                            ),
                        ),
                    ],
                ),
            ),
        ]

        # Add variable assignments for output variables
        compiled_ast += [
            ast.Assign(
                targets=[ast.Name(id=var.name, ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Name(id=output_vars_dict_var, ctx=ast.Load()),
                    slice=ast.Constant(value=var.name),
                    ctx=ast.Load(),
                ),
            )
            for var in output_vars
        ]

        # Wrap in a try block
        try_ast = ast.Try(
            body=compiled_ast,
            handlers=[
                ast.ExceptHandler(
                    type=ast.Attribute(
                        value=ast.Name(LIBNAME, ctx=ast.Load()),
                        attr=EffectException.__qualname__,
                    ),
                    name=f"{NJ_VAR_PREFIX}effect_exc",
                    body=[
                        ast.Match(
                            subject=ast.Attribute(
                                value=ast.Name(f"{NJ_VAR_PREFIX}effect_exc", ctx=ast.Load()),
                                attr="name",
                            ),
                            cases=cases,
                        )
                    ],
                ),
            ],
            orelse=[],
            finalbody=[],
        )

        return try_ast


class NaturalTransformer(ast.NodeTransformer):
    def __init__(self, compiler: Compiler):
        self.compiler = compiler
        self.loop_depth = 0
        self.function_depth = 0

    def visit_For(self, node):
        self.loop_depth += 1
        node = self.generic_visit(node)
        self.loop_depth -= 1
        return node

    def visit_While(self, node):
        self.loop_depth += 1
        node = self.generic_visit(node)
        self.loop_depth -= 1
        return node

    def visit_FunctionDef(self, node):
        self.function_depth += 1
        node = self.generic_visit(node)
        self.function_depth -= 1
        return node

    def visit_AsyncFunctionDef(self, node):
        self.function_depth += 1
        node = self.generic_visit(node)
        self.function_depth -= 1
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        if isinstance(node.value, ast.JoinedStr):
            # Natural block
            block = node.value.values
            if len(block) > 0 and isinstance(block[0], ast.Constant):
                if isinstance(block[0].value, str) and block[0].value.startswith(f"{LANGUAGE_IDENTIFIER}\n"):
                    # remove natural prefix
                    block[0].value = block[0].value[len(f"{LANGUAGE_IDENTIFIER}\n") :]

                    return self.compiler.compile_natural_block(
                        node,
                        kwargs={
                            "function_depth": self.function_depth,
                            "loop_depth": self.loop_depth,
                        },
                    )
        elif isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str) and node.value.value.startswith(f"{LANGUAGE_IDENTIFIER}\n"):
                # remove natural prefix
                node.value.value = node.value.value[len(f"{LANGUAGE_IDENTIFIER}\n") :]
                return self.compiler.compile_natural_block(
                    node,
                    kwargs={
                        "function_depth": self.function_depth,
                        "loop_depth": self.loop_depth,
                    },
                )

        return ast.Expr(cast(ast.expr, self.generic_visit(node.value)))


def truncate_source_code(source_code: str) -> str:
    """
    Removes all code sections in the source code that are enclosed between
    the TRUNCATE_START and TRUNCATE_END markers, including the markers themselves.

    Args:
        source_code: The input source code as a string.

    Returns:
        The source code with all truncated sections removed.
    """

    # Use regex to remove all code between TRUNCATE_START and TRUNCATE_END (including the markers)
    pattern = re.compile(
        rf"{re.escape(TRUNCATE_START)}.*?{re.escape(TRUNCATE_END)}",
        flags=re.DOTALL,
    )
    return re.sub(pattern, "", source_code)


def compile_nj(
    source_code: str,
    config: Config,
    filename: str,
    funcname: str,
) -> str:
    """
    Compiles Nightjar source code into Python source code
    """

    # Clean up indentation
    source_code = textwrap.dedent(source_code)

    # Truncate source code to remove code that shouldn't be shown to LLM
    truncated_source_code = truncate_source_code(source_code)

    # Select compiler
    if config.execution_strategy == ExecutionStrategy.AOTCOMPILATION:
        compiler = AOTCompiler(
            config=config,
            src_code=truncated_source_code,
            filename=filename,
            funcname=funcname,
        )
    elif config.execution_strategy == ExecutionStrategy.INTERPRETER:
        compiler = InterpreterTransform(
            config=config,
            src_code=truncated_source_code,
            filename=filename,
            funcname=funcname,
        )
    else:
        raise ValueError("Unknown execution strategy")

    # Compile natural blocks
    parsed_ast = ast.parse(source_code)
    transformer = NaturalTransformer(compiler)
    parsed_ast = transformer.visit(parsed_ast)
    ast.fix_missing_locations(parsed_ast)

    parsed_ast = compiler.add_preamble(parsed_ast)
    ast.fix_missing_locations(parsed_ast)

    return ast.unparse(parsed_ast)

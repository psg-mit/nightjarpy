import ast
import functools
import inspect
import linecache
import logging
import os
import textwrap
import types
from typing import Callable, ParamSpec, TypeVar, cast, overload

import dotenv

from nightjarpy.compilation import compile_nj
from nightjarpy.configs import DEFAULT_CONFIG, Config
from nightjarpy.utils.utils import enable_nj_logging

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

P = ParamSpec("P")
T = TypeVar("T")


class StripDecorator(ast.NodeTransformer):
    """Remove the final decorator in the list, which should be the decorator that is called on the function."""

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)
        node.decorator_list = node.decorator_list[:-1]
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.generic_visit(node)
        node.decorator_list = node.decorator_list[:-1]
        return node

    def visit_ClassDef(self, node: ast.ClassDef):
        self.generic_visit(node)
        node.decorator_list = node.decorator_list[:-1]
        return node


@overload
def fn(
    func: Callable[P, T],
    *,
    config: Config = DEFAULT_CONFIG,
) -> Callable[P, T]: ...


@overload
def fn(
    *,
    config: Config = DEFAULT_CONFIG,
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def fn(
    func: Callable[P, T] | None = None,
    *,
    config: Config = DEFAULT_CONFIG,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Bootstrap function with embedded natural code
    """
    verbose = os.getenv("NJ_VERBOSE", False)
    if verbose:
        enable_nj_logging()

    def wrapper(func: Callable[P, T]) -> Callable[P, T]:
        try:
            source_code = inspect.getsource(func)
        except OSError as e:
            raise RuntimeError("Cannot compile function") from e

        # Remove the current decorator
        source_code = textwrap.dedent(source_code)
        mod = ast.parse(source_code)
        mod = StripDecorator().visit(mod)
        source_code = ast.unparse(mod)

        filename = inspect.getsourcefile(func) or "<nightjar>"

        compiled_source_code = compile_nj(
            source_code=source_code,
            config=config,
            filename=filename,
            funcname=func.__name__,
        )
        logger.info("Compiled function:\n%s", compiled_source_code)

        # Rebuild the function
        compiled_code = compile(compiled_source_code, filename, "exec")

        globs = func.__globals__
        exec(compiled_code, globs)
        new_func = globs.get(func.__name__)

        if not isinstance(new_func, types.FunctionType):
            raise RuntimeError("Compiled code did not define the expected function.")

        lines = compiled_source_code.splitlines(keepends=True)
        # linecache.cache[filename] = (
        #     len(compiled_source_code),  # fake mtime
        #     None,  # size hint
        #     lines,  # lines of code
        #     filename,  # file name
        # )

        new_func = types.FunctionType(
            new_func.__code__,
            globs,
            func.__name__,
            func.__defaults__,
            func.__closure__,  # preserves free-var bindings / closure cells
        )
        new_func.__kwdefaults__ = func.__kwdefaults__
        new_func.__annotations__ = func.__annotations__
        new_func.__dict__.update(func.__dict__)
        new_func.__doc__ = func.__doc__
        functools.update_wrapper(new_func, func)

        return cast(Callable[P, T], new_func)

    return wrapper(func) if func is not None else wrapper

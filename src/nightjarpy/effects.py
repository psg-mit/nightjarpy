import logging
import re
import types
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Literal,
    Mapping,
    NamedTuple,
    NoReturn,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from nightjarpy.context import Context

from nightjarpy.types import (
    SCHEMA,
    SCHEMA_DEFS,
    SCHEMA_DEFS_NOFUNC,
    SUCCESS,
    VALUE_SCHEMA,
    Done,
    EffectError,
    EffectParams,
    Immutable,
    Label,
    NaturalCode,
    Ref,
    RegName,
    Success,
    UndefinedLocal,
    Value,
    Variable,
)

logger = logging.getLogger(__name__)


class Parameter(NamedTuple):
    name: str
    type: type[EffectParams] | types.UnionType
    access: Optional[Literal["read", "write"]] = None


class Effect:
    name: str
    description: str
    # Parameters for the LLM to generate, does not include parameters that must be given by the runtime handler
    parameters: Sequence[Parameter]
    schema_def: bool
    use_functions: bool

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Sequence[Parameter],
        handler,
        schema_def: bool = False,
        use_functions: bool = False,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.schema_def = schema_def
        self.handler = handler
        self.use_functions = use_functions

    def handler(self, context: "Context", *args, **kwargs) -> Any: ...

    def to_schema(self, model_name: str) -> Dict[str, Any]:
        model = model_name.lower()

        if model.startswith("openai/"):
            return self.to_openai_function()
        elif model.startswith("anthropic/"):
            return self.to_anthropic_function()
        else:
            raise ValueError(f"Unsupported model provider: {model}. Supported providers: openai/, anthropic/")

    def _parameter_schema(self) -> Dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {k: SCHEMA[v] for k, v, _ in self.parameters},
            "required": [p.name for p in self.parameters],
            "additionalProperties": False,
        }

        if self.schema_def:
            schema["$defs"] = SCHEMA_DEFS if self.use_functions else SCHEMA_DEFS_NOFUNC

        return schema

    def to_openai_schema(self) -> Dict[str, Any]:
        """
        Returns as an OpenAI structured output schema
        """
        schema = {
            "type": "object",
            "description": self.description,
            "properties": {
                "name": {"enum": [self.name]},
                "args": self._parameter_schema(),
            },
            "additionalProperties": False,
            "required": ["name", "args"],
            "strict": True,
        }

        return schema

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._parameter_schema(),
                "strict": True,
            },
        }

    def to_anthropic_function(self) -> Dict[str, Any]:
        """Convert tool to Anthropic's tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._parameter_schema(),
        }

    def __call__(self, context: "Context", *args, **kwargs) -> Any:
        return self.handler(context, *args, **kwargs)


@dataclass(frozen=True)
class EffectSet:
    effects: FrozenSet[Effect]
    final_effects: FrozenSet[str]  # Name of the effects to force as the final effect in a compilation/jit setup
    disable_compile: FrozenSet[str]  # Name of effects that are disabled during compilation

    def set_use_functions(self, use_functions: bool):
        for effect in self.effects:
            effect.use_functions = use_functions


def sanitize_code(expr: str) -> str:
    if re.match(r"help\(.*?\)", expr, flags=re.DOTALL) is not None:
        raise RuntimeError("`help` is not allowed")
    if "locals" in expr:
        raise RuntimeError("`locals` is not allowed")
    if "globals" in expr:
        raise RuntimeError("`globals` is not allowed")
    if "dir()" in expr:
        raise RuntimeError("`dir()` is not allowed")
    return expr


def lookup_handler(context: "Context", var: Variable) -> Immutable:
    return context.lookup(var, False)


lookup_effect = Effect(
    name="lookup",
    description="Lookup a variable in the context. Returns immutable value or reference",
    parameters=(Parameter("var", Variable, "read"),),
    handler=lookup_handler,
)


# def import_handler(context: "Context", module: str) -> Ref[Obj]:
#     res = context.import_module(module)
#     return res


# import_effect = Effect(
#     name="import",
#     description="Import a module into the heap, but does not assign the module to a variable. The module name must be given as a string. Returns the reference to the module. To refer to the module by variable name, use the `assign` or `assign_global` tool after receiving the reference.",
#     parameters={
#         "type": "object",
#         "properties": {
#             "module": {"type": "string"},
#         },
#         "required": ["module"],
#         "additionalProperties": False,
#     },
#     handler=import_handler,
# )


def assign_handler(context: "Context", var: Variable, val: Immutable) -> Success:
    return context.assign(var, val)


assign_effect = Effect(
    name="assign",
    description="Assign the immutable value or reference to the local variable `var` in the current program scope. Only works if the variable is specified as an output variable.",
    parameters=(Parameter("var", Variable, "write"), Parameter("val", Immutable, "read")),
    handler=assign_handler,
    schema_def=True,
)


# def assign_global_handler(context: "Context", var: Variable, ref: Ref[Value]) -> Literal["Success"]:
#     return context.assign_global(var, ref)


# assign_global_effect = Effect(
#     name="assign_global",
#     description="Assign a reference to a global variable",
#     parameters={
#         "type": "object",
#         "properties": {
#             "var": {"type": "string"},
#             "ref": Ref.json_schema(),
#         },
#         "required": ["var", "ref"],
#         "additionalProperties": False,
#     },
#     handler=assign_global_handler,
# )


# def delete_handler(context: "Context", var: Variable) -> Literal["Success"]:
#     return context.delete(var)


# delete_effect = Effect(
#     name="delete",
#     description="Delete a variable from the current frame",
#     parameters={
#         "type": "object",
#         "properties": {
#             "var": {"type": "string"},
#         },
#         "required": ["var"],
#         "additionalProperties": False,
#     },
#     handler=delete_handler,
# )


# def store_handler(context: "Context", value: Immutable) -> Ref[Immutable]:
#     return context.store(value)


# alloc_int_effect = Effect(
#     name="alloc_int",
#     description="Allocate an integer in the heap. Returns the reference to the value",
#     parameters={
#         "type": "object",
#         "properties": {
#             "value": {"type": "integer"},
#         },
#         "required": ["value"],
#         "additionalProperties": False,
#     },
#     handler=store_handler,
# )

# alloc_number_effect = Effect(
#     name="alloc_number",
#     description="Allocate an number in the heap. Returns the reference to the value",
#     parameters={
#         "type": "object",
#         "properties": {
#             "value": {"type": "number"},
#         },
#         "required": ["value"],
#         "additionalProperties": False,
#     },
#     handler=store_handler,
# )

# alloc_string_effect = Effect(
#     name="alloc_string",
#     description="Allocate an string in the heap. Returns the reference to the value",
#     parameters={
#         "type": "object",
#         "properties": {
#             "value": {"type": "string"},
#         },
#         "required": ["value"],
#         "additionalProperties": False,
#     },
#     handler=store_handler,
# )

# alloc_boolean_effect = Effect(
#     name="alloc_boolean",
#     description="Allocate an boolean in the heap. Returns the reference to the value",
#     parameters={
#         "type": "object",
#         "properties": {
#             "value": {"type": "boolean"},
#         },
#         "required": ["value"],
#         "additionalProperties": False,
#     },
#     handler=store_handler,
# )

# alloc_null_effect = Effect(
#     name="alloc_null",
#     description="Allocate an null/None in the heap. Returns the reference to the value",
#     parameters={
#         "type": "object",
#         "properties": {
#             "value": {"type": "null"},
#         },
#         "required": ["value"],
#         "additionalProperties": False,
#     },
#     handler=store_handler,
# )


# def alloc_func_handler(context: "Context", func_name: str, parameters: list | None, full_func: str) -> Ref:
#     return context.create_func(
#         func_name=func_name,
#         parameters=parameters,
#         full_func=full_func,
#     )


# alloc_func_effect = Effect(
#     name="alloc_func",
#     description="Allocate a new function in the heap with the current frame as the locals of the function. Give the name of the function, the parameters, and then a string of the Python function with the given name and parameters. You cannot embed tool calls in the code. Make sure all variables used in the function are assigned in the frames. Make sure all the libraries (such as `string` or `random`) you use are imported in `full_func`, assume there are no existing imports. YOU CANNOT USE THIS TOOL TO RAISE ERRORS, use `raise` instead. `alloc_func` will return the reference to the created function. You must still assign the function reference to a variable or attribute in the context if it is an output or you want to reference it by name.",
#     parameters={
#         "type": "object",
#         "properties": {
#             "func_name": {"type": "string"},
#             "parameters": Func.parameters_json_schema(),
#             "full_func": {"type": "string"},
#         },
#         "required": [
#             "func_name",
#             "parameters",
#             "full_func",
#         ],
#         "additionalProperties": False,
#     },
#     handler=alloc_func_handler,
# )


# def alloc_class_handler(
#     context: "Context",
#     class_name: str,
#     base_classes: list,
#     kwds: list,
# ) -> Ref[Obj]:
#     return context.alloc_class(
#         class_name=class_name,
#         base_classes=base_classes,
#         kwds=kwds,
#     )


# alloc_class_effect = Effect(
#     name="alloc_class",
#     description="Allocate a new class object in the heap. Requires `class_name` (the name of the class), `base_classes` (a list of base class objects), and `kwds` (keyword arguments). Returns the reference to the class object. Methods can be added to the class by using the `setattr` tool.",
#     parameters={
#         "type": "object",
#         "properties": {
#             "class_name": {"type": "string"},
#             "base_classes": {
#                 "type": "array",
#                 "items": Ref.json_type_obj(),
#             },
#             "kwds": {
#                 "type": "array",
#                 "items": {
#                     "type": "object",
#                     "properties": {
#                         "arg_name": {"type": ["string"]},
#                         "arg_ref": Ref.json_schema(),
#                     },
#                     "required": ["arg_name", "arg_ref"],
#                     "additionalProperties": False,
#                 },
#             },
#         },
#         "required": ["class_name", "base_classes", "kwds"],
#         "additionalProperties": False,
#     },
#     handler=alloc_class_handler,
# )


# def signature_handler(context: "Context", func_ref: Ref[Func]) -> str | None:
#     sig = context.signature(func_ref=func_ref)
#     if sig is None:
#         return sig
#     else:
#         return str(sig)


# signature_effect = Effect(
#     name="signature",
#     description="Get the signature of a function (whose reference is given) in the heap.",
#     parameters={
#         "type": "object",
#         "properties": {
#             "func_ref": Ref.json_type_func(),
#         },
#         "required": ["func_ref"],
#         "additionalProperties": False,
#     },
#     handler=signature_handler,
# )


# def call_handler(
#     context: "Context", func_ref: Ref[Func], arguments: List[Tuple[Optional[str], Ref[Value]]]
# ) -> Ref[Value]:
#     return context.call(func_ref=func_ref, arguments=arguments)


# call_effect = Effect(
#     name="call",
#     description="Call a function (whose reference is given) in the heap. The function must be called with the correct number of arguments. Provide the array of parameter name and reference of the value that should passed as the argument. The function will return the reference to the return value.",
#     parameters={
#         "type": "object",
#         "properties": {
#             "func_ref": Ref.json_type_func(),
#             "arguments": {
#                 "type": "array",
#                 "items": {
#                     "type": "object",
#                     "properties": {
#                         "arg_name": {"type": ["string", "null"]},
#                         "arg_ref": Ref.json_schema(),
#                     },
#                     "required": ["arg_name", "arg_ref"],
#                     "additionalProperties": False,
#                 },
#             },
#         },
#         "required": ["func_ref", "arguments"],
#         "additionalProperties": False,
#     },
#     handler=call_handler,
# )


def deref_handler(context: "Context", ref: Ref) -> Value:
    return context.deref(ref)


deref_effect = Effect(
    name="deref",
    description="Dereference the value at the reference in the heap",
    parameters=(Parameter("ref", Ref, "read"),),
    handler=deref_handler,
)


def ref_handler(context: "Context", val: Value) -> Ref:
    return context.ref(val)


ref_effect = Effect(
    name="ref",
    description="Create a reference pointing to the given value",
    parameters=(Parameter("val", Value, "read"),),
    handler=ref_handler,
    schema_def=True,
)


def setref_handler(context: "Context", ref: Ref, val: Value) -> Success:
    return context.setref(ref, val)


setref_effect = Effect(
    name="setref",
    description="In-place updates the value that the reference points to to the value. This tool does not create fresh references, use `ref` to create references.",
    parameters=(Parameter("ref", Ref, "write"), Parameter("val", Value, "read")),
    handler=setref_handler,
    schema_def=True,
)


def goto_handler(context: "Context", label: Label, val: Immutable) -> NoReturn:
    context.goto(label, val)


goto_effect = Effect(
    name="goto",
    description="Jumps the evaluation context to the program label `label` with the value `val` or the value at reference `val`. This also raises errors with the `raise` program label (`val` must hold reference that points to an exception object)",
    parameters=(Parameter("label", Label, "read"), Parameter("val", Immutable, "read")),
    handler=goto_handler,
    schema_def=True,
)


# def listattr_handler(context: "Context", ref: Ref[Obj]) -> Set[Variable]:
#     return context.listattr(ref)


# listattr_effect = Effect(
#     name="listattr",
#     description="List all defined attributes of an object (whose ref is given).",
#     parameters={
#         "type": "object",
#         "properties": {
#             "ref": Ref.json_type_obj(),
#         },
#         "required": ["ref"],
#         "additionalProperties": False,
#     },
#     handler=listattr_handler,
# )


# def getattr_handler(context: "Context", ref: Ref[Obj], attr: str) -> Ref[Value]:
#     return context.getattr(ref, attr)


# getattr_effect = Effect(
#     name="gettattr",
#     description="Get an attribute of an object (whose ref is given) from the heap, returning the reference to the value. If the attribute is a function, a function ref is given, which can be called using the `call` tool (not to be confused with variables or attributes named `call` or `__call__`).",
#     parameters={
#         "type": "object",
#         "properties": {
#             "ref": Ref.json_type_obj(),
#             "attr": {"type": "string"},
#         },
#         "required": ["ref", "attr"],
#         "additionalProperties": False,
#     },
#     handler=getattr_handler,
# )


# def setattr_handler(context: "Context", obj_ref: Ref[Obj], attr: str, val_ref: Ref[Value]) -> str:
#     return context.setattr(obj_ref=obj_ref, attr=attr, val_ref=val_ref)


# setattr_effect = Effect(
#     name="setattr",
#     description="Set an attribute (property or method) of an object (whose ref is given in `obj_ref`) in the heap to the reference given in `val_ref`.",
#     parameters={
#         "type": "object",
#         "properties": {
#             "obj_ref": Ref.json_type_obj(),
#             "attr": {"type": "string"},
#             "val_ref": Ref.json_schema(),
#         },
#         "required": ["obj_ref", "attr", "val_ref"],
#         "additionalProperties": False,
#     },
#     handler=setattr_handler,
# )


# def delattr_handler(context: "Context", ref: Ref[Obj], attr: str) -> str:
#     return context.delattr(obj_ref=ref, attr=attr)


# delattr_effect = Effect(
#     name="delattr",
#     description="Delete an attribute of an object (whose ref is given) from the heap",
#     parameters={
#         "type": "object",
#         "properties": {
#             "ref": Ref.json_type_obj(),
#             "attr": {"type": "string"},
#         },
#         "required": ["ref", "attr"],
#         "additionalProperties": False,
#     },
#     handler=delattr_handler,
# )


# def raise_handler(context: "Context", error: Ref[Obj]) -> str:
#     err = context.get_obj(error)
#     if not isinstance(err, BaseException):
#         raise ValueError(f"Did not get an exception, got {type(err)}")
#     if err.__class__.__name__ == "BaseException":
#         result = "BaseException is not allowed. Exception type must be a subclass of `Exception`"
#         logger.info(f"Raise tried to raise BaseException")
#         return result
#     raise Raise(err)


# raise_effect = Effect(
#     name="raise",
#     description="Use when you want to raise an error. This will raise an error. The error object (e.g. ValueError, TypeError, etc.) must be given as a reference. This function cannot take a string as an argument, only an error object.",
#     parameters={
#         "type": "object",
#         "properties": {
#             "error": {
#                 "type": "object",
#                 "properties": {
#                     "value_type": {
#                         "type": "string",
#                         "enum": [
#                             "object",
#                         ],
#                     },
#                     "addr": {"type": "integer"},
#                 },
#                 "required": ["value_type", "addr"],
#                 "additionalProperties": False,
#             }
#         },
#         "required": ["error"],
#         "additionalProperties": False,
#     },
#     handler=raise_handler,
# )


def raise_var_handler(context: "Context", error: Variable) -> NoReturn:
    ref = context.lookup(error)
    context.goto(Label("raise"), ref)


raise_var_effect = Effect(
    name="raise",
    description="Use when you want to raise an error. This will raise an error. The error object (e.g. ValueError, TypeError, etc.) must be given via a variable. Make sure to make variable assignment before you use this. This function cannot take a string as an argument, only an error object.",
    parameters=(Parameter("error", Variable, "read"),),
    handler=raise_var_handler,
)


def break_handler(context: "Context") -> NoReturn:
    context.goto(Label("break"), None)


break_effect = Effect(
    name="break",
    description="Break out of the loop context.",
    parameters=(),
    handler=break_handler,
)


def continue_handler(context: "Context") -> NoReturn:
    context.goto(Label("continue"), None)


continue_effect = Effect(
    name="continue",
    description="Continue the loop.",
    parameters=(),
    handler=continue_handler,
)


# def return_handler(context: "Context", value: Ref[Value]) -> None:
#     raise Return(context.get_python_value(value))


# return_effect = Effect(
#     name="return",
#     description="Return a value (passed by its reference) from the context. This will end the function and return the value.",
#     parameters={
#         "type": "object",
#         "properties": {
#             "value": Ref.json_schema(),
#         },
#         "required": ["value"],
#         "additionalProperties": False,
#     },
#     handler=return_handler,
# )


# def return_immut_handler(context: "Context", value: Immutable | Ref[Value]) -> None:
#     if isinstance(value, Ref):
#         val = context.get_python_value(value)
#     else:
#         val = value

#     raise Return(val)


# return_immut_effect = Effect(
#     name="return",
#     description="Return a value (immutable value or object reference) from the context. This will end the function and return the value.",
#     parameters={
#         "type": "object",
#         "properties": {"value": immutable_json_schema},
#         "required": ["value"],
#         "additionalProperties": False,
#     },
#     handler=return_immut_handler,
# )


def return_var_handler(context: "Context", variable: Variable) -> None:
    val = context.lookup(variable)
    context.goto(Label("return"), val)


return_var_effect = Effect(
    name="return",
    description="Return a value from the context. If the instruction says 'return' you MUST use this tool, NOT `done`. Otherwise, you MUST NOT use this tool.",
    parameters=(Parameter("variable", Variable, "read"),),
    handler=return_var_handler,
)


def eval_isolated_handler(context: "Context", expr: str) -> Immutable:
    expr = sanitize_code(expr)
    try:
        v = eval(expr)
    except Exception as e:
        raise RuntimeError(f"Python code raised an error, {e}")
    ref = context.encode_python_value(v, {})
    return ref


eval_isolated_effect = Effect(
    name="eval",
    description="Evaluate a Python expression. The tool returns the result of the expression. Only string, int, float, None, bool will be returned. All other data types (including lists and dictionaries) will be returned as a reference to the object. Use serialize with `str` or custom functions to inspect them. To inspect attributes of an object (created inline or via a variable), use `getattr`, `str(dir(...))`, `hasattr`, etc. This tool does not alter the current context. You cannot use `help`. Do not use <var> or <:var> syntax. Code must be valid Python code. This tool cannot see the context; it cannot access variables nor write variables. It cannot take statements",
    parameters=(Parameter("expr", str),),
    handler=eval_isolated_handler,
)


def eval_handler(context: "Context", expr: str) -> Immutable:
    expr = sanitize_code(expr)
    python_locals, python_globals = context.get_closure(context.fp)
    try:
        v = eval(expr, python_globals, python_locals)
    except Exception as e:
        raise RuntimeError(f"Python code raised an error, {e}")
    ref = context.encode_python_value(v, {})
    return ref
    # if isinstance(ref, Ref)
    #     val = context.deref(ref)  # type: ignore
    #     return val
    # else:
    #     return ref


eval_effect = Effect(
    name="eval",
    description="Evaluate a Python expression in the current context. The tool returns the result of the expression. Only string, int, float, None, bool will be returned. All other data types (including lists and dictionaries) will be returned as a reference to the object. Use serialize with `str` or custom functions to inspect them. To inspect attributes of an object (created inline or via a variable), use `getattr`, `str(dir(...))`, `hasattr`, etc. This tool does not alter the current context. You cannot use `help`. Do not use <var> or <:var> syntax. Code must be valid Python code",
    parameters=(Parameter("expr", str),),
    handler=eval_handler,
)


def exec_handler(context: "Context", code: str) -> Success:
    code = sanitize_code(code)
    # Execute the function
    logger.info(f"Python code:\n{code}")

    python_locals, python_globals = context.get_closure(context.fp)
    full_code = "\n    ".join(code.split("\n"))
    extract = "\n    ".join(
        [f"{k} = nj__locals['{k}']" for k in python_locals.keys() if k is not None and not k.lower().startswith("nj__")]
    )
    # put locals into globals
    python_globals.update(python_locals)

    code = f"""def nj__wrapper(nj__locals):
    {extract}
    {full_code}

    for nj__k, nj__v in locals().items():
        nj__locals[nj__k] = nj__v

    return nj__locals
"""

    logger.info(f"Running code:\n{code}")

    exec(code, python_globals)

    try:
        # Call the function with the arguments
        new_python_locals = python_globals["nj__wrapper"](python_locals)
    except Exception as e:
        raise RuntimeError(f"Code raised an error, {e}")

    # Update the locals
    for k, v in new_python_locals.items():
        if k.startswith("nj_") or k in ["self"]:
            continue
        ref = context.encode_python_value(v, {})

        context.current_frame.insert(Variable(k), ref)
        context.valid_vars.add(Variable(k))
    return SUCCESS


exec_effect = Effect(
    name="exec",
    description="Execute a Python code block in the current context. It cannot contain natural blocks. This tool does not return anything. Issue only 1-10 lines of code at a time. You cannot use `help`. You cannot see `print` statements. Do not use <var> or <:var> syntax. Code must be valid Python code",
    parameters=(Parameter("code", str),),
    handler=exec_handler,
)


def done_handler(context: "Context") -> str:
    outputs: Dict[str, Any] = {}
    for var in context.output_vars:
        key = var.name
        try:
            val = context.lookup(var, local_only=True)
            outputs[key] = context.decode_and_sync_python_value(val, {})
        except UndefinedLocal:
            try:
                val = context.lookup(var, local_only=False)
                outputs[key] = context.decode_and_sync_python_value(val, {})
                if context.python_frame:
                    context.python_frame.f_globals[key] = outputs[key]
                else:
                    raise ValueError(f"Can't find calling Python frame")
            except ValueError:
                return f"Error: Cannot exit agent loop; output variable is not defined `{key}`. Define `{key}` before try `done` again."
            except Exception as e:
                raise e
        except Exception as e:
            raise ValueError(f"Output variable `{key}` is ill-defined. Please fix before exiting: {e}")
    raise Done(outputs)


done_effect = Effect(
    name="done",
    description="Use this tool when done computing the instructions if the instruction does not say to break, return, or continue. Only use this when all the output variables have been assigned to references in the frame. You must use this tool to exit the computation if break, return, or continue are not triggered.",
    parameters=(),
    handler=done_handler,
)


def load_reg_handler(context: "Context", reg: RegName) -> Value:
    return context.loadreg(reg)


load_reg_effect = Effect(
    name="load_reg",
    description="Returns the value held in register `reg`",
    parameters=(Parameter("reg", RegName, "read"),),
    handler=load_reg_handler,
)


def store_reg_handler(context: "Context", dest: RegName, val: Value) -> Success:
    return context.storereg(dest, val)


store_reg_effect = Effect(
    name="store_reg",
    description="Stores into register `dest` the given value `val`. `val` cannot be an expression. It must data or references. Use this to store references into a register so dereference can be done on the register",
    parameters=(Parameter("dest", RegName, "write"), Parameter("val", Value)),
    handler=store_reg_handler,
    schema_def=True,
)


def lookup_reg_handler(context: "Context", dest: RegName, var: Variable) -> Success:
    return context.lookup_reg(dest, var, False)


lookup_reg_effect = Effect(
    name="lookup",
    description="Lookup a variable `var` in the context and stores the value into the destination register `dest`",
    parameters=(Parameter("dest", RegName, "write"), Parameter("var", Variable, "read")),
    handler=lookup_reg_handler,
)


def assign_reg_handler(context: "Context", var: Variable, src: RegName) -> Success:
    return context.assign_reg(var, src)


assign_reg_effect = Effect(
    name="assign",
    description="Assign the immutable value or reference stored in `src` to the local variable `var` in the current program scope. Only works if the variable is specified as an output variable.",
    parameters=(Parameter("var", Variable, "write"), Parameter("src", RegName, "read")),
    handler=assign_reg_handler,
)


def deref_reg_handler(context: "Context", dest: RegName, refreg: RegName) -> Success:
    return context.deref_reg(dest, refreg)


deref_reg_effect = Effect(
    name="deref",
    description="Dereference the value at the reference stored in the register `refreg` and store it into the `dest` register.",
    parameters=(Parameter("dest", RegName, "write"), Parameter("refreg", RegName, "read")),
    handler=deref_reg_handler,
)


def ref_reg_handler(context: "Context", dest: RegName, valreg: RegName) -> Success:
    return context.ref_reg(dest, valreg)


ref_reg_effect = Effect(
    name="ref",
    description="Create a reference pointing to the value stored in register `valreg` and then stores the created reference into the register `dest`. Use `load_reg` to read the reference.",
    parameters=(Parameter("dest", RegName, "write"), Parameter("valreg", RegName, "read")),
    handler=ref_reg_handler,
)


def setref_reg_handler(context: "Context", refreg: RegName, valreg: RegName) -> Success:
    return context.setref_reg(refreg, valreg)


setref_reg_effect = Effect(
    name="setref",
    description="In-place updates the value that the reference stored in register `refreg` points to to the value stored in register `valreg`. This tool does not create fresh references, use `ref` to create references.",
    parameters=(Parameter("refreg", RegName, "write"), Parameter("valreg", RegName, "read")),
    handler=setref_reg_handler,
)


def goto_reg_handler(context: "Context", label: Label, valreg: RegName) -> NoReturn:
    context.goto_reg(label, valreg)


goto_reg_effect = Effect(
    name="goto",
    description="Jumps the evaluation context to the program label `label` with the value stored in `valreg` This also raises errors with the `raise` program label (valreg must hold reference that points to an exception object)",
    parameters=(Parameter("label", Label, "read"), Parameter("valreg", RegName, "read")),
    handler=goto_reg_handler,
)


def compute_reg_handler(context: "Context", instruction: NaturalCode, dest: RegName, src: List[RegName]) -> Success:
    return context.compute_reg(instruction, dest, src)


compute_reg_effect = Effect(
    name="compute",
    description="Performs functional computation using a LLM. Instruction can be any language (natural or formal), but keep it simple and ask for only one step of computation at a time. It cannot generate Python values, only in the encoding format. Specify the expected return type and include the type annotations if it is not one of the primitives. Reads inputs from the `src` registers and writes a return value into the `dest` register. All inputs and type information must be included in instruction or as `src` as the LLM will have no access the context. References are not automatically dereferenced for the LLM, so make sure to only pass in registers that hold the relevant dereferenced values. The LLM cannot perform in-place updates and will get confused if asked to do so, so don't use 'mutable' language like 'append'. The LLM cannot create references; this needs to be done by you before you call this tool. The LLM cannot created nested lists, dictionaries, set, objects, classes (which all need to be allocated on the heap to get references), so do not ask it to create data with nested mutable data unless you give it the references to those data.",
    parameters=(
        Parameter("instruction", NaturalCode),
        Parameter("dest", RegName, "write"),
        Parameter("src", List[RegName], "read"),
    ),
    handler=compute_reg_handler,
)

BASE_EFFECTS_NOREG = EffectSet(
    effects=frozenset(
        [
            lookup_effect,
            assign_effect,
            deref_effect,
            ref_effect,
            setref_effect,
            goto_effect,
            done_effect,
        ]
    ),
    final_effects=frozenset([done_effect.name, goto_effect.name]),
    disable_compile=frozenset(),
)

PYTHON_BASE_ISOLATED_EFFECTS_NOREG = EffectSet(
    effects=frozenset(
        [
            eval_isolated_effect,
            lookup_effect,
            assign_effect,
            deref_effect,
            ref_effect,
            setref_effect,
            raise_var_effect,
            break_effect,
            continue_effect,
            return_var_effect,
            done_effect,
        ]
    ),
    final_effects=frozenset(
        [done_effect.name, raise_var_effect.name, break_effect.name, continue_effect.name, return_var_effect.name]
    ),
    disable_compile=frozenset(),
)

PYTHON_BASE_EFFECTS_NOREG = EffectSet(
    effects=frozenset(
        [
            eval_effect,
            assign_effect,
            deref_effect,
            setref_effect,
            raise_var_effect,
            break_effect,
            continue_effect,
            return_var_effect,
            done_effect,
        ]
    ),
    final_effects=frozenset(
        [done_effect.name, raise_var_effect.name, break_effect.name, continue_effect.name, return_var_effect.name]
    ),
    disable_compile=frozenset(),
)

PYTHON_EFFECTS_V1 = EffectSet(
    effects=frozenset(
        [
            eval_effect,
            exec_effect,
            raise_var_effect,
            break_effect,
            continue_effect,
            return_var_effect,
            done_effect,
        ]
    ),
    final_effects=frozenset(
        [done_effect.name, raise_var_effect.name, break_effect.name, continue_effect.name, return_var_effect.name]
    ),
    disable_compile=frozenset(),
)

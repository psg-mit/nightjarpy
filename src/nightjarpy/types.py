import ast
import inspect
import json
import logging
import types
from abc import abstractmethod, update_abstractmethods
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    get_args,
)

import anthropic
import openai
from anthropic.types.beta import BetaUsage
from openai.types.responses import ResponseUsage
from pydantic import BaseModel, PlainSerializer

if TYPE_CHECKING:
    from nightjarpy.context import Context

logger = logging.getLogger(__name__)

NJ_VAR_PREFIX = "nj__"

T = TypeVar("T")

JsonSchemaValue: TypeAlias = Union[str, bool, Mapping[str, "JsonSchemaValue"], Sequence["JsonSchemaValue"]]
JsonSchema: TypeAlias = Mapping[str, JsonSchemaValue]

JsonType: TypeAlias = Union[
    int,
    float,
    str,
    bool,
    None,
    Mapping[str, "JsonType"],
    Sequence["JsonType"],
]


class Success:
    def __str__(self) -> str:
        return "Success"

    def __repr__(self) -> str:
        return self.__str__()


SUCCESS = Success()


class GetFail(Exception):
    pass


class UndefinedLocal(Exception):
    pass


class EffectError(Exception):
    pass


@dataclass(frozen=True)
class Variable:
    name: str

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.__str__()

    def json_value(self) -> JsonType:
        return self.name

    @classmethod
    def json_schema(cls) -> JsonSchema:
        return {"type": "string"}

    @classmethod
    def from_json(cls, json: JsonType) -> "Variable":
        if not isinstance(json, str):
            raise ValueError("Unexpected serialization")
        return Variable(json)


@dataclass(frozen=True)
class Label:
    name: str

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.__str__()

    def json_value(self) -> JsonType:
        return self.name

    @classmethod
    def json_schema(cls) -> JsonSchema:
        return {"type": "string"}

    @classmethod
    def from_json(cls, json: JsonType) -> "Label":
        if not isinstance(json, str):
            raise ValueError("Unexpected serialization")
        return Label(json)


# Variable: TypeAlias = str
# Label: TypeAlias = str
Address: TypeAlias = int
NaturalCode: TypeAlias = str


class EffectException(Exception):
    def __init__(self, name: str, value: Any):
        super().__init__(name, value)
        self.name = name
        self.value = value


# class BreakLoop(EffectException):
#     pass


# class ContinueLoop(EffectException):
#     pass


# class Return(EffectException):
#     def __init__(self, value: Any):
#         super().__init__(value)
#         self.value = value


class Done(Exception):
    def __init__(self, outputs: Dict[str, Any]):
        super().__init__(outputs)
        self.outputs = outputs


# class Raise(Exception):
#     def __init__(self, err: Any):
#         super().__init__(err)
#         self.err = err


@dataclass(frozen=True)
class Ref:
    addr: Address

    def __str__(self) -> str:
        return f"Ref({self.addr})"

    def __repr__(self) -> str:
        return self.__str__()

    def json_value(self) -> JsonType:
        return {
            "type": "Ref",
            "addr": self.addr,
        }

    @classmethod
    def json_schema(cls) -> JsonSchema:
        return {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["Ref"]},
                "addr": {"type": "integer"},
            },
            "required": ["type", "addr"],
            "additionalProperties": False,
        }

    @classmethod
    def from_json(cls, json: JsonType) -> "Ref":
        if not isinstance(json, dict):
            raise ValueError("Unexpected serialization")
        if "type" not in json:
            raise ValueError("Unknown value type")
        if json["type"] != "Ref" or "addr" not in json or not isinstance(json["addr"], int):
            raise ValueError("Unexpected serialization")
        return Ref(addr=json["addr"])


@dataclass(frozen=True)
class RegName:
    name: str

    def __str__(self) -> str:
        return f"Register({self.name})"

    def __repr__(self) -> str:
        return self.__str__()

    def json_value(self) -> JsonType:
        return {
            "type": "Register",
            "name": self.name,
        }

    @classmethod
    def json_schema(cls) -> JsonSchema:
        return {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["Register"]},
                "name": {"type": "string"},
            },
            "required": ["type", "name"],
            "additionalProperties": False,
        }

    @classmethod
    def from_json(cls, json: JsonType) -> "RegName":
        if not isinstance(json, dict):
            raise ValueError("Unexpected serialization")
        if "type" not in json:
            raise ValueError("Unknown value type")
        if json["type"] != "Register" or "name" not in json or not isinstance(json["name"], str):
            raise ValueError("Unexpected serialization")
        return RegName(name=json["name"])


@dataclass()
class NotSupportedDataType:
    pass


Primitive: TypeAlias = Union[
    types.NoneType,
    str,
    int,
    # complex,
    float,
    bool,
    Ref,
    datetime,
    # range,
    # bytes,
    # bytearray,
    # memoryview
]

Immutable: TypeAlias = Union[
    Primitive,
    NotSupportedDataType,
    Tuple["Immutable", ...],
]


@dataclass
class Class:
    name: str
    annotations: Dict[str, str]
    attributes: Dict[str, Immutable]
    bases: Tuple[Ref, ...]

    def __str__(self) -> str:
        return f"Class[{self.name}]({self.bases}, {self.annotations}, {self.attributes})"

    def __repr__(self) -> str:
        return self.__str__()


# Encoded version of object that only uses Immutables in the dictionary
@dataclass
class Object:
    _class: str
    attributes: Dict[str, Immutable]

    def __str__(self) -> str:
        return f"Object[{self._class}]({self.attributes})"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(frozen=True)
class Param:
    name: str
    annotation: str
    kind: Literal[
        "positional-only",
        "positional or keyword",
        "variadic positional",
        "keyword-only",
        "variadic keyword",
    ]
    default: Immutable

    def __str__(self) -> str:
        return f"Param[{self.name}]({self.annotation}, {self.kind}, {self.default})"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(frozen=True)
class Signature:
    params: Tuple[Param, ...]

    def __str__(self) -> str:
        return f"Signature({str(self.params)})"

    def __repr__(self) -> str:
        return self.__str__()


class Func:
    """Represents a function in Context

    This class encapsulates the metadata and implementation details of a function,
    including its frame pointer, name, parameters, source code, and an optional
    Python callable.
    """

    def __init__(
        self,
        context: "Context",
        name: str,
        signature: Signature,
        full_func: str,
        python_func: Optional[Callable] = None,
    ):
        self.context = context
        self.name = name
        self.signature: Signature = signature
        self.full_func = full_func
        self.llm_def_func = None

        if python_func is None:

            def _extract_function_name(f_def: str) -> str:
                tree = ast.parse(f_def)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        return node.name
                raise ValueError(
                    f"No function definition found in `full_func`. The code must be wrapped in a function definition."
                )

            # Check if function exists in the compiled code
            if name != _extract_function_name(self.full_func):
                raise ValueError(
                    f"Function name `{name}` does not match the function definition in `full_func`. The code must be wrapped in a function definition."
                )

            python_locals, python_globals = context.get_closure(context.fp)
            extract = "\n    ".join(
                [
                    f"{k} = nj__locals['{k}']"
                    for k in python_locals.keys()
                    if k is not None and not k.lower().startswith("nj__")
                ]
            )

            # prepend prefix to function name to avoid clashes
            # if not self.name.startswith(NJ_VAR_PREFIX):
            # self.full_func = self.full_func.replace(f"def {name}", f"def {NJ_VAR_PREFIX}{name}")
            # self.name = f"{NJ_VAR_PREFIX}{name}"

            func = "\n    ".join(self.full_func.split("\n"))

            # Try to get parameters from signature
            #             try:
            #                 code = f"""import inspect
            # def nj__wrapper(nj__locals):
            #     {extract}
            #     {func}
            #     return list(inspect.signature({self.name}).parameters.values())
            # """

            #                 exec(code, python_globals)
            #                 # Save extracted parameters
            #                 py_params = cast(List[inspect.Parameter], python_globals["nj__wrapper"](python_locals))
            #                 extracted_signature = self.context.encode_signature(py_params)
            #             except Exception as e:
            #                 logger.info(f"Failed to get parameters from signature, using given parameters by LLM")
            #                 logger.info(f"Error: {e}")
            #                 extracted_signature = signature

            # if extracted_signature != signature:
            # logger.info(f"Received differing signature, using given parameters by LLM")
            # self.signature = signature
            # else:
            # self.signature = extracted_signature
            self.signature = signature

            # Locals -> Globals x Locals x Method -> Result
            code = f"""import logging
logger = logging.getLogger("Func")
def nj__wrapper(nj__locals):
    {extract}
    {func}

    return {self.name}
"""
            logger.info(f"Defined code:\n{code}")

            exec(code, python_globals)

            # Python func: Globals x Locals x Method -> Result
            self.llm_def_func = python_globals["nj__wrapper"](python_locals)

        self.python_func = python_func

    def __call__(self, *args, **kwargs):
        if self.python_func is None:
            assert self.llm_def_func is not None
            # python_locals, python_globals = self.context.get_closure(self.context.fp)
            # python_locals["nj__args"] = args
            # python_locals["nj__kwargs"] = kwargs

            try:
                # Call the function with the arguments
                return self.llm_def_func.__call__(*args, **kwargs)
                # return python_globals["nj__wrapper"](python_locals)
            except Exception as e:
                e_ref = self.context.encode_python_value(e, {})
                try:
                    logger.info(f"Error: {e_ref}: {e}")
                except Exception as e:
                    pass
                raise RuntimeError(f"{self.name} raised an error, {e_ref}")
        else:
            return self.python_func(*args, **kwargs)

    def __signature__(self):
        return inspect.Signature(parameters=self.context.decode_signature(self.signature, {}))

    def to_python_func(
        self,
    ) -> Callable | "Func":
        if self.python_func is not None:
            return self.python_func
        return self

    def __repr__(self):
        return f"Func({self.name}, {self.signature}, {self.full_func})"


Mutable: TypeAlias = Union[
    List["Immutable"],
    Dict["Immutable", "Immutable"],
    Set["Immutable"],
    # frozenset,
    Object,
    Class,
    Func,
]

# Mutable: TypeAlias = Union[
#     List["Immutable"],
#     Dict["Immutable", "Immutable"],
#     Set["Immutable"],
#     # frozenset,
#     Object,
#     Class,
#     # Func,
# ]

Value: TypeAlias = Union[Immutable, Mutable]

SCHEMA: Dict[Type | types.UnionType, JsonSchema] = {
    types.NoneType: {"type": "null"},
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    datetime: {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["datetime"]},
            "value": {"type": "string", "format": "date-time"},
        },
        "required": ["type", "value"],
        "additionalProperties": False,
    },
    Ref: Ref.json_schema(),
    Variable: Variable.json_schema(),
    Label: Label.json_schema(),
    RegName: RegName.json_schema(),
    List[RegName]: {
        "type": "array",
        "items": RegName.json_schema(),
    },
    Tuple["Immutable", ...]: {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["tuple"]},
            "items": {
                "type": "array",
                "items": {"$ref": "#/$defs/immutable"},
            },
        },
        "required": ["type", "items"],
        "additionalProperties": False,
    },
    NotSupportedDataType: {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["notsupporteddatatype"]},
        },
        "required": ["type"],
        "additionalProperties": False,
    },
    NaturalCode: {"type": "string"},
    Immutable: {"$ref": "#/$defs/immutable"},
    Value: {"$ref": "#/$defs/value"},
}


PRIMITIVE_SCHEMA: JsonSchema = {"anyOf": [SCHEMA[t] for t in get_args(Primitive)]}
IMMUTABLE_SCHEMA: JsonSchema = {"anyOf": [SCHEMA[t] for t in get_args(Immutable)]}

SCHEMA |= {
    List["Immutable"]: {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["list"]},
            "items": {
                "type": "array",
                "items": {"$ref": "#/$defs/immutable"},
            },
        },
        "required": ["type", "items"],
        "additionalProperties": False,
    },
    Dict["Immutable", "Immutable"]: {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["dict"]},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"$ref": "#/$defs/immutable"},
                        "value": {"$ref": "#/$defs/immutable"},
                    },
                    "required": ["key", "value"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["type", "items"],
        "additionalProperties": False,
    },
    Set["Immutable"]: {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["set"]},
            "items": {
                "type": "array",
                "items": {"$ref": "#/$defs/immutable"},
            },
        },
        "required": ["type", "items"],
        "additionalProperties": False,
    },
    Object: {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["Object"]},
            "class": {"type": "string"},
            "attributes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": {"$ref": "#/$defs/immutable"},
                    },
                    "required": ["key", "value"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["type", "class", "attributes"],
        "additionalProperties": False,
    },
    Class: {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["Class"]},
            "name": {"type": "string"},
            "bases": {
                "type": "array",
                "items": Ref.json_schema(),
            },
            "annotations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    "required": ["key", "value"],
                    "additionalProperties": False,
                },
            },
            "attributes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": {"$ref": "#/$defs/immutable"},
                    },
                    "required": ["key", "value"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["type", "name", "bases", "annotations", "attributes"],
        "additionalProperties": False,
    },
    Func: {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["Func"]},
            "name": {"type": "string"},
            "full_func": {"type": "string"},
            "signature": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "annotation": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": [
                                "positional-only",
                                "positional or keyword",
                                "variadic positional",
                                "keyword-only",
                                "variadic keyword",
                            ],
                        },
                        "default": {"$ref": "#/$defs/immutable"},
                    },
                    "required": ["name", "annotation", "kind", "default"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["type", "name", "full_func", "signature"],
        "additionalProperties": False,
    },
}


VALUE_SCHEMA_TYPES: List[JsonSchemaValue] = [SCHEMA[t] for t in sorted(get_args(Value), key=lambda x: str(x))]
VALUE_SCHEMA_TYPES_NOFUNC: List[JsonSchemaValue] = [
    SCHEMA[t] for t in sorted(get_args(Value), key=lambda x: str(x)) if t != Func
]

VALUE_SCHEMA: JsonSchema = {"anyOf": VALUE_SCHEMA_TYPES}
VALUE_SCHEMA_NOFUNC: JsonSchema = {"anyOf": VALUE_SCHEMA_TYPES_NOFUNC}

SCHEMA_DEFS: JsonSchema = {"value": VALUE_SCHEMA, "immutable": IMMUTABLE_SCHEMA}
SCHEMA_DEFS_NOFUNC: JsonSchema = {"value": VALUE_SCHEMA_NOFUNC, "immutable": IMMUTABLE_SCHEMA}

EffectParams: TypeAlias = Value | RegName | List[RegName] | Variable | Label | NaturalCode


class Frame:
    def __init__(self, parent_fp: Optional[Ref]):
        self.parent_fp = parent_fp
        self.frame: Dict[Variable, Immutable] = {}

    def sync(self, other: "Frame"):
        self.parent_fp = other.parent_fp
        self.frame.update(other.frame)

    def insert(self, x: Variable, val: Immutable):
        self.frame[x] = val

    def get(self, x: Variable) -> Immutable:
        if x not in self.frame:
            raise GetFail(f"Variable {x} not found in frame")
        return self.frame[x]

    def __contains__(self, x: Variable) -> bool:
        return x in self.frame


def is_function(x):
    # Note: This might not be exhaustive.
    function_types = (
        types.FunctionType,
        types.MethodType,
        types.MethodWrapperType,
        types.MethodDescriptorType,
        types.BuiltinFunctionType,
        types.BuiltinMethodType,
        types.WrapperDescriptorType,
        types.LambdaType,
        types.ClassMethodDescriptorType,
        types.GetSetDescriptorType,
        types.MemberDescriptorType,
        Func,
    )

    return isinstance(x, function_types)
    # return hasattr(x, "__call__")


def is_function_type(x):
    # Note: This might not be exhaustive.
    function_types = (
        types.FunctionType,
        types.MethodType,
        types.MethodWrapperType,
        types.MethodDescriptorType,
        types.BuiltinFunctionType,
        types.BuiltinMethodType,
        types.WrapperDescriptorType,
        types.LambdaType,
        types.ClassMethodDescriptorType,
        types.GetSetDescriptorType,
        types.MemberDescriptorType,
        Func,
    )

    return x in function_types


class LLMUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cached_input_tokens: Optional[int] = None
    cached_output_tokens: Optional[int] = None

    @classmethod
    def from_openai_usage(cls, usage: openai.types.completion_usage.CompletionUsage) -> "LLMUsage":
        return cls(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cached_input_tokens=getattr(usage.prompt_tokens_details, "cached_tokens", None),
            cached_output_tokens=None,
        )

    @classmethod
    def from_openai_response_usage(cls, usage: ResponseUsage) -> "LLMUsage":
        return cls(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.input_tokens_details.cached_tokens,
            cached_output_tokens=None,
        )

    @classmethod
    def from_anthropic_usage(cls, usage: anthropic.types.Usage | BetaUsage) -> "LLMUsage":
        input_tokens = usage.input_tokens
        if usage.cache_read_input_tokens is not None:
            input_tokens += usage.cache_read_input_tokens
        if usage.cache_creation_input_tokens is not None:
            input_tokens += usage.cache_creation_input_tokens
        return cls(
            input_tokens=input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cache_read_input_tokens,
            cached_output_tokens=usage.cache_creation_input_tokens,
        )


class Argument(NamedTuple, Generic[T]):
    name: str
    arg: T


class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any]
    id: str


# This is a different model from ToolCall (which is used to LLM messages)
# Effect call is a convenient container for handling parsed effects that will not be parsed to LLMs
@dataclass
class EffectCall:
    name: str
    args: Sequence[Argument[EffectParams]]
    id: str

    def __hash__(self) -> int:
        return hash(self.name + ",".join([str(arg) for arg in self.args]))


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "tool"]
    time: Optional[float] = None

    @abstractmethod
    def to_openai(self) -> Dict[str, Any]: ...

    @abstractmethod
    def to_anthropic(self) -> Dict[str, Any]: ...


class UserMessage(ChatMessage):
    role: Literal["user"] = "user"
    content: str

    def to_openai(self) -> Dict[str, Any]:
        message = {"role": self.role, "content": self.content}
        return message

    def to_anthropic(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
        }


class AssistantMessage(ChatMessage):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    discarded_tool_calls: Optional[List[ToolCall]] = None
    usage: Optional[LLMUsage] = None

    def to_openai(self) -> Dict[str, Any]:
        if self.tool_calls is not None:
            return {
                "role": self.role,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.args)},
                    }
                    for call in self.tool_calls
                ],
            }
        else:
            return {"role": self.role, "content": self.content}

    def to_anthropic(self) -> Dict[str, Any]:
        if self.tool_calls is not None:
            return {
                "role": "assistant",
                "content": [
                    {"id": call.id, "type": "tool_use", "name": call.name, "input": call.args}
                    for call in self.tool_calls
                ],
            }
        else:
            return {
                "role": self.role,
                "content": self.content,
            }


class ToolMessage(ChatMessage):
    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str

    def to_openai(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content, "tool_call_id": self.tool_call_id}

    def to_anthropic(self) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": self.tool_call_id, "content": self.content}],
        }


ResponseType = TypeVar("ResponseType", bound=BaseModel)


@dataclass
class ResponseFormat(Generic[ResponseType]):
    res_schema: JsonSchema | type[ResponseType]
    name: str = "Output"

    def to_openai_schema(self) -> JsonSchema | type[ResponseType]:
        if isinstance(self.res_schema, Mapping):
            return self.res_schema
        else:
            # Given a BaseModel
            return self.res_schema
            # return {
            #     "type": "json_schema",
            #     "json_schema": {
            #         "name": self.name,
            #         "strict": True,
            #         "schema": self.res_schema.model_json_schema(),
            #     },
            # }

    def to_anthropic_schema(self) -> JsonSchema:
        # Need to use tool calls to enforce structured output
        name = self.name
        if isinstance(self.res_schema, Mapping):
            return {
                "name": name,
                "description": "Output schema of the response",
                "input_schema": self.res_schema["json_schema"]["schema"],  # type: ignore
                "cache_control": {"type": "ephemeral"},
            }
        else:
            return {
                "name": name,
                "description": "Output schema of the response",
                "input_schema": self.res_schema.model_json_schema(),
                "cache_control": {"type": "ephemeral"},
            }

    def parse(self, x: str) -> ResponseType:
        if isinstance(self.res_schema, Mapping):
            return json.loads(x)
        else:
            return self.res_schema.model_validate_json(x)

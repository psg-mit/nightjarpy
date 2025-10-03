import ast
import importlib
import inspect
import json
import logging
import os
import re
import types
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    ForwardRef,
    List,
    Literal,
    Mapping,
    Never,
    NoReturn,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeAlias,
    TypedDict,
    Union,
    cast,
)

import anthropic
import openai
import shortuuid
from pydantic import BaseModel
from tap.utils import type_to_str
from typing_inspect import get_args, get_origin

if TYPE_CHECKING:
    from nightjarpy.context import Context

from nightjarpy.effects import Effect, Parameter
from nightjarpy.types import (
    NJ_VAR_PREFIX,
    Argument,
    ChatMessage,
    Class,
    EffectCall,
    EffectError,
    EffectParams,
    Func,
    Immutable,
    JsonSchema,
    JsonSchemaValue,
    JsonType,
    Label,
    LLMUsage,
    NotSupportedDataType,
    Object,
    Param,
    Primitive,
    Ref,
    RegName,
    ResponseFormat,
    ResponseType,
    Signature,
    Success,
    ToolCall,
    Value,
    Variable,
)
from nightjarpy.utils.cache import Cache

NJ_BUILD_DIR = "nj__build"
MAX_SERIALIZE_LEN = 1000

logger = logging.getLogger(__name__)


@dataclass
class Telemetry:
    # Maps `filename.functionname` to LLM usage
    llm_usage: Dict[str, List[LLMUsage]] = field(default_factory=dict)
    n_tool_calls: int = 0
    trace: Optional[List[ChatMessage]] = None

    def reset(self):
        self.llm_usage = {}
        self.n_tool_calls = 0
        self.trace = None

    def log_llm_usage(self, filename: str, funcname: str, usage: LLMUsage):
        key = f"{filename}.{funcname}"
        self.llm_usage.setdefault(key, []).append(usage)

    def get_llm_usage(self, filename: str, funcname: str) -> List[LLMUsage]:
        key = f"{filename}.{funcname}"
        if key not in self.llm_usage:
            raise ValueError(f"Unknown file or function")
        return self.llm_usage[key]

    def total_llm_usage(self) -> LLMUsage:
        return sum_usage([sum_usage(x) for x in self.llm_usage.values()])

    def log_messages(self, messages: Sequence[ChatMessage]):
        if self.trace is None:
            self.trace = []
        self.trace.extend(messages)

    def dump_trace(self, trace_path: str, append: bool = True):
        if self.trace is None:
            logger.warning("No trace logged")
            return
        with open(trace_path, "a" if append else "w") as f:
            trace_json = [x.model_dump() for x in self.trace]

            f.write(json.dumps(trace_json) + "\n")


# Initialize global telemetry
NJ_TELEMETRY = Telemetry()


def sum_usage(usages: List[LLMUsage]) -> LLMUsage:
    """
    Sums a list of LLMUsage objects into a single LLMUsage object.
    If any cached token fields are None, they are treated as 0.
    """
    input_tokens = sum(u.input_tokens for u in usages)
    output_tokens = sum(u.output_tokens for u in usages)
    cached_input_tokens = sum((u.cached_input_tokens or 0) for u in usages)
    cached_output_tokens = sum((u.cached_output_tokens or 0) for u in usages)

    # If all cached_input_tokens are None, set to None, else use the sum
    if all(u.cached_input_tokens is None for u in usages):
        cached_input_tokens_val = None
    else:
        cached_input_tokens_val = cached_input_tokens

    if all(u.cached_output_tokens is None for u in usages):
        cached_output_tokens_val = None
    else:
        cached_output_tokens_val = cached_output_tokens

    return LLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens_val,
        cached_output_tokens=cached_output_tokens_val,
    )


disk_cache_dir = os.environ.get("NJ_CACHEDIR") or os.path.join(Path.home(), ".nj_cache")
disk_cache_limit = int(os.environ.get("NJ_CACHE_LIMIT", 3e10))

NJ_CACHE = Cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
    disk_cache_dir=disk_cache_dir,
    disk_size_limit_bytes=disk_cache_limit,
    memory_max_entries=1000000,
)


def with_cache(
    func: Callable,
    request_params: Dict,
):
    res = NJ_CACHE.get(request_params)
    if res is None:
        res = func(**request_params)
        NJ_CACHE.put(request_params, res)
    else:
        logger.info("Found request in cache...")
    return res


def extract_variable(natural_code: str) -> Tuple[Dict[Variable, str], Set[Variable]]:
    """
    Extracts variable definitions and references from a natural language code block.

    Args:
        natural_code: The natural language code as a string.

    Returns:
        A tuple containing:
            - A variable definition dictionary mapping variable names (defined with <:var> or <:var:type>) to their types (as strings).
            - A set of referenced variable names (with syntax <var>).

    Notes:
        - Skips over Python blocks enclosed in braces ({}), unless they are escaped.
        - Attribute references (e.g., <foo.bar>) are dropped, and only the object itself is kept (e.g. foo)
    """
    # Skip over Python blocks, but not escaped braces
    natural_code = re.sub(r"(?<!\\)\{.*?(?<!\\)\}", "", natural_code, flags=re.DOTALL)

    # Negative lookbehind (?<!\\) ensures we don't match escaped brackets
    def_pattern = re.compile(r"(?<!\\)<:(.*?)(?<!\\)>")
    ref_pattern = re.compile(r"(?<!\\)<([^:]*?)(?<!\\)>")

    def_vars: List[str] = def_pattern.findall(natural_code)
    ref_vars: List[str] = ref_pattern.findall(natural_code)

    # Need to filter this because of how parsing works for nested blocks
    ref_vars_set = set[Variable]()
    for var in ref_vars:
        if var not in ["natural", "/natural"]:
            if "." in var:
                name_split = var.split(".")
                name = name_split[0]
                ref_vars_set.add(Variable(name))
            else:
                ref_vars_set.add(Variable(var))

    def_vars_dict: Dict[Variable, str] = {}
    for var in def_vars:
        if ":" not in var:
            name, ty = var.strip(), "typing.Any"
        else:
            name, ty = [x.strip() for x in var.split(":")]

        # Check if the var is an attribute
        if "." in name:
            name_split = name.split(".")
            name = name_split[0]
            ty = "typing.Any"
        def_vars_dict[Variable(name)] = ty

    return def_vars_dict, ref_vars_set


def extract_label(natural_code: str) -> Set[Label]:
    # Skip over Python blocks, but not escaped braces
    natural_code = re.sub(r"(?<!\\)\{.*?(?<!\\)\}", "", natural_code, flags=re.DOTALL)

    # Negative lookbehind (?<!\\) ensures we don't match escaped brackets
    label_pattern = re.compile(r"(?<!\\)\|(.*?)(?<!\\)\|")
    labels: List[str] = label_pattern.findall(natural_code)

    return set([Label(x) for x in labels])


class VarGenerator:
    def __init__(self, init: int = 0):
        self._ids = count(init)

    def __call__(self):
        return f"{NJ_VAR_PREFIX}var{next(self._ids)}"

    def current_id(self):
        id = next(self._ids)
        self._ids = count(id)
        return id


def enable_nj_logging():
    """Enable logging for nightjarpy with console output."""
    # Get the nightjarpy logger
    logger = logging.getLogger("nightjarpy")
    logger.setLevel(logging.INFO)

    # Prevent propagation to root logger to avoid duplicates
    logger.propagate = False

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter with typical format
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)


def disable_nj_logging():
    """Disable logging for nightjarpy."""
    logger = logging.getLogger("nightjarpy")
    logger.setLevel(logging.WARNING)

    # Remove all handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


def to_strict_json_schema(json_schema: JsonSchema) -> JsonSchema:
    """
    Makes every "type": "object" schema strict by setting "additionalProperties": false.
    This includes schemas in the main schema and in the "defs" section.
    Any existing additionalProperties values (true or false) will be overridden to false.

    Args:
        json_schema: The input JSON schema

    Returns:
        A new JSON schema with additionalProperties: false set for all object schemas
    """

    def make_strict_recursive(schema: JsonSchemaValue) -> JsonSchemaValue:
        """Recursively process schema values to set additionalProperties: false for all objects."""
        if isinstance(schema, dict):
            # Create a copy of the dictionary
            result = schema.copy()

            # If this is an object schema, set additionalProperties: false (override any existing value)
            if result.get("type") == "object":
                result["additionalProperties"] = False

            # Recursively process all values in the dictionary
            for key, value in result.items():
                result[key] = make_strict_recursive(value)

            return result
        elif isinstance(schema, list):
            # Recursively process all items in the list
            return [make_strict_recursive(item) for item in schema]
        else:
            # For strings, booleans, and other primitive types, return as-is
            return schema

    # Process the main schema
    result_schema = make_strict_recursive(json_schema)

    # Ensure we return a JsonSchema (which is a Dict[str, JsonSchemaValue])
    if not isinstance(result_schema, dict):
        raise ValueError("Invalid JSON schema: root must be a dictionary")

    return result_schema


def serialize_json(x: EffectParams | Success | EffectError) -> str:
    def _serialize_dict(x: Value) -> JsonType:
        if isinstance(x, Ref):
            result = x.json_value()
        elif isinstance(x, datetime):
            result = {"type": "datetime", "value": x.isoformat()}
        elif isinstance(x, (types.NoneType, str, int, float, bool)):
            result = x
        elif isinstance(x, NotSupportedDataType):
            result = {"type": "notsupporteddatatype"}
        elif isinstance(x, tuple):
            result = {
                "type": "tuple",
                "items": [_serialize_dict(v) for v in x],
            }
        elif isinstance(x, list):
            result = {
                "type": "list",
                "items": [_serialize_dict(v) for v in x],
            }
        elif isinstance(x, dict):
            result = {
                "type": "dict",
                "items": [
                    {
                        "key": _serialize_dict(k),
                        "value": _serialize_dict(v),
                    }
                    for k, v in sorted(list(x.items()), key=lambda x: str(x[0]))
                ],
            }
        elif isinstance(x, set):
            result = {
                "type": "set",
                "items": [_serialize_dict(v) for v in sorted(list(x), key=lambda x: str(x))],
            }
        # elif hasattr(x, "nj_to_json"):
        #     # Use custom serialization
        #     result = getattr(x, "nj_to_json")()
        elif isinstance(x, Object):
            result = {
                "type": "Object",
                "class": x._class,
                "attributes": [
                    {"key": _serialize_dict(k), "value": _serialize_dict(v)}
                    for k, v in sorted(list(x.attributes.items()), key=lambda x: str(x[0]))
                ],
            }
        elif isinstance(x, Class):
            result = {
                "type": "Class",
                "name": x.name,
                "bases": [_serialize_dict(b) for b in x.bases],
                "annotations": [
                    {"key": k, "value": v} for k, v in sorted(list(x.annotations.items()), key=lambda x: str(x[0]))
                ],
                "attributes": [
                    {"key": _serialize_dict(k), "value": _serialize_dict(v)}
                    for k, v in sorted(list(x.attributes.items()), key=lambda x: str(x[0]))
                ],
            }
        elif isinstance(x, Func):
            body = x.full_func
            # if len(body) > MAX_SERIALIZE_LEN:
            #     body = body[:MAX_SERIALIZE_LEN] + "..."
            return {
                "type": "Func",
                "name": x.name,
                "full_func": body,
                "signature": [
                    {
                        "name": p.name,
                        "annotation": p.annotation,
                        "kind": p.kind,
                        "default": _serialize_dict(p.default),
                    }
                    for p in x.signature.params
                ],
            }
        else:
            raise ValueError("Unsupported data type")
        return result

    if isinstance(x, Success):
        return "Success"
    if isinstance(x, EffectError):
        return str(x)
    if isinstance(x, RegName):
        return json.dumps(x.json_value())
    if isinstance(x, Label):
        return json.dumps(x.json_value())
    if isinstance(x, Variable):
        return json.dumps(x.json_value())
    if isinstance(x, list) and all(isinstance(v, RegName) for v in x):
        # List[RegName]
        return json.dumps([cast(RegName, v).json_value() for v in x])

    return json.dumps(_serialize_dict(x))  # type: ignore


def deserialize_json(json: JsonType, context: "Context") -> EffectParams:
    def _deserialize_immutable(json: JsonType) -> Immutable:
        if isinstance(json, (types.NoneType, str, int, float, bool)):
            return json
        elif isinstance(json, Dict):
            if "type" not in json:
                raise ValueError("Unknown value type")
            elif json["type"] == "Ref":
                return Ref.from_json(json)
            elif json["type"] == "notsupporteddatatype":
                return NotSupportedDataType()
            elif json["type"] == "datetime":
                if "value" not in json or not isinstance(json["value"], str):
                    raise ValueError("Unexpected serialization")
                return datetime.fromisoformat(json["value"])
            elif json["type"] == "tuple":
                if "items" not in json or not isinstance(json["items"], list):
                    raise ValueError("Unexpected serialization")
                return tuple([_deserialize_immutable(x) for x in json["items"]])
            else:
                raise ValueError("Expected immutable type")
        else:
            raise ValueError("Expected immutable type")

    def _deserialize_value(json: JsonType) -> Value:
        if isinstance(json, (types.NoneType, str, int, float, bool)):
            return _deserialize_immutable(json)
        elif isinstance(json, Sequence):
            raise ValueError("Unexpected serialization")
            # return [_deserialize_value(x) for x in json]
            # return [Ref.from_json(x) for x in json]
        else:
            if "type" not in json:
                raise ValueError("Unknown value type")

            if json["type"] == "datetime":
                return _deserialize_immutable(json)
            if json["type"] == "tuple":
                return _deserialize_immutable(json)
            if json["type"] == "notsupporteddatatype":
                return NotSupportedDataType()
            if json["type"] == "list":
                if "items" not in json or not isinstance(json["items"], list):
                    raise ValueError("Unexpected serialization")
                v = [_deserialize_immutable(x) for x in json["items"]]
                return list(v)
            elif json["type"] == "dict":
                if "items" not in json or not isinstance(json["items"], list):
                    raise ValueError("Unexpected serialization")
                if any(not isinstance(x, dict) or "key" not in x or "value" not in x for x in json["items"]):
                    raise ValueError("Unexpected serialization")
                return {
                    _deserialize_immutable(x["key"]): _deserialize_immutable(x["value"])
                    for x in json["items"]
                    if isinstance(x, dict)
                }
            elif json["type"] == "set":
                if "items" not in json or not isinstance(json["items"], list):
                    raise ValueError("Unexpected serialization")
                return set([_deserialize_immutable(x) for x in json["items"]])
            elif json["type"] == "Object":
                if (
                    "class" not in json
                    or not isinstance(json["class"], str)
                    or "attributes" not in json
                    or not isinstance(json["attributes"], list)
                ):
                    raise ValueError("Unexpected serialization")

                d: Dict[str, Immutable] = {}
                for x in json["attributes"]:
                    if not isinstance(x, dict) or not "key" in x or not "value" in x or not isinstance(x["key"], str):
                        raise ValueError("Unexpected serialization")
                    d[x["key"]] = _deserialize_immutable(x["value"])

                return Object(_class=json["class"], attributes=d)
            elif json["type"] == "Class":
                if (
                    "name" not in json
                    or not isinstance(json["name"], str)
                    or "bases" not in json
                    or not isinstance(json["bases"], list)
                    or "annotations" not in json
                    or not isinstance(json["annotations"], list)
                    or "attributes" not in json
                    or not isinstance(json["attributes"], list)
                ):
                    raise ValueError("Unexpected serialization")

                bases: List[Ref] = []
                for x in json["bases"]:
                    if not isinstance(x, Dict):
                        raise ValueError("Unexpected serialization")
                    bases.append(Ref.from_json(x))

                annotations: Dict[str, str] = {}
                for x in json["annotations"]:
                    if (
                        not isinstance(x, dict)
                        or not "key" in x
                        or not "value" in x
                        or not isinstance(x["key"], str)
                        or not isinstance(x["value"], str)
                    ):
                        raise ValueError("Unexpected serialization")
                    annotations[x["key"]] = x["value"]

                attributes: Dict[str, Immutable] = {}
                for x in json["attributes"]:
                    if not isinstance(x, dict) or not "key" in x or not "value" in x or not isinstance(x["key"], str):
                        raise ValueError("Unexpected serialization")
                    attributes[x["key"]] = _deserialize_immutable(x["value"])

                return Class(name=json["name"], bases=tuple(bases), attributes=attributes, annotations=annotations)
            elif json["type"] == "Func":
                if (
                    "name" not in json
                    or not isinstance(json["name"], str)
                    or "full_func" not in json
                    or not isinstance(json["full_func"], str)
                    or "signature" not in json
                    or not isinstance(json["signature"], list)
                ):
                    raise ValueError("Unexpected serialization")

                params: List[Param] = []
                for p in json["signature"]:
                    if (
                        not isinstance(p, dict)
                        or "name" not in p
                        or not isinstance(p["name"], str)
                        or "annotation" not in p
                        or not isinstance(p["annotation"], str)
                        or "kind" not in p
                        or p["kind"]
                        not in (
                            "positional-only",
                            "positional or keyword",
                            "variadic positional",
                            "keyword-only",
                            "variadic keyword",
                        )
                        or "default" not in p
                    ):
                        raise ValueError("Unexpected serialization")
                    params.append(
                        Param(
                            name=p["name"],
                            annotation=p["annotation"],
                            kind=p["kind"],
                            default=_deserialize_immutable(p["default"]),
                        )
                    )

                return Func(
                    context=context,
                    name=json["name"],
                    full_func=json["full_func"],
                    signature=Signature(
                        tuple(params),
                    ),
                )
            elif json["type"] == "Ref":
                return _deserialize_immutable(json)
            else:
                raise ValueError("Unknown value type")

    if isinstance(json, Dict) and "type" in json and json["type"] == "Register":
        return RegName.from_json(json)
    if isinstance(json, List):
        # List of registers
        return [RegName.from_json(x) for x in json]
    else:
        return _deserialize_value(json)


def serialize(x: EffectParams | Success | EffectError, is_string: bool = True) -> str:
    """
    Serialize a Value, Success, or EffectError to using format of func_name(args1, arg2, ...)
    """

    def _serialize_value(v: Value) -> str:
        # Immutable types
        if v is None:
            return "None"
        elif isinstance(v, bool):
            # bool must come before int since bool is a subclass of int
            return str(v)
        elif isinstance(v, int):
            return str(v)
        elif isinstance(v, float):
            return str(v)
        elif isinstance(v, str):
            if not is_string:
                return v
            return repr(v)
        elif isinstance(v, Ref):
            return f"Ref({v.addr})"
        elif isinstance(v, datetime):
            return f"datetime({v.isoformat()})"
        elif isinstance(v, tuple):
            if len(v) == 0:
                return "()"
            elif len(v) == 1:
                return f"({_serialize_value(v[0])},)"
            else:
                items = ", ".join(_serialize_value(item) for item in v)
                return f"({items})"
        # Mutable types
        elif isinstance(v, list):
            if len(v) == 0:
                return "[]"
            items = ", ".join(_serialize_value(item) for item in v)
            return f"[{items}]"
        elif isinstance(v, dict):
            if len(v) == 0:
                return "{}"
            items = ", ".join(f"{_serialize_value(k)}: {_serialize_value(val)}" for k, val in v.items())
            return f"{{{items}}}"
        elif isinstance(v, set):
            if len(v) == 0:
                return "set()"
            items = ", ".join(_serialize_value(item) for item in sorted(v, key=lambda x: str(x)))
            return f"{{{items}}}"
        elif isinstance(v, Object):
            attr_str = _serialize_value(v.attributes)  # type: ignore
            return f"Object[{v._class}]({attr_str})"
        elif isinstance(v, Class):
            # Class[MyClass]((Ref(1),Ref(81)), {"attr_1": str, age: int}, {class_attr1: "alice", test: None})

            bases_str = _serialize_value(v.bases)  # type: ignore

            annotations_str = _serialize_value(v.annotations)  # type: ignore
            attributes_str = _serialize_value(v.attributes)  # type: ignore

            return f"Class[{v.name}]({bases_str}, {annotations_str}, {attributes_str})"
        else:
            raise ValueError(f"Unsupported data type: {type(v)}")

    if isinstance(x, Success):
        return "Success"
    if isinstance(x, EffectError):
        return str(x)
    if isinstance(x, RegName):
        return x.name
    if isinstance(x, list) and all(isinstance(v, RegName) for v in x):
        # List[RegName]
        return f"[{",".join([cast(RegName, v).name for v in x])}]"
    if isinstance(x, Label):
        return x.name
    if isinstance(x, Variable):
        return x.name
    else:
        return _serialize_value(x)  # type:ignore


def deserialize(encoded_x: str) -> EffectParams:
    """
    Deserialize a string representation back to a Value, List[RegName], or RegName.
    Parses the schema format from prompts.py.
    """
    encoded_x = encoded_x.strip()

    def _split_comma_items(s: str) -> List[str]:
        """Split comma-separated items, respecting nesting and strings."""
        items = []
        current = []
        depth = 0
        in_string = False
        string_char = None

        for i, char in enumerate(s):
            if char in ('"', "'") and (i == 0 or s[i - 1] != "\\"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None

            if not in_string:
                if char in "([{":
                    depth += 1
                elif char in ")]}":
                    depth -= 1
                elif char == "," and depth == 0:
                    items.append("".join(current).strip())
                    current = []
                    continue

            current.append(char)

        if current:
            items.append("".join(current).strip())

        return items

    def _split_key_value(s: str) -> List[str]:
        """Split a key:value pair, respecting nesting and strings."""
        depth = 0
        in_string = False
        string_char = None
        for i, char in enumerate(s):
            if char in ('"', "'") and (i == 0 or s[i - 1] != "\\"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None

            if not in_string:
                if char in "([{":
                    depth += 1
                elif char in ")]}":
                    depth -= 1
                elif char == ":" and depth == 0:
                    return [s[:i].strip(), s[i + 1 :].strip()]
        raise ValueError(f"No colon found in dict item: {s}")

    def _parse_immutable_or_register(s: str) -> Immutable | RegName:
        """Parse an immutable value."""
        s = s.strip()

        # Handle None
        if s == "None":
            return None

        # Handle boolean
        if s in ("True", "False"):
            return s == "True"

        # Handle Ref
        if s.startswith("Ref"):
            match = re.match(r"Ref\s*\(\s*(\d+)\s*\)", s)
            if match:
                return Ref(addr=int(match.group(1)))
            raise ValueError(f"Invalid Ref format: {s}")

        # Handle datetime
        if s.startswith("datetime"):
            match = re.match(r"datetime\s*\(\s*(.+?)\s*\)", s)
            if match:
                return datetime.fromisoformat(match.group(1))
            raise ValueError(f"Invalid datetime format: {s}")

        # Handle tuple
        if s.startswith("(") and s.endswith(")"):
            items_str = s[1:-1].strip()
            if not items_str:
                return ()
            # Handle single element tuple
            if items_str.endswith(","):
                return (_parse_immutable(items_str[:-1].strip()),)
            items = _split_comma_items(items_str)
            return tuple(_parse_immutable(item) for item in items)

        # Try to parse as a Python literal (int, float, str)
        try:
            return ast.literal_eval(s)
        except (SyntaxError, ValueError):
            # Assume is a register
            # Note: it can also be a variable, but let type validate do the conversion to Variable
            return RegName(s)

    def _parse_immutable(s: str) -> Immutable:
        res = _parse_immutable_or_register(s)
        if isinstance(res, RegName):
            # expected only immutable values, so registers are actually malformed values
            raise ValueError(f"Unable to parse immutable value: {s}")
        return res

    def _parse_dict(s: str, string_keys: bool = False) -> Dict:
        """Parse a dictionary string. If string_keys=True, enforces string keys."""
        s = s.strip()
        if not (s.startswith("{") and s.endswith("}")):
            raise ValueError(f"Invalid dict format: {s}")

        # Check if empty (with or without whitespace)
        content = s[1:-1].strip()
        if not content:
            return {}

        items = _split_comma_items(s[1:-1])
        result = {}

        for item in items:
            # Skip empty items
            if not item.strip():
                continue

            key_val = _split_key_value(item)
            if len(key_val) != 2:
                raise ValueError(f"Invalid dict item: {item}")
            key_str, val_str = key_val

            key = _parse_immutable(key_str)
            if string_keys and not isinstance(key, str):
                raise ValueError(f"Expected string key, got {type(key)}")
            val = _parse_immutable(val_str)
            result[key] = val

        return result

    def _parse_object(s: str) -> Object:
        """Parse Object format: Object[ClassName]({...})"""
        match = re.match(r"Object\s*\[\s*([^\]]+?)\s*\]\s*\(\s*(.+?)\s*\)$", s, re.DOTALL)
        if not match:
            raise ValueError(f"Invalid Object format: {s}")

        class_name = match.group(1).strip()
        attrs_str = match.group(2).strip()

        # Parse dict handles empty dicts with whitespace
        attributes = _parse_dict(attrs_str, string_keys=True)
        return Object(_class=class_name, attributes=attributes)

    def _parse_class(s: str) -> Class:
        """Parse Class format: Class[Name]((bases...), {annotations...}, {attributes...})"""
        match = re.match(r"Class\s*\[\s*([^\]]+?)\s*\]\s*\(\s*(.+?)\s*\)$", s, re.DOTALL)
        if not match:
            raise ValueError(f"Invalid Class format: {s}")

        class_name = match.group(1).strip()
        rest = match.group(2).strip()

        # Split into bases, annotations, attributes at top level
        parts = []
        current = []
        depth = 0
        for char in rest:
            if char in "([{":
                depth += 1
                current.append(char)
            elif char in ")]}":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            parts.append("".join(current).strip())

        if len(parts) != 3:
            raise ValueError(f"Invalid Class format: expected 3 parts, got {len(parts)}")

        bases_str, annotations_str, attributes_str = parts

        # Parse bases (tuple of Refs)
        bases = []
        if bases_str.strip() not in ["()", "(,)"]:
            bases_tuple = _parse_immutable(bases_str)
            if not isinstance(bases_tuple, tuple):
                raise ValueError(f"Expected tuple for bases, got {type(bases_tuple)}")
            for item in bases_tuple:
                if not isinstance(item, Ref):
                    raise ValueError(f"Expected Ref in bases, got {type(item)}")
                bases.append(item)

        # Parse annotations (dict of str -> str)
        annot_content = annotations_str.strip()
        annotations = _parse_dict(annot_content, string_keys=True)

        # Parse attributes (dict of str -> Immutable)
        # _parse_dict now handles empty dicts with whitespace
        attributes = _parse_dict(attributes_str, string_keys=True)

        return Class(name=class_name, bases=tuple(bases), annotations=annotations, attributes=attributes)

    def _parse_value(s: str) -> EffectParams:
        """Parse any Value type."""
        s = s.strip()

        # Check for mutable types first (they have specific prefixes)
        if s.startswith("Object"):
            return _parse_object(s)

        if s.startswith("Class"):
            return _parse_class(s)

        # Handle set
        if re.match(r"set\s*\(\s*\)", s):
            return set()

        if s.startswith("{"):
            # Check if empty first - {} is always a dict, not a set
            content = s[1:-1].strip()
            if not content:
                return {}

            # Distinguish between set and dict by checking for colons at depth 0
            if ":" not in s.split("}")[0]:
                # It's a set
                items = _split_comma_items(content)
                # Filter out empty items
                return set(_parse_immutable(item) for item in items if item.strip())
            # It's a dict
            return _parse_dict(s, string_keys=False)

        # Handle list
        if s.startswith("["):
            content = s[1:-1].strip()
            if not content:
                return []
            items = _split_comma_items(s[1:-1])
            items = [_parse_immutable_or_register(item) for item in items if item.strip()]
            if any(isinstance(x, RegName) for x in items) and not all(isinstance(x, RegName) for x in items):
                raise ValueError(f"Unable to parse list value {content}")
            return cast(List[RegName] | List[Immutable], items)

        # Everything else is immutable
        return _parse_immutable_or_register(s)

    return _parse_value(encoded_x)


# --- Maps for simple/builtin types ---
_BUILTIN_TYPE_TO_NAME = {
    int: "int",
    float: "float",
    bool: "bool",
    str: "str",
    bytes: "bytes",
    complex: "complex",
    list: "list",
    tuple: "tuple",
    dict: "dict",
    set: "set",
    frozenset: "frozenset",
    type(None): "NoneType",
    object: "object",
    datetime: "datetime",
}

_NAME_TO_BUILTIN_TYPE = {v: k for k, v in _BUILTIN_TYPE_TO_NAME.items()}

# Namespace allowed when parsing generic annotations like list[int] or dict[str, int]
_SAFE_EVAL_NS = {
    **_NAME_TO_BUILTIN_TYPE,
    # typing helpers
    "Any": Any,
    "Optional": Optional,
    "Union": Union,
    "Literal": Literal,
    "Annotated": Annotated,
    "Callable": Callable,
    "ClassVar": ClassVar,
    "Final": Final,
    "Never": Never,
    "NoReturn": NoReturn,
    "Type": Type,
    "TypedDict": TypedDict,
    # Capitalized typing generics for backwards compatibility
    "List": List,
    "Dict": Dict,
    "Set": Set,
    "Tuple": Tuple,
    "Sequence": Sequence,
    "Mapping": Mapping,
    "ForwardRef": ForwardRef,
}


def type_to_string(tp: type) -> str:
    """
    Convert a Python type object into a round-trippable string.
    - Builtins -> 'int', 'list', ...
    - Typing/generics -> uses the canonical repr (e.g., 'list[int]', 'dict[str, int]', 'int | None')
    - User classes -> 'package.module:QualName'
    """
    # Builtins and common types
    if tp in _BUILTIN_TYPE_TO_NAME:
        return _BUILTIN_TYPE_TO_NAME[tp]

    # typing / PEP 585 / PEP 604 generics: their repr is already parseable with our safe eval
    # Examples: list[int], dict[str, int], tuple[int, ...], int | None, typing.Union[int, str]
    try:
        origin = get_origin(tp)
        args = get_args(tp)
        if origin is not None or "|" in repr(tp) or repr(tp).startswith("typing."):
            return repr(tp).replace("typing.", "")  # normalize to shorter form
    except Exception:
        pass

    # Fallback: user-defined class -> fully qualified name "module:QualName"
    mod = getattr(tp, "__module__", None)
    qn = getattr(tp, "__qualname__", None)
    if mod and qn:
        return f"{mod}:{qn}"

    # Last resort: simple name
    return getattr(tp, "__name__", repr(tp))


def string_to_type(s: str, classes: dict[str, type]) -> type:
    """
    Parse a string produced by type_to_string back into a Python type object.
    Supports:
      - Builtins: 'int', 'list', ...
      - Generics/typing: 'list[int]', 'dict[str, int]', 'tuple[int, ...]', 'int | None', 'Union[int, str]'
      - User classes: 'package.module:QualName'
      - Custom classes in generics: 'List[package.module:QualName]', 'List[package.module.QualName]', 'dict[str, package.module:QualName]'
      - Placeholders: Strings containing '<run_path>' are treated as Any (cannot be resolved without runtime context)

    Args:
        s: The type string to parse
    """
    s = s.strip().replace("<run_path>.", "")
    if s == "":
        return Any  # type: ignore[return-value]

    # Builtins
    if s in _NAME_TO_BUILTIN_TYPE:
        return _NAME_TO_BUILTIN_TYPE[s]

    # Module-qualified user class, format "pkg.mod:QualName"
    if ":" in s and not any(ch in s for ch in "[]()|,"):
        module_name, qualname = s.split(":", 1)
        mod = importlib.import_module(module_name)
        obj = mod
        for part in qualname.split("."):
            obj = getattr(obj, part)
        return obj  # type: ignore[return-value]

    # Handle generics with custom classes inside
    # e.g., "List[package.module:QualName]" or "List[package.module.QualName]" or "dict[str, package.module:QualName]"
    if "[" in s:
        # Check if this might contain custom classes
        # Pattern 1: module:QualName format
        # Pattern 2: module.QualName format where it's a custom class (not a builtin like typing.List)
        import re

        # First, try to detect custom classes in either format
        # We need to distinguish between module paths and generic type names
        # Strategy: look for patterns inside brackets that look like module paths

        custom_classes = {}
        counter = [0]  # Use list to allow modification in nested function

        def replace_custom_class(match):
            full_match = match.group(0)

            # Check if this is a colon-separated format
            if ":" in full_match:
                module_name, qualname = full_match.split(":", 1)
            else:
                # Dot-separated format - need to figure out where module ends and class begins
                # Heuristic: try each possible split point
                parts = full_match.split(".")

                # Try to import progressively longer module paths
                module_name = None
                qualname = None
                for i in range(len(parts) - 1, 0, -1):
                    try_module = ".".join(parts[:i])
                    try_qualname = ".".join(parts[i:])
                    try:
                        importlib.import_module(try_module)
                        module_name = try_module
                        qualname = try_qualname
                        break
                    except (ImportError, ModuleNotFoundError):
                        continue

                if module_name is None:
                    # Could not determine module, return unchanged
                    return full_match

            # Ensure we have both module_name and qualname
            if module_name is None or qualname is None:
                return full_match

            # Create a unique placeholder name
            placeholder = f"_CustomClass_{counter[0]}_"
            counter[0] += 1

            # Load the actual class
            try:
                mod = importlib.import_module(module_name)
                obj = mod
                for part in qualname.split("."):
                    obj = getattr(obj, part)
                custom_classes[placeholder] = obj
                return placeholder
            except (ImportError, ModuleNotFoundError, AttributeError):
                # Could not load, return unchanged
                return full_match

        # Pattern to match potential custom class references
        # Match sequences like: word.word.Word or word.word:Word
        # But only when they're likely to be custom classes (contain dots or colons)
        pattern = r"\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*[.:][A-Z][a-zA-Z0-9_.]*)\b"

        # Also match dot-separated paths that might be classes (start lowercase, end uppercase)
        pattern_dotted = r"\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*\.[A-Z][a-zA-Z0-9_]*(?:\.[A-Z][a-zA-Z0-9_]*)*)\b"

        modified_s = s

        # First pass: handle colon format
        modified_s = re.sub(pattern, replace_custom_class, modified_s)

        # Second pass: handle dot format
        modified_s = re.sub(pattern_dotted, replace_custom_class, modified_s)

        # If we found any custom classes, evaluate with extended namespace
        if custom_classes:
            try:
                eval_ns = {**_SAFE_EVAL_NS, **custom_classes}
                return eval(modified_s, {"__builtins__": {}}, eval_ns)  # type: ignore[return-value]
            except Exception:
                if modified_s in classes:
                    return classes[modified_s]
                logger.warning(f"Type {s} is not valid, assuming as Any")
                return Any  # type: ignore[return-value]

    # Typing/generics: evaluate in a restricted namespace
    # Allow PEP 604 unions (e.g., int | None) and PEP 585 generics (list[int])
    try:
        return eval(s, {"__builtins__": {}}, _SAFE_EVAL_NS)  # type: ignore[return-value]
    except Exception:
        if modified_s in classes:
            return classes[modified_s]
        logger.warning(f"Type {s} is not valid, assuming as Any")
        return Any  # type: ignore[return-value]


def get_object_attributes(obj: Any) -> Set[str]:
    """
    Get all attribute names from an object or type, handling both __dict__ and __slots__.

    Objects with __slots__ don't have __dict__ by default, but some classes may have
    both if __dict__ is explicitly included in __slots__ or through inheritance.
    For types, checks the type's own __slots__ if present.

    Also includes attributes from dir() to capture attributes that may not be in __dict__
    or __slots__ (e.g., ValueError.args).
    """
    attr_names: Set[str] = set()

    # Determine which object to check for __slots__
    # For instances, check the class; for types, check the type itself
    if isinstance(obj, type):
        # It's a type/class, check its own __slots__ and __dict__
        slots_source = obj
    else:
        # It's an instance, check the class's __slots__
        slots_source = obj.__class__

    # Handle __slots__
    slots = getattr(slots_source, "__slots__", None)
    if slots is not None:
        # __slots__ can be a tuple, string, or other iterable
        if isinstance(slots, str):
            attr_names.add(slots)
        elif isinstance(slots, (tuple, list)):
            for slot in slots:
                if isinstance(slot, str):
                    attr_names.add(slot)
        elif isinstance(slots, dict):
            # Sometimes __slots__ is stored as a dict
            attr_names.update(slots.keys())

    # Handle __dict__ if it exists on the actual object
    if hasattr(obj, "__dict__"):
        try:
            # Access __dict__ directly, but catch errors for objects with slots
            dict_obj = obj.__dict__
            if dict_obj is not None:
                attr_names.update(dict_obj.keys())
        except (AttributeError, TypeError):
            # Objects with __slots__ and no __dict__ in slots will raise AttributeError
            # when trying to access __dict__ directly
            pass

    # Also include attributes from dir() to capture any other attributes
    # that might not be in __dict__ or __slots__ (e.g., exception.args)
    try:
        dir_attrs = dir(obj)
        attr_names.update(dir_attrs)
    except (AttributeError, TypeError):
        # If dir() fails for some reason, just skip it
        pass

    return attr_names


# TODO: Only supports kwargs
def call_function_by_sig(func: Callable, kwargs: Dict[str, Any]) -> Any:
    sig = inspect.signature(func)

    args = kwargs.get("args", ())
    if "args" in kwargs:
        kwargs.pop("args")

    # Manually bind arguments, ignoring extra values in args/kwargs
    bound_args = inspect.BoundArguments(sig, OrderedDict())

    # Find special parameter kinds
    var_positional = None
    var_keyword = None
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            var_positional = p.name
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            var_keyword = p.name

    # Add positional arguments (limited to parameters that accept them)
    pos_params = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    args_to_apply = args[: len(pos_params)] if args else ()
    for i, arg in enumerate(args_to_apply):
        if i < len(pos_params):
            bound_args.arguments[pos_params[i].name] = arg

    # If there's a *args parameter, collect remaining positional arguments
    if var_positional and args and len(args) > len(pos_params):
        bound_args.arguments[var_positional] = tuple(args[len(pos_params) :])

    param_names = set(p.name for p in sig.parameters.values())
    # Add keyword arguments for parameters that exist in the signature
    extra_kwargs = {}
    for k, v in kwargs.items():
        if k in param_names:
            bound_args.arguments[k] = v
        elif var_keyword:
            # If there's a **kwargs parameter, collect extra keyword arguments
            extra_kwargs[k] = v

    if var_keyword and extra_kwargs:
        bound_args.arguments[var_keyword] = extra_kwargs

    # Manually fill in any remaining default values that bind_partial might leave out
    for param in sig.parameters.values():
        if param.name not in bound_args.arguments and param.default is not param.empty:
            bound_args.arguments[param.name] = param.default

    try:
        return func(*bound_args.args, **bound_args.kwargs)
    except Exception:
        return func(*bound_args.args)


def openai_schema_to_function_schema(schema: ResponseFormat[ResponseType]) -> Dict[str, Any]:
    """Convert ResponseFormat to OpenAI function schema."""
    if isinstance(schema.res_schema, Mapping):
        # Already a JSON schema dict
        json_schema = schema.res_schema["json_schema"]["schema"]  # type: ignore
    else:
        # Pydantic model
        json_schema = schema.res_schema.model_json_schema()
        json_schema = to_strict_json_schema(json_schema)

    return {
        "type": "function",
        "strict": True,
        "function": {
            "name": schema.name,
            "parameters": json_schema,
            "strict": True,
        },
    }


def serialize_effect(effects: List[EffectCall], effect_mapping: Dict[str, Effect]) -> str:
    effects_str = []
    for effect in effects:
        serialized_args = []
        effect_params = {k: v for k, v, _ in effect_mapping[effect.name].parameters}
        for k, v in effect.args:
            if effect_params[k] in (Variable, Label):
                serialized_args.append(serialize(v, is_string=False))
            else:
                serialized_args.append(serialize(v))
        args_str = ",".join(serialized_args)
        effects_str.append(f"{effect.name}({args_str})")

    return "\n".join(effects_str)


def extract_effects(raw_str: str) -> List[ToolCall]:
    effects = [
        ToolCall(name=x.strip(), args={}, id=str(shortuuid.uuid()))
        for x in raw_str.split("\n")
        if x.strip() != "" and re.match(r"(\w+)\((.*)\)$", x) is not None
    ]
    return effects


def parse_effect(effect_str: str, effect_mapping: Dict[str, Effect]) -> EffectCall:
    """Parse effect tool call from string format to EffectCall object. Arguments are mapped to the parameter names given in the schema, but their types are not validated.

    Expected format:
        lookup(x_reg, x)
        store_reg(dest, Ref(39))
        compute("Add the two values", sum_reg, [y_reg, x10_valreg])
        done()
    """

    # Parse function call: function_name(arg1, arg2, ...)
    match = re.match(r"(\w+)\((.*)\)$", effect_str)
    if not match:
        raise ValueError(
            f"Invalid tool call format: {effect_str}. Make sure you're using the expected tool call format in your response: func_name(arg1, arg2, ...)"
        )

    func_name = match.group(1)
    args_str = match.group(2).strip()

    if func_name not in effect_mapping:
        raise ValueError(f"Effect {func_name} not found")

    # Parse arguments
    parsed_args = _parse_arguments(args_str) if args_str else []

    parameters = effect_mapping[func_name].parameters
    mapped_args = validate_args(parsed_args, parameters)

    return EffectCall(name=func_name, args=mapped_args, id=str(shortuuid.uuid()))


def _parse_arguments(args_str: str) -> List[EffectParams]:
    """Parse comma-separated arguments"""
    args: List[EffectParams] = []
    current = ""
    depth = 0
    in_string = False
    string_char = None

    for i, char in enumerate(args_str):
        if char in ('"', "'") and (i == 0 or args_str[i - 1] != "\\"):
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = None

        if not in_string:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char == "," and depth == 0:
                args.append(deserialize(current.strip()))
                current = ""
                continue

        current += char

    if current.strip():
        args.append(deserialize(current.strip()))

    return args


def _check_isinstance_generic(value: Any, expected_type: Type | types.UnionType | ForwardRef) -> bool:
    """
    Check if a value is an instance of the expected type, handling generics.

    Args:
        value: The value to check
        expected_type: The expected type, which may be a generic like List[RegName]

    Returns:
        True if the value matches the expected type, False otherwise
    """
    # Handle forward references (string type annotations)
    if isinstance(expected_type, str):
        # Resolve string forward references to actual types
        if expected_type == "Immutable":
            return _check_isinstance_generic(value, Immutable)
        # For other forward references, we can't resolve them without more context
        # Be permissive for now
        return True

    # Handle ForwardRef objects
    if isinstance(expected_type, ForwardRef):
        # Try to resolve the forward reference
        if expected_type.__forward_arg__ == "Immutable":
            return _check_isinstance_generic(value, Immutable)
        # For other forward references, we can't resolve them without more context
        # Be permissive for now
        return True

    # Get the origin and args of the expected type
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    # If it's not a generic type, use regular isinstance
    if origin is None:
        try:
            return isinstance(value, expected_type)
        except TypeError:
            # Handle cases where expected_type is not a valid isinstance target
            return False

    # Handle List[T]
    if origin is list:
        if not isinstance(value, list):
            return False
        # If there are no type args, any list is valid
        if not args:
            return True
        # Check if all elements match the expected type
        element_type = args[0]
        return all(_check_isinstance_generic(item, element_type) for item in value)

    # Handle Tuple[T, ...]
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        # If there are no type args, any tuple is valid
        if not args:
            return True
        # Check for variadic tuple (Tuple[T, ...])
        if len(args) == 2 and args[1] is Ellipsis:
            element_type = args[0]
            return all(_check_isinstance_generic(item, element_type) for item in value)
        # Check for fixed-length tuple (Tuple[T1, T2, ...])
        if len(value) != len(args):
            return False
        return all(_check_isinstance_generic(v, t) for v, t in zip(value, args))

    # Handle Dict[K, V]
    if origin is dict:
        if not isinstance(value, dict):
            return False
        # If there are no type args, any dict is valid
        if not args:
            return True
        if len(args) != 2:
            return True  # Malformed type, be permissive
        key_type, value_type = args
        return all(
            _check_isinstance_generic(k, key_type) and _check_isinstance_generic(v, value_type)
            for k, v in value.items()
        )

    # Handle Set[T]
    if origin is set:
        if not isinstance(value, set):
            return False
        # If there are no type args, any set is valid
        if not args:
            return True
        element_type = args[0]
        return all(_check_isinstance_generic(item, element_type) for item in value)

    # Handle Union types (including Optional)
    # Note: typing.Union (Union[X, Y]) and types.UnionType (X | Y) are different
    if origin is Union or origin is types.UnionType:
        return any(_check_isinstance_generic(value, arg) for arg in args)

    # Handle other generic types by checking the origin
    try:
        return isinstance(value, origin)
    except TypeError:
        return False


def validate_args(args: Sequence[EffectParams], parameters: Sequence[Parameter]) -> List[Argument[EffectParams]]:
    if len(args) != len(parameters):
        raise ValueError(f"Expected {len(parameters)} arguments, but only received {len(args)}")

    mapped_args: List[Argument[EffectParams]] = []
    for arg, param in zip(args, parameters):
        expected_type = param.type
        if _check_isinstance_generic(arg, expected_type):
            mapped_args.append(Argument(param.name, arg))
        elif isinstance(arg, RegName):
            if expected_type in (Variable, Label):
                mapped_args.append(Argument(param.name, expected_type(name=arg.name)))
        elif isinstance(arg, str):
            if expected_type in (Variable, Label):
                mapped_args.append(Argument(param.name, expected_type(name=arg)))
        else:
            raise ValueError(f"Expected argument of type {expected_type} but received value of type {type(arg)}")

    return mapped_args


def validate_kwargs(args: Mapping[str, EffectParams], parameters: Sequence[Parameter]) -> Dict[str, EffectParams]:
    if len(args) != len(parameters):
        raise ValueError(f"Expected {len(parameters)} arguments, but only received {len(args)}")

    params = {k: v for k, v, _ in parameters}

    mapped_kwargs: Dict[str, EffectParams] = {}
    for arg_name, arg in args.items():
        expected_type = params[arg_name]
        if _check_isinstance_generic(arg, expected_type):
            mapped_kwargs[arg_name] = arg
        elif isinstance(arg, RegName):
            if expected_type in (Variable, Label):
                mapped_kwargs[arg_name] = expected_type(name=arg.name)
        elif isinstance(arg, str):
            if expected_type in (Variable, Label):
                mapped_kwargs[arg_name] = expected_type(name=arg)
        else:
            raise ValueError(f"Expected argument of type {expected_type} but received value of type {type(arg)}")

    return mapped_kwargs


def parallelize_effects(effects: List[ToolCall], effect_mapping: Dict[str, Effect]) -> List[List[ToolCall]]:
    """
    Determine which effects can be executed in parallel based on their register, reference, and variable dependencies.

    Uses read/write access modes from parameter definitions:
    - Multiple reads to the same resource can happen in parallel
    - Writes are exclusive - they cannot overlap with any reads or writes to the same resource
    - Effects sharing any written RegName, Ref, or Variable cannot be parallelized with effects that read or write the same resource

    Returns a list of lists where each inner list contains EffectCalls that can be
    executed in parallel. The outer list represents the execution order - each group
    must complete before the next group can start.

    Args:
        effects: List of EffectCall objects to parallelize
        effect_mapping: Mapping from effect names to Effect objects

    Returns:
        List of lists of EffectCall objects, where each inner list can be executed in parallel
    """
    if not effects:
        return []

    def extract_dependencies(effect: EffectCall) -> Tuple[Set[RegName | Variable | Ref], Set[RegName | Variable | Ref]]:
        """Extract read and write dependencies from an effect's arguments.

        Returns:
            (read_deps, write_deps) - Sets of RegName, Variable, and Ref objects
        """
        read_deps = set()
        write_deps = set()

        def _extract_from_value(value: EffectParams) -> Set[RegName | Variable | Ref]:
            """Recursively extract RegNames, Refs, and Variables from a value."""
            deps = set()
            if isinstance(value, RegName):
                deps.add(value)
            elif isinstance(value, Variable):
                deps.add(value)
            elif isinstance(value, Ref):
                deps.add(value)
            elif isinstance(value, list):
                for item in value:
                    deps.update(_extract_from_value(item))
            elif isinstance(value, tuple):
                for item in value:
                    deps.update(_extract_from_value(item))
            elif isinstance(value, dict):
                for k, v in value.items():
                    deps.update(_extract_from_value(k))
                    deps.update(_extract_from_value(v))
            elif isinstance(value, set):
                for item in value:
                    deps.update(_extract_from_value(item))
            elif isinstance(value, Object):
                for v in value.attributes.values():
                    deps.update(_extract_from_value(v))
            elif isinstance(value, Class):
                for base in value.bases:
                    deps.update(_extract_from_value(base))
                for v in value.attributes.values():
                    deps.update(_extract_from_value(v))
            return deps

        # Get the effect parameters with access modes
        effect_def = effect_mapping[effect.name]
        param_dict = {param.name: param for param in effect_def.parameters}

        for arg in effect.args:
            param = param_dict.get(arg.name)
            deps = _extract_from_value(arg.arg)

            if param and param.access == "read":
                read_deps.update(deps)
            elif param and param.access == "write":
                write_deps.update(deps)
            else:
                # If access mode not specified, conservatively treat as both read and write
                read_deps.update(deps)
                write_deps.update(deps)

        return read_deps, write_deps

    def can_parallelize(
        reads1: Set[RegName | Variable | Ref],
        writes1: Set[RegName | Variable | Ref],
        reads2: Set[RegName | Variable | Ref],
        writes2: Set[RegName | Variable | Ref],
    ) -> bool:
        """Check if two effects can be parallelized based on their dependencies.

        Rules:
        - Write-Write conflict: Cannot parallelize if they both write to the same resource
        - Write-Read conflict: Cannot parallelize if one writes and the other reads the same resource
        - Read-Read: Can parallelize if they only read the same resource
        """
        # Check for write-write conflicts
        if writes1 & writes2:
            return False

        # Check for write-read conflicts
        if (writes1 & reads2) or (writes2 & reads1):
            return False

        # No conflicts - can parallelize
        return True

    could_parallelize: List[Tuple[ToolCall, EffectCall]] = []
    cannot_parallelize: List[ToolCall] = []
    for e_i, effect in enumerate(effects):
        try:
            could_parallelize.append((effect, parse_effect(effect.name, effect_mapping=effect_mapping)))
        except Exception:
            # If fails, then don't parallelize the rest, just stick them all as sequential after, which will get re-triggered as wrong
            cannot_parallelize = effects[e_i:]
            break

    # Build dependency information for each effect
    effect_deps = [(effect, *extract_dependencies(parsed_effect)) for (effect, parsed_effect) in could_parallelize]

    # Group effects into parallel-executable batches using a greedy algorithm
    # We respect input order: only add effects to a batch if all earlier effects
    # have been processed (either in current or previous batches)
    batches: List[List[ToolCall]] = []
    remaining = list(range(len(could_parallelize)))

    while remaining:
        # Find effects that can run in this batch
        current_batch_indices = []
        batch_reads = set()
        batch_writes = set()

        # Track if we skipped any effects
        first_skipped_idx = None

        for idx in remaining[:]:
            effect, reads, writes = effect_deps[idx]

            # Check if this effect can be added to the current batch
            if can_parallelize(reads, writes, batch_reads, batch_writes):
                # If we haven't skipped any effects yet, we can add this one
                if first_skipped_idx is None:
                    # No conflict - can add to batch
                    current_batch_indices.append(idx)
                    batch_reads.update(reads)
                    batch_writes.update(writes)
                    remaining.remove(idx)
                # If we've already skipped an earlier effect, we can't add this one
                # to maintain order dependencies
            else:
                # Mark that we've skipped this effect
                if first_skipped_idx is None:
                    first_skipped_idx = idx

        # If we couldn't add any effects, force add the first one to make progress
        # This handles edge cases and ensures forward progress
        if not current_batch_indices:
            idx = remaining[0]
            current_batch_indices.append(idx)
            remaining.remove(idx)

        batches.append([effect_deps[idx][0] for idx in current_batch_indices])

    # add the rest from the one that had an error as sequential
    batches.extend([[x] for x in cannot_parallelize])
    return batches

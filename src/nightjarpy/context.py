import inspect
import logging
import types
from abc import update_abstractmethods
from datetime import datetime
from time import sleep
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    NoReturn,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import numpy as np
from pydantic import BaseModel

from nightjarpy.configs import InterpreterConfig, LLMConfig
from nightjarpy.llm.factory import create_llm
from nightjarpy.prompts.base import PromptTemplate
from nightjarpy.types import (
    SCHEMA_DEFS,
    SCHEMA_DEFS_NOFUNC,
    SUCCESS,
    VALUE_SCHEMA,
    VALUE_SCHEMA_TYPES,
    Class,
    EffectError,
    EffectException,
    Frame,
    Func,
    GetFail,
    Immutable,
    JsonSchema,
    JsonSchemaValue,
    Label,
    NaturalCode,
    NotSupportedDataType,
    Object,
    Param,
    Primitive,
    Ref,
    RegName,
    ResponseFormat,
    Signature,
    Success,
    UndefinedLocal,
    UserMessage,
    Value,
    Variable,
    is_function,
)
from nightjarpy.utils import VarGenerator
from nightjarpy.utils.utils import (
    call_function_by_sig,
    deserialize,
    deserialize_json,
    get_object_attributes,
    serialize,
    serialize_json,
    string_to_type,
    type_to_string,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F")
U = TypeVar("U", bound=Value)


class Heap(Generic[T]):
    def __init__(self):
        self.heap: Dict[Ref, Union[T, Frame]] = {}

    def sync(self, other: "Heap"):
        self.heap.update(other.heap)

    def insert(self, ref: Ref, val: Union[T, Frame]):
        self.heap[ref] = val

    def get_frame(self, ref: Ref) -> Frame:
        if ref not in self.heap:
            raise GetFail(f"Address {ref} not found in heap")
        f = self.heap[ref]
        if not isinstance(f, Frame):
            raise ValueError(f"Expected address {ref} to be a Frame, but got {type(f)}")
        return f

    def get(self, ref: Ref) -> T:
        if ref not in self.heap:
            raise GetFail(f"Address {ref} not found in heap")
        v = self.heap[ref]
        if isinstance(v, Frame):
            raise ValueError(f"Expected address {ref} to be a value, but got a Frame")
        return v

    def __contains__(self, ref: Ref) -> bool:
        return ref in self.heap

    def __len__(self) -> int:
        return len(self.heap)


class Register(Generic[T]):
    def __init__(self):
        self._register: Dict[RegName, T] = {}

    def sync(self, other: "Register"):
        self._register.update(other._register)

    def insert(self, name: RegName, val: T):
        self._register[name] = val

    def get(self, name: RegName) -> T:
        if name not in self._register:
            raise GetFail(f"Register name {name} is not in registers")
        v = self._register[name]
        return v

    def __contains__(self, name: RegName) -> bool:
        return name in self._register

    def __len__(self) -> int:
        return len(self._register)


class Context:
    def __init__(
        self,
        temp_var_init: int,
        valid_vars: Set[Variable],
        output_vars: Set[Variable],
        valid_labels: Set[Label],
        python_frame: Optional[types.FrameType],
        llm_config: LLMConfig,
        compute_prompt_template: Optional[PromptTemplate],
        use_functions: bool,
    ):
        self.rng = np.random.default_rng(seed=42)
        frame = Frame(parent_fp=None)
        self.heap = Heap[Value]()
        self.fp = self.new_ref()
        self.global_frame = Frame(parent_fp=None)
        self.heap.insert(ref=self.fp, val=frame)
        self.orig_py_obj: Dict[Ref, Any] = {}
        self.register = Register[Value]()
        self.temp_var_generator = VarGenerator(init=temp_var_init)
        # Checkpoint the frame when entering natural so we don't accidentally look up the wrong value when searching the python stack
        self.python_frame: Optional[types.FrameType] = python_frame
        self.valid_vars = valid_vars
        self.output_vars = output_vars
        self.valid_labels = valid_labels
        self.classes: Dict[str, Type] = {}
        self.llm = create_llm(llm_config)
        self.llm_config = llm_config
        self.compute_prompt_template = compute_prompt_template
        self.forbidden = set([property, object])
        self.use_functions = use_functions

    def reset(self):
        frame = Frame(parent_fp=None)
        self.fp = self.new_ref()
        self.global_frame = Frame(parent_fp=None)
        self.heap = Heap[Value]()
        self.heap.insert(self.fp, frame)
        self.orig_py_obj = {}
        self.register = Register[Value]()
        self.python_frame = None
        self.classes = {}

    def sync(self, other: "Context"):
        """
        Incorporates the changes from the other context into this context
        """
        self.heap.sync(other.heap)
        self.current_frame.sync(other.current_frame)
        self.global_frame.sync(other.global_frame)
        self.orig_py_obj.update(other.orig_py_obj)
        self.register.sync(other.register)
        self.temp_var_generator = VarGenerator(init=other.temp_var_generator.current_id())
        self.valid_vars.update(other.valid_vars)
        self.output_vars.update(other.output_vars)
        self.valid_labels.update(other.valid_labels)
        self.classes.update(other.classes)

    def new_ref(self) -> Ref:
        def _gen_ref():
            x = int(self.rng.integers(0, max(1, len(self.heap)) * 10, size=1)[0])
            ref = Ref(addr=x)
            while ref in self.heap:
                x = int(self.rng.integers(0, max(1, len(self.heap)) * 10, size=1)[0])
                ref = Ref(addr=x)
            return ref

        return _gen_ref()  # type: ignore

    @property
    def current_frame(self) -> Frame:
        current_frame = self.heap.get_frame(self.fp)
        return current_frame

    def encode_signature(self, signature: List[inspect.Parameter], enc_memo: Dict[int, Immutable]) -> Signature:
        params: List[Param] = []
        for param in signature:
            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                kind = "positional-only"
            elif param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                kind = "positional or keyword"
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                kind = "variadic positional"
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                kind = "keyword-only"
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                kind = "variadic keyword"
            else:
                raise ValueError(f"Unsupported parameter kind: {param.kind}")

            params.append(
                Param(
                    name=param.name,
                    annotation=param.annotation,
                    kind=kind,
                    default=self.encode_python_value(
                        param.default if param.default != inspect.Parameter.empty else None,
                        enc_memo,
                    ),
                )
            )
        return Signature(params=tuple(params))

    def decode_signature(self, signature: Signature, dec_memo: Dict[Ref, Any]) -> List[inspect.Parameter]:
        params: List[inspect.Parameter] = []
        for param in signature.params:
            if param.kind == "positional-only":
                kind = inspect.Parameter.POSITIONAL_ONLY
            elif param.kind == "positional or keyword":
                kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
            elif param.kind == "variadic positional":
                kind = inspect.Parameter.VAR_POSITIONAL
            elif param.kind == "keyword-only":
                kind = inspect.Parameter.KEYWORD_ONLY
            elif param.kind == "variadic keyword":
                kind = inspect.Parameter.VAR_KEYWORD
            else:
                raise ValueError(f"Unsupported parameter kind: {param.kind}")
            params.append(
                inspect.Parameter(
                    name=param.name,
                    kind=kind,
                    default=self.decode_and_sync_python_value(param.default, dec_memo),
                    annotation=string_to_type(param.annotation, self.classes),
                )
            )
        return params

    def encode_python_value(self, val: Any, enc_memo: Dict[int, Immutable]) -> Immutable:

        def _inner(val: Any) -> Immutable:
            if id(val) in enc_memo:
                return enc_memo[id(val)]
            if isinstance(val, tuple):
                enc = tuple([_inner(x) for x in val])
                enc_memo[id(val)] = enc
                return enc
            elif isinstance(val, (Primitive, datetime)):
                enc_memo[id(val)] = val
                return val
            elif isinstance(val, list):
                enc_list = []
                list_ref = self.new_ref()
                # Placeholder so this ref doesn't get stolen
                self.heap.insert(list_ref, None)
                # This goes before recursion to avoid infinite recursion
                enc_memo[id(val)] = list_ref

                for x in val:
                    if id(val) == id(x):
                        # Don't recurse on self
                        continue
                    enc_x = _inner(x)
                    enc_list.append(enc_x)

                self.heap.insert(list_ref, enc_list)
                # Maintain pointer to original list so that we can sync the changes when exiting natural env
                self.orig_py_obj[list_ref] = val

                return list_ref
            elif isinstance(val, dict):
                enc_dict = {}
                dict_ref = self.new_ref()
                # Placeholder so this ref doesn't get stolen
                self.heap.insert(dict_ref, None)
                # This goes before recursion to avoid infinite recursion
                enc_memo[id(val)] = dict_ref

                for k, v in sorted(val.items(), key=lambda x: str(x[0])):
                    if not isinstance(k, (Primitive, tuple, NotSupportedDataType)):
                        raise NotImplementedError(
                            f"Dictionary keys must be an immutable type for now, got {type(k)} for {val}"
                        )

                    if id(val) == id(v):
                        # Don't recurse on self
                        continue

                    enc_v = _inner(v)
                    # ref = self.new_ref()
                    # self.heap.insert(ref, enc_v)
                    enc_dict[k] = enc_v

                self.heap.insert(dict_ref, enc_dict)
                # Maintain pointer to original dict so that we can sync the changes when exiting natural env
                self.orig_py_obj[dict_ref] = val

                return dict_ref
            elif isinstance(val, set):
                enc_set = set()
                set_ref = self.new_ref()
                # Placeholder so this ref doesn't get stolen
                self.heap.insert(set_ref, None)
                # This goes before recursion to avoid infinite recursion
                enc_memo[id(val)] = set_ref
                for x in sorted(val, key=lambda x: str(x)):
                    if id(val) == id(x):
                        # Don't recurse on self
                        continue
                    enc_x = _inner(x)

                    enc_set.add(enc_x)
                self.heap.insert(set_ref, enc_set)
                # Maintain pointer to original set so that we can sync the changes when exiting natural env
                self.orig_py_obj[set_ref] = val

                return set_ref
            elif is_function(val):

                try:
                    parameters = list(inspect.signature(val).parameters.values())
                except:
                    parameters = []

                if isinstance(val, types.MethodType):
                    parameters = [
                        inspect.Parameter(
                            name="self",
                            kind=inspect.Parameter.POSITIONAL_ONLY,
                            default=inspect.Parameter.empty,
                        )
                    ] + parameters

                try:
                    full_func = inspect.getsource(val)
                except OSError:
                    full_func = ""
                except TypeError:
                    full_func = ""

                func_ref = self.new_ref()

                self.heap.insert(func_ref, None)
                enc_memo[id(val)] = func_ref

                if not self.use_functions:
                    func = NotSupportedDataType()
                else:
                    func = Func(
                        context=self,
                        name=val.__name__,
                        signature=self.encode_signature(parameters, enc_memo),
                        full_func=full_func,
                        python_func=val,
                    )

                self.heap.insert(func_ref, func)
                self.orig_py_obj[func_ref] = val

                return func_ref
            elif isinstance(val, type):
                enc_attributes: Dict[str, Immutable] = {}
                obj_ref = self.new_ref()
                # Placeholder so this ref doesn't get stolen
                self.heap.insert(obj_ref, None)
                # This goes before recursion to avoid infinite recursion
                enc_memo[id(val)] = obj_ref
                for k in sorted(get_object_attributes(val)):
                    if not isinstance(k, str):
                        raise ValueError("Key of object attributes must be string")
                    if k.startswith("__"):
                        # Convention: underdscore means private, so skip
                        continue
                    try:
                        v = getattr(val, k)
                    except AttributeError:
                        # logger.warning(f"Attribute `{k}` cannot be retrieved... skipped")
                        continue

                    try:
                        if any(v.__qualname__.startswith(f.__qualname__) for f in self.forbidden):
                            # skip methods of forbidden classes
                            continue
                    except (AttributeError, TypeError):
                        pass

                    if id(val) == id(v):
                        # Don't recurse on self
                        continue

                    enc_v = _inner(v)
                    # ref = self.new_ref()
                    # self.heap.insert(ref, enc_v)
                    enc_attributes[k] = enc_v

                enc_annotations: Dict[str, str] = {}
                if getattr(val, "__annotations__", None) is not None:
                    # Helper to check if a type is a primitive
                    def is_primitive_type(t: Any) -> bool:
                        """Check if a type is one of the primitive types."""
                        return t in (types.NoneType, str, int, float, bool, Ref, datetime)

                    # Helper to extract non-primitive classes from a type
                    def extract_classes_from_type(t: Any) -> None:
                        """Recursively extract non-primitive classes from a type and add to self.classes."""
                        if t is None:
                            return

                        # Check if it's a primitive type
                        if is_primitive_type(t):
                            return

                        # Get the origin for generic types (List, Dict, Optional, Union, etc.)
                        origin = get_origin(t)
                        if origin is not None:
                            # For generic types, check all args
                            args = get_args(t)
                            for arg in args:
                                extract_classes_from_type(arg)
                            return

                        # If it's a user-defined class (type), add it
                        if isinstance(t, type) and not t in self.forbidden:
                            qualname = getattr(t, "__qualname__", None)
                            if qualname:
                                qualname = qualname.replace("<run_path>.", "")
                                if qualname not in self.classes:
                                    self.classes[qualname] = t
                                # if t not in self.forbidden:
                                #     self.valid_vars.add(Variable(qualname))
                            return

                        # For other types (like typing aliases), try to get the underlying type
                        if hasattr(t, "__origin__"):
                            extract_classes_from_type(t.__origin__)  # type: ignore
                        if hasattr(t, "__args__"):
                            for arg in t.__args__:  # type: ignore
                                extract_classes_from_type(arg)

                    for k, v in val.__annotations__.items():
                        enc_annotations[k] = type_to_string(v)
                        # Extract and register non-primitive classes from type annotations
                        extract_classes_from_type(v)

                enc_bases: List[Ref] = []
                for b in val.__bases__:
                    enc_b = _inner(b)
                    if not isinstance(enc_b, Ref):
                        raise ValueError("One of the baseclasses is not a class")
                    enc_bases.append(enc_b)

                self.heap.insert(
                    obj_ref,
                    Class(
                        name=val.__qualname__.replace("<run_path>.", ""),
                        bases=tuple(enc_bases),
                        attributes=enc_attributes,
                        annotations=enc_annotations,
                    ),
                )
                self.orig_py_obj[obj_ref] = val
                self.classes[val.__qualname__.replace("<run_path>.", "")] = val
                # if val not in self.forbidden:
                #     self.valid_vars.add(Variable(val.__qualname__))

                return obj_ref
            elif isinstance(val, object):
                enc_obj_dict: Dict[str, Immutable] = {}
                obj_ref = self.new_ref()
                # Placeholder so this ref doesn't get stolen
                self.heap.insert(obj_ref, None)
                # This goes before recursion to avoid infinite recursion
                enc_memo[id(val)] = obj_ref
                for k in sorted(get_object_attributes(val)):
                    if not isinstance(k, str):
                        raise ValueError("Key of object attributes must be string")
                    if k.startswith("__"):
                        # Convention: underdscore means private, so skip
                        continue
                    try:
                        v = getattr(val, k)
                    except AttributeError:
                        logger.warning(f"Attribute `{k}` cannot be retrieved... skipped")
                        continue

                    try:
                        if any(v.__qualname__.startswith(f.__qualname__) for f in self.forbidden):
                            # skip methods of forbidden classes
                            continue
                    except (AttributeError, TypeError):
                        pass

                    if id(val) == id(v):
                        # Don't recurse on self
                        continue

                    enc_v = _inner(v)
                    # ref = self.new_ref()
                    # self.heap.insert(ref, enc_v)
                    enc_obj_dict[k] = enc_v

                self.heap.insert(
                    obj_ref,
                    Object(_class=val.__class__.__qualname__.replace("<run_path>.", ""), attributes=enc_obj_dict),
                )
                self.orig_py_obj[obj_ref] = val
                self.classes[val.__class__.__qualname__.replace("<run_path>.", "")] = val.__class__
                # if val.__class__ not in self.forbidden:
                #     self.valid_vars.add(Variable(val.__class__.__qualname__))

                return obj_ref
            else:
                raise ValueError("Unsupported data type")

        return _inner(val)

    def decode_and_sync_python_value(self, val: Immutable, dec_memo: Dict[Ref, Any]) -> Any:

        def _inner(val: Immutable) -> Any:

            if isinstance(val, tuple):
                return tuple([_inner(x) for x in val])
            elif isinstance(val, Ref):
                if val in dec_memo:
                    return dec_memo[val]

                enc_val = self.heap.get(val)
                if isinstance(enc_val, list):
                    dec_val = []
                    if val in self.orig_py_obj:
                        dec_memo[val] = self.orig_py_obj[val]
                    else:
                        dec_memo[val] = dec_val
                    for x in enc_val:
                        if id(val) == id(x):
                            # Don't recurse on self
                            continue
                        dec_val.append(_inner(x))

                    if val in self.orig_py_obj:
                        orig_py_val = self.orig_py_obj[val]
                        # Sync with original object
                        if not isinstance(orig_py_val, list):
                            raise ValueError(f"{val}'s value type mismatch with existing value")
                        orig_py_val[:] = dec_val
                        return orig_py_val
                    return dec_val

                elif isinstance(enc_val, dict):
                    dec_val = {}
                    if val in self.orig_py_obj:
                        dec_memo[val] = self.orig_py_obj[val]
                    else:
                        dec_memo[val] = dec_val
                    for k, v in enc_val.items():
                        if not isinstance(k, (Primitive, tuple, NotSupportedDataType)):
                            raise NotImplementedError("Dictionary keys must be an immutable type for now")

                        # enc_v = self.heap.get(v)
                        if id(val) == id(v):
                            # Don't recurse on self
                            continue
                        dec_val[k] = _inner(v)

                    if val in self.orig_py_obj:
                        orig_py_val = self.orig_py_obj[val]
                        if not isinstance(orig_py_val, dict):
                            raise ValueError(f"{val}'s value type mismatch with existing value")
                        deletion = []
                        for k in orig_py_val:
                            if k not in dec_val:
                                deletion.append(k)
                        for k in deletion:
                            del orig_py_val[k]
                        orig_py_val |= {k: v for k, v in dec_val.items()}
                        return orig_py_val
                    return dec_val

                elif isinstance(enc_val, set):
                    dec_val = set()
                    if val in self.orig_py_obj:
                        dec_memo[val] = self.orig_py_obj[val]
                    else:
                        dec_memo[val] = dec_val
                    for x in enc_val:
                        if id(val) == id(x):
                            # Don't recurse on self
                            continue
                        dec_val.add(_inner(x))

                    if val in self.orig_py_obj:
                        orig_py_val = self.orig_py_obj[val]
                        if not isinstance(orig_py_val, set):
                            raise ValueError(f"{val}'s value type mismatch with existing value")
                        orig_py_val.clear()
                        orig_py_val.update(dec_val)
                        return orig_py_val
                    return dec_val

                elif isinstance(enc_val, Object):
                    if val in self.orig_py_obj:
                        obj = self.orig_py_obj[val]
                        assert isinstance(obj, object)

                        if obj.__class__.__qualname__.replace("<run_path>.", "") != enc_val._class:
                            raise ValueError(
                                f"Class doesn't match: Original is `{obj.__class__.__qualname__.replace("<run_path>.", "")}` but encoded is `{enc_val._class}"
                            )
                    else:
                        # Remove Object wrapper
                        if enc_val._class not in self.classes:
                            raise ValueError(f"Undefined class `{enc_val._class}`")
                        _class = self.classes[enc_val._class]

                        obj = _class.__new__(_class)  # type: ignore

                    dec_memo[val] = obj

                    dec_dict = {}
                    for k, v in enc_val.attributes.items():
                        if id(val) == id(v):
                            # Don't recurse on self
                            continue
                        dec_dict[k] = _inner(v)

                    if not val in self.orig_py_obj:
                        # use init to initialize object, but then we update all attributes one by one later
                        dec_dict["self"] = obj
                        call_function_by_sig(obj.__init__, dec_dict)

                    # deletion = []
                    # for k in get_object_attributes(obj):
                    #     if k.startswith("__"):
                    #         continue
                    #     if any(v.__qualname__.contains(f) for f in self.forbidden):
                    #         # skip methods of forbidden classes
                    #         continue
                    #     try:
                    #         v = getattr(obj, k)
                    #     except AttributeError:
                    #         logger.warning(f"Attribute `{k}` cannot be retrieved... skipped")
                    #         continue
                    #     if k not in dec_dict:
                    #         deletion.append(k)
                    # for k in deletion:
                    #     try:
                    #         if k in obj.__dict__:
                    #             del obj.__dict__[k]
                    #         # delattr(obj, k)
                    #     except (TypeError, AttributeError, ValueError):
                    #         logger.warning(f"Attribute `{k}` could not be deleted... skipped")
                    #         continue

                    for k, v in dec_dict.items():
                        try:
                            if is_function(v):
                                if "self" in inspect.signature(v).parameters:
                                    v = types.MethodType(v, obj)

                            obj.__dict__[k] = v
                            # setattr(obj, k, v)
                        except (TypeError, AttributeError, ValueError):
                            # logger.warning(f"Attribute `{k}` could not be updated... skipped")
                            continue

                    return obj

                elif isinstance(enc_val, Class):
                    if val in self.orig_py_obj:
                        _cls = self.orig_py_obj[val]
                        assert isinstance(_cls, type)

                        if _cls.__qualname__.replace("<run_path>.", "") != enc_val.name:
                            raise ValueError(
                                f"Name doesn't match: Original is `{_cls.__qualname__.replace("<run_path>.", "")}` but encoded is `{enc_val.name}"
                            )

                        if _cls in self.forbidden:
                            return _cls
                    else:
                        # Remove Class wrapper
                        dec_bases = []
                        for b in enc_val.bases:
                            dec_bases.append(_inner(b))
                        # object is special
                        # if not _cls == object:
                        #     _cls.__bases__ = tuple(dec_bases)
                        _cls = types.new_class(enc_val.name, tuple(dec_bases), {})
                        _cls.__annotations__ = {}
                        self.classes[enc_val.name] = _cls

                    dec_memo[val] = _cls

                    dec_dict = {}
                    for k, v in enc_val.attributes.items():
                        if id(val) == id(v):
                            # Don't recurse on self
                            continue
                        dec_dict[k] = _inner(v)

                    for k, v in dec_dict.items():
                        try:
                            if is_function(v):
                                if "self" in inspect.signature(v).parameters:
                                    v = types.MethodType(v, _cls)

                            # _cls.__dict__[k] = v
                            setattr(_cls, k, v)
                        except (TypeError, AttributeError, ValueError):
                            # logger.warning(f"Attribute `{k}` could not be updated... skipped")
                            continue

                    update_abstractmethods(_cls)

                    deletion = []
                    if getattr(_cls, "__annotations__", None) is not None:
                        for k in _cls.__annotations__.keys():
                            if k not in enc_val.annotations:
                                deletion.append(k)
                        for k in deletion:
                            del _cls.__annotations__[k]
                    for k, v in enc_val.annotations.items():
                        try:
                            _cls.__annotations__[k] = string_to_type(v, self.classes)
                        except (ValueError, NameError):
                            logger.warning(f"Annotation `{k}` could not be updated... skipped")

                    return _cls

                elif isinstance(enc_val, Func):
                    if val in self.orig_py_obj:
                        func = self.orig_py_obj[val]
                        assert is_function(func)

                        return func
                    else:
                        func = enc_val.to_python_func()
                        return func
                elif isinstance(enc_val, NotSupportedDataType):
                    if val in self.orig_py_obj:
                        return self.orig_py_obj[val]
                    else:
                        logger.error("Could not retrieve original data")
                        return enc_val
                else:
                    return _inner(enc_val)

            elif isinstance(val, Primitive):
                return val
            elif isinstance(val, NotSupportedDataType):
                return val
            else:
                raise ValueError("Unsupport data type")

        return _inner(val)

    def get_closure(self, fp: Ref) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        f_locals: Dict[Variable, Immutable] = {}
        # Need to update starting with the parent frame to the current frame to get the right closure
        frames = [fp]
        curr_fp = fp
        while curr_fp is not None:
            f = self.heap.get_frame(curr_fp)
            if f.parent_fp is None:
                break
            frames.append(f.parent_fp)
            curr_fp = f.parent_fp

        for fp in reversed(frames):
            f = self.heap.get_frame(fp)
            f_locals.update(f.frame)

        python_frames: List[types.FrameType] = []
        python_globals: Dict[str, Any] = self.python_frame.f_globals if self.python_frame else {}

        curr_frame = self.python_frame
        while curr_frame is not None:
            parent_fp = curr_frame.f_back
            if parent_fp is None:
                break
            python_frames.append(curr_frame)
            curr_frame = curr_frame.f_back

        python_locals: Dict[str, Any] = {}
        for frame in reversed(python_frames):
            python_locals.update(frame.f_locals)

        for var, val in f_locals.items():
            if var in self.valid_vars:
                python_locals[var.name] = self.decode_and_sync_python_value(val, {})

        final_python_locals, final_python_globals = {}, {}
        for k, v in python_locals.items():
            if Variable(k) in self.valid_vars:
                final_python_locals[k] = v

        for k, v in python_globals.items():
            if Variable(k) in self.valid_vars:
                final_python_globals[k] = v

        return final_python_locals, final_python_globals

    def loadreg(self, reg: RegName) -> Value:
        return self.register.get(reg)

    def _check_attribute_type(
        self, value: Value, expected_type: Type[Any], attr_name: str, class_name: Optional[str] = None
    ) -> None:
        """
        Check if a value matches a type annotation.
        Raises TypeError if the type doesn't match.
        """
        expected_type_str = type_to_string(expected_type)
        origin = get_origin(expected_type)

        def create_error_msg(val: Value, desc: str) -> str:
            prefix = (
                f"Attribute `{attr_name}` of object with class `{class_name}` has type mismatch. "
                if class_name
                else f"Value does not match type `{expected_type_str}`. "
            )
            return f"{prefix}Expected `{expected_type_str}`, but got value `{val}` of type `{desc}`"

        # 1. Handle Union/Optional - try each member
        if origin in (types.UnionType, Union, Optional):
            args = get_args(expected_type)
            if len(args) == 2 and type(None) in args:
                # Optional[X]
                if value is None:
                    return
                non_none_type = next(a for a in args if a is not type(None))
                self._check_attribute_type(value, non_none_type, attr_name, class_name)
                return
            # Union[X, Y, ...]
            for arg in args:
                try:
                    self._check_attribute_type(value, arg, attr_name, class_name)
                    return  # Success!
                except TypeError:
                    continue  # Try next union member
            # All members failed
            actual_desc = f"Object of type `{value._class}`" if isinstance(value, Object) else type(value).__name__
            raise TypeError(create_error_msg(value, actual_desc))

        # 2. Handle Literal types
        if origin is Literal:
            literal_args = get_args(expected_type)
            actual_value = self.heap.get(value) if isinstance(value, Ref) and value in self.heap else value
            if actual_value in literal_args:
                return
            raise TypeError(create_error_msg(value, f"not in {literal_args}"))

        # 3. Handle None
        if value is None:
            if expected_type in (type(None), None):
                return
            raise TypeError(create_error_msg(value, "NoneType"))

        # 4. Handle Object with user-defined class (NOT behind Ref)
        if isinstance(value, Object):
            if isinstance(expected_type, type):
                expected_class_name = expected_type.__qualname__.replace("<run_path>.", "")
                if value._class == expected_class_name:
                    return
                raise TypeError(create_error_msg(value, f"Object of type `{value._class}`"))

        # 5. Handle Ref - dereference and check the contained value
        if isinstance(value, Ref):
            if value not in self.heap:
                raise ValueError(f"`{value}` does not point to anything in the heap")
            derefed_value = self.heap.get(value)

            # Check Ref type annotation
            if expected_type is Ref:
                return

            # Check collection types behind Ref
            if origin in (list, List):
                if not isinstance(derefed_value, list):
                    raise TypeError(create_error_msg(value, type(derefed_value).__name__))
                args = get_args(expected_type)
                if args:
                    for i, item in enumerate(derefed_value):
                        self._check_attribute_type(item, args[0], f"{attr_name}[{i}]", None)
                return

            if origin in (dict, Dict):
                if not isinstance(derefed_value, dict):
                    raise TypeError(create_error_msg(value, type(derefed_value).__name__))
                args = get_args(expected_type)
                if len(args) == 2:
                    for k, v in derefed_value.items():
                        self._check_attribute_type(k, args[0], f"{attr_name}.key({k})", None)
                        self._check_attribute_type(v, args[1], f"{attr_name}[{k}]", None)
                return

            if origin in (set, Set):
                if not isinstance(derefed_value, set):
                    raise TypeError(create_error_msg(value, type(derefed_value).__name__))
                args = get_args(expected_type)
                if args:
                    for item in derefed_value:
                        self._check_attribute_type(item, args[0], f"{attr_name}.element", None)
                return

            # Check user-defined class behind Ref
            if isinstance(expected_type, type) and isinstance(derefed_value, Object):
                expected_class_name = expected_type.__qualname__.replace("<run_path>.", "")
                if derefed_value._class == expected_class_name:
                    return
                raise TypeError(create_error_msg(value, f"Object of type `{derefed_value._class}`"))

            # Recursively check other dereferenced values
            self._check_attribute_type(derefed_value, expected_type, attr_name, class_name)
            return

        # 6. Handle Tuple (NOT behind Ref)
        if isinstance(value, tuple):
            if origin in (tuple, Tuple):
                args = get_args(expected_type)
                if args:
                    if len(args) == 2 and args[1] is ...:
                        # Tuple[X, ...]
                        for i, item in enumerate(value):
                            self._check_attribute_type(item, args[0], f"{attr_name}[{i}]", None)
                    else:
                        # Tuple[X, Y, Z]
                        if len(value) != len(args):
                            raise ValueError(f"Expected tuple of length {len(args)}, got {len(value)}")
                        for i, (item, arg) in enumerate(zip(value, args)):
                            self._check_attribute_type(item, arg, f"{attr_name}[{i}]", None)
                return
            if expected_type == tuple:
                return
            raise TypeError(create_error_msg(value, "tuple"))

        # 7. Handle primitive types
        if origin is None and isinstance(expected_type, type):
            if isinstance(value, expected_type):
                return

        # 8. Default: type mismatch
        raise TypeError(create_error_msg(value, type(value).__name__))

    def _check_valid_val(self, val: U) -> U:
        """
        Performs checks that val is valid. Returns the value if checks pass, else raises ValueError
        """
        if isinstance(val, Object):
            if not val._class in self.classes:
                raise ValueError(
                    "Undefined class... Make sure to use the qualname of the class and that class is already in defined the heap"
                )

            # Check attribute types against class annotations
            # class_type = self.classes[val._class]
            # class_annotations = getattr(class_type, "__annotations__", {})

            # Check each attribute in the object
            # for attr_name, attr_value in val.attributes.items():
            #     if attr_name in class_annotations:
            #         expected_type = class_annotations[attr_name]
            #         self._check_attribute_type(attr_value, expected_type, attr_name, val._class)

        if isinstance(val, (tuple, list, set)):
            for x in val:
                if isinstance(x, Ref) and x not in self.heap:
                    raise ValueError(f"`{x}` does not point to anything in the heap")
        elif isinstance(val, dict):
            for k, v in val.items():
                for x in (k, v):
                    if isinstance(x, Ref) and x not in self.heap:
                        raise ValueError(f"`{x}` does not point to anything in the heap")
        elif isinstance(val, (Object, Class)):
            for x in val.attributes.values():
                if isinstance(x, Ref) and x not in self.heap:
                    raise ValueError(f"`{x}` does not point to anything in the heap")
            if isinstance(val, Object):
                if val._class not in self.classes:
                    raise ValueError(f"`{val._class}` is not a defined class")
            else:
                # Make sure class is registered in self.classes
                ref = self.new_ref()
                self.heap.insert(ref, val)
                self.decode_and_sync_python_value(ref, {})
        elif isinstance(val, Ref) and val not in self.heap:
            raise ValueError(f"`{val}` does not point to anything in the heap")
        elif isinstance(val, (Primitive, Func)):
            pass
        else:
            raise ValueError("Unexpected value type")

        return val

    def storereg(self, dest: RegName, val: Value) -> Success:
        self.register.insert(dest, self._check_valid_val(val))
        return SUCCESS

    def lookup(self, var: Variable, local_only: bool = False) -> Immutable:
        # x_var = Variable(var)
        var_str = var.name

        if not var in self.valid_vars:
            raise ValueError(f"Undefined variable `{var_str}`")

        # if "." in var:
        # This is an attribute, so throw an error
        # raise ValueError(f"Cannot lookup an attribute of an object directly.")

        def _python_lookup(frame: types.FrameType | None, local: bool) -> Immutable:

            if frame is None:
                raise ValueError(f"Undefined variable `{var_str}`")
            if local:
                # First check the local frame
                f_locals = frame.f_locals
                if var_str in f_locals:
                    return self.encode_python_value(f_locals[var_str], {})
                else:
                    parent_fp = frame.f_back
                    if parent_fp is None:
                        raise ValueError(f"Undefined variable `{var_str}`")
                    return _python_lookup(parent_fp, local=True)
            else:
                # Then check the global frame
                f_globals = frame.f_globals
                if var_str in f_globals:
                    return self.encode_python_value(f_globals[var_str], {})
                # Then check the builtins
                f_builtins = frame.f_builtins
                if var_str in f_builtins:
                    return self.encode_python_value(f_builtins[var_str], {})
                raise ValueError(f"Undefined variable `{var_str}`")

        def _lookup(fp: Ref) -> Immutable:
            try:
                ref = self.heap.get_frame(fp).get(var)
                return ref
            except GetFail:
                # Look at previous frames
                parent_fp = self.heap.get_frame(fp).parent_fp
                if parent_fp is not None:
                    return _lookup(parent_fp)
                try:
                    return _python_lookup(self.python_frame, local=True)
                except ValueError:
                    if local_only:
                        raise UndefinedLocal(f"Undefined variable `{var_str}`")
                    # Check if the variable is a global variable
                    if var in self.global_frame.frame:
                        ref = self.global_frame.frame[var]
                        return ref
                    else:
                        return _python_lookup(self.python_frame, local=False)

        val = _lookup(self.fp)
        return val

    def lookup_reg(self, dest: RegName, var: Variable, local_only: bool = False) -> Success:
        val = self.lookup(var, local_only)
        self.register.insert(dest, val)
        return SUCCESS

    def assign(self, var: Variable, val: Immutable) -> Success:
        if "." in var.name:
            # This is an attribute, so throw an error
            raise ValueError(f"Cannot store an attribute of an object directly.")
        self.current_frame.insert(var, self._check_valid_val(val))
        self.valid_vars.add(var)
        return SUCCESS

    def assign_reg(self, var: Variable, src: RegName) -> Success:
        val = self.loadreg(src)
        if not (isinstance(val, Primitive) or isinstance(val, tuple)):
            raise ValueError("Can only assign immutable values and references to variables.")
        return self.assign(var, val)

    def deref(self, ref: Ref) -> Value:
        val = self.heap.get(ref)
        return val

    def deref_reg(self, dest: RegName, refreg: RegName) -> Success:
        ref = self.loadreg(refreg)
        if not isinstance(ref, Ref):
            raise ValueError(f"Value stored in register `{refreg}` is not a reference")
        val = self.deref(ref)
        self.register.insert(dest, val)
        return SUCCESS

    def ref(self, value: Value) -> Ref:
        ref = self.new_ref()
        self.heap.insert(ref, self._check_valid_val(value))
        return ref

    def ref_reg(self, dest: RegName, valreg: RegName) -> Success:
        val = self.loadreg(valreg)
        ref = self.ref(val)
        self.register.insert(dest, ref)
        return SUCCESS

    def setref(self, ref: Ref, value: Value) -> Success:
        if ref not in self.heap:
            raise ValueError("Unknown reference. Setting/updating references can only be done to existing references")
        original_val = self.heap.get(ref)
        if type(original_val) != type(value):
            raise ValueError(f"{value}'s value type mismatch with existing value")
        self.heap.insert(ref, self._check_valid_val(value))

        # Sync with original object
        if ref in self.orig_py_obj:
            self.decode_and_sync_python_value(ref, {})
        return SUCCESS

    def setref_reg(self, refreg: RegName, valreg: RegName) -> Success:
        ref = self.loadreg(refreg)
        if not isinstance(ref, Ref):
            raise ValueError(f"Value stored in register `{refreg}` is not a reference")
        val = self.loadreg(valreg)
        return self.setref(ref, val)

    # Goto is implemented using exception handling
    # This function raises a specialized exception with the label name
    # The program is assumed to have wrapped the natural code with a try-except block
    # catching EffectException that checks if the label name is the corresponding one.
    # Otherwise, reraise the EffectException to an outer context that label corresponds to.
    # The defaults nightjar implements (e.g. break, continue, return) implements the try-except
    # automatically with compilation.
    def goto(self, label: Label, value: Immutable) -> NoReturn:
        if label not in self.valid_labels:
            raise ValueError(f"Program label {label} is not a valid label")

        dec_value = self.decode_and_sync_python_value(value, {})

        raise EffectException(name=label.name, value=dec_value)

    def goto_reg(self, label: Label, valreg: RegName) -> NoReturn:
        val = self.loadreg(valreg)
        if not isinstance(val, Primitive) and not isinstance(val, Tuple):
            raise ValueError("Goto value must be primitive or reference")
        self.goto(label, val)

    def compute(self, instruction: NaturalCode, args: List[Value]) -> Value:
        json_structured_output = self.llm_config.json_structured_output

        if self.compute_prompt_template is None:
            raise RuntimeError("Missing prompt for `compute`")

        system_prompt = self.compute_prompt_template.system.format()

        if json_structured_output:
            serialize_fun = serialize_json if json_structured_output else serialize
        else:
            serialize_fun = serialize

        serialized_args = ", ".join([serialize_fun(x) for x in args])

        prompt = f"{instruction}\n<args>[{serialized_args}]</args>"

        data_types: List[JsonSchemaValue] = [
            {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["error"]},
                    "message": {"type": "string"},
                },
                "required": ["type", "message"],
                "additionalProperties": False,
            }
        ]
        data_types += VALUE_SCHEMA_TYPES

        if json_structured_output:
            res = self.llm.gen_structured_output(
                [UserMessage(content=prompt)],
                schema=ResponseFormat(
                    res_schema={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "Result",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "result": {"anyOf": data_types},
                                },
                                "required": ["result"],
                                "additionalProperties": False,
                                "$defs": SCHEMA_DEFS if self.use_functions else SCHEMA_DEFS_NOFUNC,
                            },
                        },
                    },
                ),
                system=system_prompt,
            )
        else:
            res: str = self.llm.gen(
                [UserMessage(content=prompt)],
                system=system_prompt,
            )

        if json_structured_output:
            if not isinstance(res, Dict) or "result" not in res:
                raise ValueError("LLM did not return a response")
            if isinstance(res["result"], Dict) and "type" in res["result"] and res["result"]["type"] == "error":
                logger.info(f"Compute error: {res["result"]["message"]}")
                raise ValueError(f"Error during computation: {res["result"]["message"]}")
            res_value = deserialize_json(res["result"], self)
        else:
            res = res.strip()
            if res.startswith("Error:"):
                logger.info(f"Compute error: {res}")
                raise ValueError(f"Error during computation: {res}")
            elif "Result:" in res:
                res_value = deserialize(res.split("Result:")[1].strip())
            else:
                raise ValueError(f"Could not parse computation output {res}")

        if isinstance(res_value, RegName):
            raise ValueError("Unexpected result value type")

        if isinstance(res_value, List) and any(isinstance(x, RegName) for x in res_value):
            raise ValueError("Unexpected result value type")

        logger.info(f"Compute result: {res_value}")

        return res_value  # type: ignore

    def compute_reg(self, instruction: NaturalCode, dest: RegName, src: List[RegName]) -> Success:
        args = [self.loadreg(r) for r in src]
        res = self.compute(instruction, args)
        return self.storereg(dest, res)

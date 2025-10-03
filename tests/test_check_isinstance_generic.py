"""Test the _check_isinstance_generic function with all EffectParams types."""

from datetime import datetime

from nightjarpy.types import Class, Label, NaturalCode, Object, Ref, RegName, Variable
from nightjarpy.utils.utils import _check_isinstance_generic


def test_basic_types():
    """Test basic Python types."""
    # str (Variable, Label, NaturalCode are all str type aliases)
    assert _check_isinstance_generic("hello", str)
    assert not _check_isinstance_generic(123, str)

    # int
    assert _check_isinstance_generic(42, int)
    assert not _check_isinstance_generic("42", int)

    # float
    assert _check_isinstance_generic(3.14, float)
    assert not _check_isinstance_generic(3, float)

    # bool
    assert _check_isinstance_generic(True, bool)
    assert _check_isinstance_generic(False, bool)

    # None
    assert _check_isinstance_generic(None, type(None))


def test_dataclasses():
    """Test dataclass types."""
    # Ref
    ref = Ref(addr=123)
    assert _check_isinstance_generic(ref, Ref)
    assert not _check_isinstance_generic("not a ref", Ref)

    # RegName
    reg = RegName(name="test_reg")
    assert _check_isinstance_generic(reg, RegName)
    assert not _check_isinstance_generic("not a regname", RegName)

    # Object
    obj = Object(_class="TestClass", attributes={"x": 1, "y": "hello"})
    assert _check_isinstance_generic(obj, Object)
    assert not _check_isinstance_generic({}, Object)

    # Class
    cls = Class(
        name="TestClass",
        bases=(Ref(addr=0),),
        annotations={"x": "int"},
        attributes={"y": "test"},
    )
    assert _check_isinstance_generic(cls, Class)
    assert not _check_isinstance_generic({}, Class)


def test_datetime():
    """Test datetime type."""
    dt = datetime.now()
    assert _check_isinstance_generic(dt, datetime)
    assert not _check_isinstance_generic("2024-01-01", datetime)


def test_list_generic():
    """Test List[T] generic types."""
    # List[RegName]
    reg_list = [RegName(name="r1"), RegName(name="r2")]
    assert _check_isinstance_generic(reg_list, list[RegName])
    assert not _check_isinstance_generic([1, 2, 3], list[RegName])
    assert not _check_isinstance_generic("not a list", list[RegName])

    # Empty list should pass
    assert _check_isinstance_generic([], list[RegName])

    # Mixed types should fail
    assert not _check_isinstance_generic([RegName(name="r1"), "string"], list[RegName])


def test_tuple_generic():
    """Test Tuple[T, ...] generic types (variadic tuples)."""
    # Tuple[int, ...]
    assert _check_isinstance_generic((1, 2, 3), tuple[int, ...])
    assert _check_isinstance_generic((), tuple[int, ...])
    assert not _check_isinstance_generic((1, "two", 3), tuple[int, ...])
    assert not _check_isinstance_generic([1, 2, 3], tuple[int, ...])

    # Tuple[Ref, ...]
    assert _check_isinstance_generic((Ref(addr=1), Ref(addr=2)), tuple[Ref, ...])
    assert not _check_isinstance_generic((Ref(addr=1), "not a ref"), tuple[Ref, ...])


def test_tuple_fixed():
    """Test fixed-length Tuple[T1, T2, ...] types."""
    # Tuple[int, str]
    assert _check_isinstance_generic((1, "hello"), tuple[int, str])
    assert not _check_isinstance_generic((1, 2), tuple[int, str])
    assert not _check_isinstance_generic((1, "hello", 3), tuple[int, str])


def test_dict_generic():
    """Test Dict[K, V] generic types."""
    # Dict[str, int]
    assert _check_isinstance_generic({"a": 1, "b": 2}, dict[str, int])
    assert _check_isinstance_generic({}, dict[str, int])
    assert not _check_isinstance_generic({"a": "not int"}, dict[str, int])
    assert not _check_isinstance_generic({1: 2}, dict[str, int])
    assert not _check_isinstance_generic("not a dict", dict[str, int])


def test_set_generic():
    """Test Set[T] generic types."""
    # Set[int]
    assert _check_isinstance_generic({1, 2, 3}, set[int])
    assert _check_isinstance_generic(set(), set[int])
    assert not _check_isinstance_generic({1, "two"}, set[int])
    assert not _check_isinstance_generic([1, 2, 3], set[int])


def test_union_types():
    """Test Union and | types."""
    from typing import Union

    # Union[int, str]
    assert _check_isinstance_generic(42, Union[int, str])
    assert _check_isinstance_generic("hello", Union[int, str])
    assert not _check_isinstance_generic(3.14, Union[int, str])

    # int | str (PEP 604 syntax)
    assert _check_isinstance_generic(42, int | str)
    assert _check_isinstance_generic("hello", int | str)
    assert not _check_isinstance_generic(3.14, int | str)

    # Optional[int] (which is Union[int, None])
    from typing import Optional

    assert _check_isinstance_generic(42, Optional[int])
    assert _check_isinstance_generic(None, Optional[int])
    assert not _check_isinstance_generic("not int", Optional[int])


def test_nested_generics():
    """Test nested generic types."""
    # List[List[int]]
    assert _check_isinstance_generic([[1, 2], [3, 4]], list[list[int]])
    assert not _check_isinstance_generic([[1, "two"]], list[list[int]])

    # Dict[str, List[int]]
    assert _check_isinstance_generic({"a": [1, 2], "b": [3]}, dict[str, list[int]])
    assert not _check_isinstance_generic({"a": [1, "two"]}, dict[str, list[int]])

    # Tuple[Tuple[int, ...], ...]
    assert _check_isinstance_generic(((1, 2), (3, 4, 5)), tuple[tuple[int, ...], ...])


def test_complex_effectparams_types():
    """Test complex types that appear in EffectParams through Value."""
    # Nested immutables in tuples
    nested_tuple = (1, "hello", Ref(addr=42), (True, None))
    # Note: We can't easily construct the full Immutable type programmatically,
    # but individual checks work

    # Object with immutable attributes
    obj = Object(
        _class="TestClass",
        attributes={
            "num": 42,
            "text": "hello",
            "ref": Ref(addr=1),
            "nested": (1, 2, 3),
        },
    )
    assert _check_isinstance_generic(obj, Object)

    # List of immutables (common in Mutable)
    immutable_list = [1, "hello", Ref(addr=1), None, True, 3.14]
    assert _check_isinstance_generic(immutable_list, list)

    # Dict with immutable keys and values
    immutable_dict = {
        "key1": 1,
        "key2": Ref(addr=2),
        (1, 2): "tuple_key",
    }
    assert _check_isinstance_generic(immutable_dict, dict)

    # Set of immutables
    immutable_set = {1, 2, 3, "hello"}
    assert _check_isinstance_generic(immutable_set, set)


def test_edge_cases():
    """Test edge cases and error conditions."""
    # Empty collections
    assert _check_isinstance_generic([], list)
    assert _check_isinstance_generic({}, dict)
    assert _check_isinstance_generic(set(), set)
    assert _check_isinstance_generic((), tuple)

    # Wrong container type
    assert not _check_isinstance_generic([1, 2], tuple[int, ...])
    assert not _check_isinstance_generic((1, 2), list[int])
    assert not _check_isinstance_generic({1, 2}, list[int])

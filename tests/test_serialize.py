from datetime import datetime

from nightjarpy.types import Class, EffectError, Object, Ref, Success
from nightjarpy.utils.utils import deserialize, serialize


def test_basic_types():
    """Test serialization of basic immutable types."""
    print("Testing basic types...")

    # None
    assert serialize(None) == "None"
    assert deserialize("None") is None

    # Booleans
    assert serialize(True) == "True"
    assert deserialize("True") is True
    assert serialize(False) == "False"
    assert deserialize("False") is False

    # Integers
    assert serialize(42) == "42"
    assert deserialize("42") == 42

    # Floats
    assert serialize(3.14) == "3.14"
    assert deserialize("3.14") == 3.14

    # Strings
    assert serialize("hello") == "'hello'"
    assert deserialize("'hello'") == "hello"

    print("✓ Basic types passed")


def test_ref():
    """Test Ref serialization."""
    print("Testing Ref...")

    ref = Ref(addr=42)
    serialized = serialize(ref)
    assert serialized == "Ref(42)", f"Expected 'Ref(42)', got '{serialized}'"

    deserialized = deserialize("Ref(42)")
    assert isinstance(deserialized, Ref)
    assert deserialized.addr == 42

    print("✓ Ref passed")


def test_tuples():
    """Test tuple serialization."""
    print("Testing tuples...")

    # Empty tuple
    assert serialize(()) == "()"
    assert deserialize("()") == ()

    # Single element tuple
    serialized = serialize((1,))
    assert serialized == "(1,)", f"Expected '(1,)', got '{serialized}'"
    assert deserialize("(1,)") == (1,)

    # Multi-element tuple
    serialized = serialize((1, 2, 3))
    assert serialized == "(1, 2, 3)", f"Expected '(1, 2, 3)', got '{serialized}'"
    assert deserialize("(1, 2, 3)") == (1, 2, 3)

    # Nested tuple with Ref
    serialized = serialize(("a", Ref(7)))
    assert serialized == "('a', Ref(7))", f"Expected '('a', Ref(7))', got '{serialized}'"
    deserialized = deserialize("('a', Ref(7))")
    assert deserialized == ("a", Ref(7))

    print("✓ Tuples passed")


def test_lists():
    """Test list serialization."""
    print("Testing lists...")

    # Empty list
    assert serialize([]) == "[]"
    assert deserialize("[]") == []

    # Simple list
    serialized = serialize([1, 2, 3])
    assert serialized == "[1, 2, 3]", f"Expected '[1, 2, 3]', got '{serialized}'"
    assert deserialize("[1, 2, 3]") == [1, 2, 3]

    # List with Ref
    serialized = serialize([Ref(14), 3, 7])
    assert serialized == "[Ref(14), 3, 7]", f"Expected '[Ref(14), 3, 7]', got '{serialized}'"
    deserialized = deserialize("[Ref(14), 3, 7]")
    assert deserialized == [Ref(14), 3, 7]

    print("✓ Lists passed")


def test_dicts():
    """Test dict serialization."""
    print("Testing dicts...")

    # Empty dict
    assert serialize({}) == "{}"
    assert deserialize("{}") == {}

    # Simple dict
    d = {"x": 1, "y": "a"}
    serialized = serialize(d)  # type: ignore
    deserialized = deserialize(serialized)
    assert deserialized == d

    # Dict with Ref
    d = {3: Ref(90)}
    serialized = serialize(d)  # type: ignore
    deserialized = deserialize(serialized)
    assert deserialized == d

    print("✓ Dicts passed")


def test_sets():
    """Test set serialization."""
    print("Testing sets...")

    # Empty set
    assert serialize(set()) == "set()"
    assert deserialize("set()") == set()

    # Simple set
    s = {1, 3}
    serialized = serialize(s)  # type: ignore
    deserialized = deserialize(serialized)
    assert deserialized == s

    # Set with Ref
    s = {1, 3, Ref(3)}
    serialized = serialize(s)
    deserialized = deserialize(serialized)
    assert deserialized == s

    print("✓ Sets passed")


def test_objects():
    """Test Object serialization."""
    print("Testing Objects...")

    # Empty object
    obj = Object(_class="Foo", attributes={})
    serialized = serialize(obj)
    assert serialized == "Object[Foo]({})", f"Expected 'Object[Foo]({{}})', got '{serialized}'"
    deserialized = deserialize(serialized)
    assert isinstance(deserialized, Object)
    assert deserialized._class == "Foo"
    assert deserialized.attributes == {}

    # Object with attributes
    obj = Object(_class="MyClass", attributes={"attr_1": "bob", "age": 3})
    serialized = serialize(obj)
    deserialized = deserialize(serialized)
    assert isinstance(deserialized, Object)
    assert deserialized._class == "MyClass"
    assert deserialized.attributes == {"attr_1": "bob", "age": 3}

    print("✓ Objects passed")


def test_classes():
    """Test Class serialization."""
    print("Testing Classes...")

    # Simple class with no bases
    cls = Class(name="Foo", bases=(), annotations={}, attributes={})
    serialized = serialize(cls)
    assert serialized == "Class[Foo]((), {}, {})", f"Expected 'Class[Foo]((), {{}}, {{}})', got '{serialized}'"
    deserialized = deserialize(serialized)
    assert isinstance(deserialized, Class)
    assert deserialized.name == "Foo"
    assert deserialized.bases == ()
    assert deserialized.annotations == {}
    assert deserialized.attributes == {}

    # Class with bases, annotations, and attributes
    cls = Class(
        name="MyClass",
        bases=(Ref(1), Ref(2)),
        annotations={"attr_1": "str", "age": "int"},
        attributes={"class_attr1": "alice", "test": None},
    )
    serialized = serialize(cls)
    deserialized = deserialize(serialized)
    assert isinstance(deserialized, Class)
    assert deserialized.name == "MyClass"
    assert deserialized.bases == (Ref(1), Ref(2))
    assert deserialized.annotations == {"attr_1": "str", "age": "int"}
    assert deserialized.attributes == {"class_attr1": "alice", "test": None}

    print("✓ Classes passed")


def test_special_cases():
    """Test special cases like Success and EffectError."""
    print("Testing special cases...")

    # Success
    assert serialize(Success()) == "Success"

    # EffectError
    err = EffectError("test error")
    serialized = serialize(err)
    assert "test error" in serialized

    print("✓ Special cases passed")

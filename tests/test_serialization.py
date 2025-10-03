import json
from typing import Any, Dict, List, Set, Tuple

import pytest

from nightjarpy.configs import LLMConfig
from nightjarpy.context import Context
from nightjarpy.prompts.base import PromptTemplate
from nightjarpy.types import (
    NJ_VAR_PREFIX,
    Class,
    EffectError,
    Func,
    JsonType,
    Object,
    Param,
    Ref,
    RegName,
    Signature,
    Success,
)
from nightjarpy.utils.utils import deserialize_json, serialize_json


@pytest.fixture
def context():
    """Create a minimal Context for testing serialization."""
    llm_config = LLMConfig()
    prompt_template = PromptTemplate(system="test")
    return Context(
        temp_var_init=0,
        valid_vars=set(),
        output_vars=set(),
        valid_labels=set(),
        python_frame=None,
        llm_config=llm_config,
        compute_prompt_template=prompt_template,
        use_functions=True,
    )


class TestPrimitiveSerialization:
    """Test serialization and deserialization of primitive types."""

    def test_none_serialization(self, context):
        """Test that None serializes and deserializes correctly."""
        original = None
        serialized = serialize_json(original)
        deserialized = deserialize_json(json.loads(serialized), context)

        assert deserialized is None
        assert original == deserialized

    def test_string_serialization(self, context):
        """Test that strings serialize and deserialize correctly."""
        test_cases = [
            "",
            "hello",
            "world",
            "special chars: !@#$%^&*()",
            "unicode: 你好世界",
            "newlines:\nand\ttabs",
        ]

        for original in test_cases:
            serialized = serialize_json(original)
            deserialized = deserialize_json(json.loads(serialized), context)

            assert deserialized == original
            assert isinstance(deserialized, str)

    def test_int_serialization(self, context):
        """Test that integers serialize and deserialize correctly."""
        test_cases = [
            0,
            1,
            -1,
            42,
            -42,
            2**31 - 1,  # Max 32-bit int
            -(2**31),  # Min 32-bit int
        ]

        for original in test_cases:
            serialized = serialize_json(original)
            deserialized = deserialize_json(json.loads(serialized), context)

            assert deserialized == original
            assert isinstance(deserialized, int)

    def test_float_serialization(self, context):
        """Test that floats serialize and deserialize correctly."""
        test_cases = [
            0.0,
            1.0,
            -1.0,
            3.14159,
            -3.14159,
            1e10,
            -1e-10,
            float("inf"),
            float("-inf"),
        ]

        for original in test_cases:
            serialized = serialize_json(original)
            deserialized = deserialize_json(json.loads(serialized), context)

            assert deserialized == original
            assert isinstance(deserialized, float)

    def test_bool_serialization(self, context):
        """Test that booleans serialize and deserialize correctly."""
        test_cases = [True, False]

        for original in test_cases:
            serialized = serialize_json(original)
            deserialized = deserialize_json(json.loads(serialized), context)

            assert deserialized == original
            assert isinstance(deserialized, bool)


class TestRefSerialization:
    """Test serialization and deserialization of Ref objects."""

    def test_ref_serialization(self, context):
        """Test that Ref objects serialize and deserialize correctly."""
        test_cases = [
            Ref(addr=0),
            Ref(addr=1),
            Ref(addr=42),
            Ref(addr=-1),
        ]

        for original in test_cases:
            serialized = serialize_json(original)
            deserialized = deserialize_json(json.loads(serialized), context)

            assert deserialized == original
            assert isinstance(deserialized, Ref)
            assert deserialized.addr == original.addr

    def test_ref_json_value(self):
        """Test that Ref.json_value() produces correct JSON structure."""
        ref = Ref(addr=42)
        json_val = ref.json_value()

        expected = {
            "type": "Ref",
            "addr": 42,
        }
        assert json_val == expected

    def test_ref_from_json(self):
        """Test that Ref.from_json() correctly deserializes JSON."""
        json_data = {
            "type": "Ref",
            "addr": 42,
        }

        ref = Ref.from_json(json_data)
        assert ref.addr == 42
        assert isinstance(ref, Ref)

    def test_ref_from_json_invalid(self):
        """Test that Ref.from_json() raises appropriate errors for invalid JSON."""
        # Missing type field
        with pytest.raises(ValueError, match="Unknown value type"):
            Ref.from_json({"addr": 42})

        # Wrong type
        with pytest.raises(ValueError, match="Unexpected serialization"):
            Ref.from_json({"type": "NotRef", "addr": 42})

        # Missing addr field
        with pytest.raises(ValueError, match="Unexpected serialization"):
            Ref.from_json({"type": "Ref"})

        # Invalid addr type
        with pytest.raises(ValueError, match="Unexpected serialization"):
            Ref.from_json({"type": "Ref", "addr": "not_a_number"})

        # Not a dict
        with pytest.raises(ValueError, match="Unexpected serialization"):
            Ref.from_json("not_a_dict")


class TestRegNameSerialization:
    """Test serialization and deserialization of RegName objects."""

    # def test_regname_serialization(self):
    #     """Test that RegName objects serialize and deserialize correctly."""
    #     test_cases = [
    #         RegName(name="x"),
    #         RegName(name="y"),
    #         RegName(name="variable_name"),
    #         RegName(name=""),
    #     ]

    #     for original in test_cases:
    #         serialized = serialize(original)
    #         deserialized = deserialize(json.loads(serialized))

    #         assert deserialized == original
    #         assert isinstance(deserialized, RegName)
    #         assert deserialized.name == original.name

    def test_regname_json_value(self):
        """Test that RegName.json_value() produces correct JSON structure."""
        regname = RegName(name="test_var")
        json_val = regname.json_value()

        expected = {
            "type": "Register",
            "name": "test_var",
        }
        assert json_val == expected

    def test_regname_from_json(self):
        """Test that RegName.from_json() correctly deserializes JSON."""
        json_data: JsonType = {
            "type": "Register",
            "name": "test_var",
        }

        regname = RegName.from_json(json_data)
        assert regname.name == "test_var"
        assert isinstance(regname, RegName)

    def test_regname_from_json_invalid(self):
        """Test that RegName.from_json() raises appropriate errors for invalid JSON."""
        # Missing type field
        with pytest.raises(ValueError, match="Unknown value type"):
            RegName.from_json({"name": "test"})

        # Wrong type
        with pytest.raises(ValueError, match="Unexpected serialization"):
            RegName.from_json({"type": "NotRegister", "name": "test"})

        # Missing name field
        with pytest.raises(ValueError, match="Unexpected serialization"):
            RegName.from_json({"type": "Register"})

        # Invalid name type
        with pytest.raises(ValueError, match="Unexpected serialization"):
            RegName.from_json({"type": "Register", "name": 123})


class TestCollectionSerialization:
    """Test serialization and deserialization of collection types."""

    def test_tuple_serialization(self, context):
        """Test that tuples serialize and deserialize correctly."""
        test_cases = [
            (),
            (1,),
            (1, 2, 3),
            ("a", "b", "c"),
            (1, "mixed", 3.14, True),
            (Ref(addr=1), Ref(addr=2)),
        ]

        for original in test_cases:
            serialized = serialize_json(original)
            deserialized = deserialize_json(json.loads(serialized), context)

            assert deserialized == original
            assert isinstance(deserialized, tuple)

    def test_list_serialization(self, context):
        """Test that lists serialize and deserialize correctly."""
        test_cases = [
            [],
            [1],
            [1, 2, 3],
            ["a", "b", "c"],
            [1, "mixed", 3.14, True],
            [Ref(addr=1), Ref(addr=2)],
        ]

        for original in test_cases:
            serialized = serialize_json(original)
            deserialized = deserialize_json(json.loads(serialized), context)

            assert deserialized == original
            assert isinstance(deserialized, list)

    def test_dict_serialization(self, context):
        """Test that dictionaries serialize and deserialize correctly."""
        test_cases = [
            {},
            {"a": 1},
            {"a": 1, "b": 2, "c": 3},
            {1: "one", 2: "two"},
            {Ref(addr=1): "ref_key", "string_key": Ref(addr=2)},
        ]

        for original in test_cases:
            serialized = serialize_json(original)
            deserialized = deserialize_json(json.loads(serialized), context)

            assert deserialized == original
            assert isinstance(deserialized, dict)

    def test_set_serialization(self, context):
        """Test that sets serialize and deserialize correctly."""
        test_cases = [
            set(),
            {1},
            {1, 2, 3},
            {"a", "b", "c"},
            {1, "mixed", 3.14, True},
            {Ref(addr=1), Ref(addr=2)},
        ]

        for original in test_cases:
            serialized = serialize_json(original)
            deserialized = deserialize_json(json.loads(serialized), context)

            assert deserialized == original
            assert isinstance(deserialized, set)


class TestObjectSerialization:
    """Test serialization and deserialization of Object instances."""

    def test_object_serialization(self, context):
        """Test that Object instances serialize and deserialize correctly."""
        test_cases = [
            Object(_class="TestClass", attributes={}),
            Object(_class="TestClass", attributes={"attr1": "value1"}),
            Object(_class="TestClass", attributes={"attr1": 1, "attr2": "value2"}),
            Object(_class="RefClass", attributes={"ref_attr": Ref(addr=42)}),
        ]

        for original in test_cases:
            serialized = serialize_json(original)
            deserialized = deserialize_json(json.loads(serialized), context)

            assert deserialized == original
            assert isinstance(deserialized, Object)
            assert deserialized._class == original._class
            assert deserialized.attributes == original.attributes

    def test_class_serialization(self, context):
        """Test that Class instances serialize and deserialize correctly."""
        test_cases = [
            Class(name="TestClass", annotations={}, attributes={}, bases=()),
            Class(name="TestClass", annotations={"x": "int"}, attributes={"x": 42}, bases=()),
            Class(
                name="ComplexClass",
                annotations={"name": "str", "age": "int", "active": "bool"},
                attributes={"name": "Alice", "age": 30, "active": True},
                bases=(),
            ),
            Class(name="RefClass", annotations={"ref": "Ref"}, attributes={"ref": Ref(addr=42)}, bases=()),
        ]

        for original in test_cases:
            serialized = serialize_json(original)
            deserialized = deserialize_json(json.loads(serialized), context)

            assert deserialized == original
            assert isinstance(deserialized, Class)
            assert deserialized.name == original.name
            assert deserialized.annotations == original.annotations
            assert deserialized.attributes == original.attributes


class TestFuncSerialization:
    """Test serialization and deserialization of Func instances."""

    def test_simple_func_serialization(self, context):
        """Test that simple Func instances serialize and deserialize correctly."""
        func_body = """def add(a, b):
    return a + b"""

        original = Func(
            context=context,
            name="add",
            signature=Signature(
                params=(
                    Param(name="a", annotation="int", kind="positional or keyword", default=None),
                    Param(name="b", annotation="int", kind="positional or keyword", default=None),
                )
            ),
            full_func=func_body,
        )

        serialized = serialize_json(original)
        deserialized = deserialize_json(json.loads(serialized), context)

        assert isinstance(deserialized, Func)
        assert deserialized.name == original.name
        assert deserialized.full_func == original.full_func
        assert len(deserialized.signature.params) == len(original.signature.params)
        assert deserialized.signature.params[0].name == "a"
        assert deserialized.signature.params[1].name == "b"

    def test_func_with_defaults_serialization(self, context):
        """Test that Func instances with default parameters serialize correctly."""
        func_body = """def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!" """

        original = Func(
            context=context,
            name="greet",
            signature=Signature(
                params=(
                    Param(name="name", annotation="str", kind="positional or keyword", default=None),
                    Param(name="greeting", annotation="str", kind="positional or keyword", default="Hello"),
                )
            ),
            full_func=func_body,
        )

        serialized = serialize_json(original)
        deserialized = deserialize_json(json.loads(serialized), context)
        assert isinstance(deserialized, Func)
        assert deserialized.name == original.name
        assert len(deserialized.signature.params) == 2
        assert deserialized.signature.params[0].default is None
        assert deserialized.signature.params[1].default == "Hello"

    def test_func_with_various_param_kinds(self, context):
        """Test that Func instances with various parameter kinds serialize correctly."""
        func_body = """def complex_func(pos_only, /, pos_or_kw, *args, kw_only, **kwargs):
    return (pos_only, pos_or_kw, args, kw_only, kwargs)"""

        original = Func(
            context=context,
            name="complex_func",
            signature=Signature(
                params=(
                    Param(name="pos_only", annotation="Any", kind="positional-only", default=None),
                    Param(name="pos_or_kw", annotation="Any", kind="positional or keyword", default=None),
                    Param(name="args", annotation="Any", kind="variadic positional", default=None),
                    Param(name="kw_only", annotation="Any", kind="keyword-only", default=None),
                    Param(name="kwargs", annotation="Any", kind="variadic keyword", default=None),
                )
            ),
            full_func=func_body,
        )

        serialized = serialize_json(original)
        deserialized = deserialize_json(json.loads(serialized), context)

        assert isinstance(deserialized, Func)
        assert len(deserialized.signature.params) == 5
        assert deserialized.signature.params[0].kind == "positional-only"
        assert deserialized.signature.params[1].kind == "positional or keyword"
        assert deserialized.signature.params[2].kind == "variadic positional"
        assert deserialized.signature.params[3].kind == "keyword-only"
        assert deserialized.signature.params[4].kind == "variadic keyword"

    def test_func_json_structure(self, context):
        """Test that Func serialization produces correct JSON structure."""
        func_body = """def test_func(x):
    return x * 2"""

        func = Func(
            context=context,
            name="test_func",
            signature=Signature(
                params=(Param(name="x", annotation="int", kind="positional or keyword", default=None),)
            ),
            full_func=func_body,
        )

        serialized = serialize_json(func)
        json_data = json.loads(serialized)

        assert json_data["type"] == "Func"
        assert json_data["name"] == f"test_func"
        assert json_data["full_func"] == func_body.replace("def test_func", f"def test_func")
        assert isinstance(json_data["signature"], list)
        assert len(json_data["signature"]) == 1
        assert json_data["signature"][0]["name"] == "x"
        assert json_data["signature"][0]["annotation"] == "int"
        assert json_data["signature"][0]["kind"] == "positional or keyword"


class TestSpecialValues:
    """Test serialization and deserialization of special values."""

    def test_success_serialization(self):
        """Test that Success objects serialize correctly."""
        success = Success()
        serialized = serialize_json(success)

        assert serialized == "Success"

    def test_effect_error_serialization(self):
        """Test that EffectError objects serialize correctly."""
        error = EffectError("Test error message")
        serialized = serialize_json(error)

        assert serialized == str(error)


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_deserialize_invalid_json(self, context):
        """Test that deserializing invalid JSON raises appropriate errors."""
        # Invalid JSON structure
        with pytest.raises(ValueError, match="Unknown value type"):
            deserialize_json({"invalid": "structure"}, context)

        # Missing type field for complex types
        with pytest.raises(ValueError, match="Unknown value type"):
            deserialize_json({"items": [1, 2, 3]}, context)

    def test_deserialize_invalid_list(self, context):
        """Test that deserializing invalid list structures raises errors."""
        # List without items field
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json({"type": "list"}, context)

        # List with non-list items
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json({"type": "list", "items": "not_a_list"}, context)

    def test_deserialize_invalid_dict(self, context):
        """Test that deserializing invalid dict structures raises errors."""
        # Dict without items field
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json({"type": "dict"}, context)

        # Dict with non-list items
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json({"type": "dict", "items": "not_a_list"}, context)

        # Dict items without key/value structure
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json({"type": "dict", "items": [{"invalid": "structure"}]}, context)

    def test_deserialize_invalid_set(self, context):
        """Test that deserializing invalid set structures raises errors."""
        # Set without items field
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json({"type": "set"}, context)

        # Set with non-list items
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json({"type": "set", "items": "not_a_list"}, context)

    def test_deserialize_invalid_object(self, context):
        """Test that deserializing invalid Object structures raises errors."""
        # Object without class field
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json({"type": "Object", "attributes": []}, context)

        # Object without attributes field
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json({"type": "Object", "class": "TestClass"}, context)

        # Object with invalid attributes structure
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json(
                {"type": "Object", "class": "TestClass", "attributes": [{"invalid": "structure"}]}, context
            )

    def test_deserialize_invalid_class(self, context):
        """Test that deserializing invalid Class structures raises errors."""
        # Class without name field
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json({"type": "Class", "annotations": [], "attributes": []}, context)

        # Class without annotations field
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json({"type": "Class", "name": "TestClass", "attributes": []}, context)

        # Class without attributes field
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json({"type": "Class", "name": "TestClass", "annotations": []}, context)

        # Class with invalid annotations structure
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json(
                {"type": "Class", "name": "TestClass", "annotations": [{"invalid": "structure"}], "attributes": []},
                context,
            )

        # Class with invalid attributes structure
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json(
                {"type": "Class", "name": "TestClass", "annotations": [], "attributes": [{"invalid": "structure"}]},
                context,
            )

    def test_deserialize_invalid_func(self, context):
        """Test that deserializing invalid Func structures raises errors."""
        # Func without name field
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json({"type": "Func", "full_func": "def f(): pass", "signature": []}, context)

        # Func without full_func field
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json({"type": "Func", "name": "test_func", "signature": []}, context)

        # Func without signature field
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json({"type": "Func", "name": "test_func", "full_func": "def f(): pass"}, context)

        # Func with invalid signature structure (non-list)
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json(
                {"type": "Func", "name": "test_func", "full_func": "def f(): pass", "signature": "invalid"}, context
            )

        # Func with invalid parameter structure
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json(
                {
                    "type": "Func",
                    "name": "test_func",
                    "full_func": "def f(): pass",
                    "signature": [{"invalid": "param"}],
                },
                context,
            )

        # Func with invalid parameter kind
        with pytest.raises(ValueError, match="Unexpected serialization"):
            deserialize_json(
                {
                    "type": "Func",
                    "name": "test_func",
                    "full_func": "def f(): pass",
                    "signature": [{"name": "x", "annotation": "int", "kind": "invalid_kind", "default": None}],
                },
                context,
            )


if __name__ == "__main__":
    pytest.main([__file__])

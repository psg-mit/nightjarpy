from typing import Any, Dict, List, Set, Tuple

import pytest

from nightjarpy.configs import LLMConfig
from nightjarpy.context import Context
from nightjarpy.prompts.base import PromptTemplate
from nightjarpy.types import (
    NJ_VAR_PREFIX,
    Class,
    Func,
    Immutable,
    Object,
    Param,
    Primitive,
    Ref,
    Signature,
)


def create_test_context():
    """Helper function to create a test context with required parameters."""
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


class TestPrimitiveEncoding:
    """Test encoding and decoding of primitive types."""

    def test_none_encoding(self):
        """Test that None encodes and decodes correctly."""
        context = create_test_context()

        original = None
        encoded = context.encode_python_value(original, {})
        decoded = context.decode_and_sync_python_value(encoded, {})

        assert decoded is None
        assert original == decoded
        assert isinstance(encoded, Primitive)

    def test_string_encoding(self):
        """Test that strings encode and decode correctly."""
        context = create_test_context()

        test_cases = [
            "",
            "hello",
            "world",
            "special chars: !@#$%^&*()",
            "unicode: 你好世界",
            "newlines:\nand\ttabs",
        ]

        for original in test_cases:
            encoded = context.encode_python_value(original, {})
            decoded = context.decode_and_sync_python_value(encoded, {})

            assert decoded == original
            assert isinstance(decoded, str)
            assert isinstance(encoded, Primitive)

    def test_int_encoding(self):
        """Test that integers encode and decode correctly."""
        context = create_test_context()

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
            encoded = context.encode_python_value(original, {})
            decoded = context.decode_and_sync_python_value(encoded, {})

            assert decoded == original
            assert isinstance(decoded, int)
            assert isinstance(encoded, Primitive)

    def test_float_encoding(self):
        """Test that floats encode and decode correctly."""
        context = create_test_context()

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
            encoded = context.encode_python_value(original, {})
            decoded = context.decode_and_sync_python_value(encoded, {})

            assert decoded == original
            assert isinstance(decoded, float)
            assert isinstance(encoded, Primitive)

    def test_bool_encoding(self):
        """Test that booleans encode and decode correctly."""
        context = create_test_context()

        test_cases = [True, False]

        for original in test_cases:
            encoded = context.encode_python_value(original, {})
            decoded = context.decode_and_sync_python_value(encoded, {})

            assert decoded == original
            assert isinstance(decoded, bool)
            assert isinstance(encoded, Primitive)


class TestCollectionEncoding:
    """Test encoding and decoding of collection types."""

    def test_tuple_encoding(self):
        """Test that tuples encode and decode correctly."""
        context = create_test_context()

        test_cases = [
            (),
            (1,),
            (1, 2, 3),
            ("a", "b", "c"),
            (1, "mixed", 3.14, True),
        ]

        for original in test_cases:
            encoded = context.encode_python_value(original, {})
            decoded = context.decode_and_sync_python_value(encoded, {})

            assert decoded == original
            assert isinstance(decoded, tuple)
            assert isinstance(encoded, tuple)

    def test_list_encoding(self):
        """Test that lists encode and decode correctly."""
        context = create_test_context()

        test_cases = [
            [],
            [1],
            [1, 2, 3],
            ["a", "b", "c"],
            [1, "mixed", 3.14, True],
            [[1, 2], [3, 4]],  # Nested lists
        ]

        for original in test_cases:
            encoded = context.encode_python_value(original, {})
            decoded = context.decode_and_sync_python_value(encoded, {})

            assert decoded == original
            assert isinstance(decoded, list)
            assert isinstance(encoded, Ref)

    def test_dict_encoding(self):
        """Test that dictionaries encode and decode correctly."""
        context = create_test_context()

        test_cases = [
            {},
            {"a": 1},
            {"a": 1, "b": 2, "c": 3},
            {1: "one", 2: "two"},
            {"nested": {"inner": "value"}},
        ]

        for original in test_cases:
            encoded = context.encode_python_value(original, {})
            decoded = context.decode_and_sync_python_value(encoded, {})

            assert decoded == original
            assert isinstance(decoded, dict)
            assert isinstance(encoded, Ref)

    def test_set_encoding(self):
        """Test that sets encode and decode correctly."""
        context = create_test_context()

        test_cases = [
            set(),
            {1},
            {1, 2, 3},
            {"a", "b", "c"},
            {1, "mixed", 3.14, True},
        ]

        for original in test_cases:
            encoded = context.encode_python_value(original, {})
            decoded = context.decode_and_sync_python_value(encoded, {})

            assert decoded == original
            assert isinstance(decoded, set)
            assert isinstance(encoded, Ref)


class TestObjectEncoding:
    """Test encoding and decoding of custom objects."""

    def test_simple_object_encoding(self):
        """Test that simple objects encode and decode correctly."""
        context = create_test_context()

        class SimpleClass:
            def __init__(self, value: int):
                self.value = value
                self.name = "test"

        original = SimpleClass(42)
        encoded = context.encode_python_value(original, {})
        decoded = context.decode_and_sync_python_value(encoded, {})

        assert decoded.value == original.value
        assert decoded.name == original.name
        assert isinstance(decoded, SimpleClass)
        assert isinstance(encoded, Ref)

    def test_object_with_mixed_attributes(self):
        """Test objects with various attribute types."""
        context = create_test_context()

        class MixedClass:
            def __init__(self):
                self.string_attr = "hello"
                self.int_attr = 42
                self.float_attr = 3.14
                self.bool_attr = True
                self.list_attr = [1, 2, 3]
                self.dict_attr = {"key": "value"}

        original = MixedClass()
        encoded = context.encode_python_value(original, {})
        decoded = context.decode_and_sync_python_value(encoded, {})

        assert decoded.string_attr == original.string_attr
        assert decoded.int_attr == original.int_attr
        assert decoded.float_attr == original.float_attr
        assert decoded.bool_attr == original.bool_attr
        assert decoded.list_attr == original.list_attr
        assert decoded.dict_attr == original.dict_attr
        assert isinstance(decoded, MixedClass)

    def test_object_with_methods(self):
        """Test that objects with methods are handled correctly."""
        context = create_test_context()

        class ClassWithMethods:
            def __init__(self, value: int):
                self.value = value

            def get_value(self):
                return self.value

            def __private_method(self):
                return "private"

        original = ClassWithMethods(42)
        encoded = context.encode_python_value(original, {})
        decoded = context.decode_and_sync_python_value(encoded, {})

        assert decoded.value == original.value
        assert isinstance(decoded, ClassWithMethods)
        # Methods should still be available
        assert decoded.get_value() == 42

    def test_class(self):
        """Test that objects with methods are handled correctly."""
        context = create_test_context()

        class TestClass:
            value: int

            def __init__(self, value: int):
                self.value = value

        encoded = context.encode_python_value(TestClass, {})
        decoded = context.decode_and_sync_python_value(encoded, {})

        assert decoded == TestClass


class TestNestedStructures:
    """Test encoding and decoding of complex nested structures."""

    def test_deeply_nested_structures(self):
        """Test deeply nested structures with mixed types."""
        context = create_test_context()

        original = {
            "level1": {
                "level2": {
                    "level3": [
                        {"item": 1, "nested": {"inner": "value"}},
                        {"item": 2, "nested": {"inner": "value2"}},
                    ]
                },
                "tuple": (1, 2, 3),
                "set": {1, 2, 3},
            },
            "list_of_objects": [
                {"attr": "value1"},
                {"attr": "value2"},
            ],
        }

        encoded = context.encode_python_value(original, {})
        decoded = context.decode_and_sync_python_value(encoded, {})

        assert decoded == original

    def test_object_with_nested_collections(self):
        """Test objects containing nested collections."""
        context = create_test_context()

        class NestedClass:
            def __init__(self):
                self.data = {
                    "list": [1, 2, 3],
                    "dict": {"nested": {"value": 42}},
                    "set": {1, 2, 3},
                    "tuple": (1, 2, 3),
                }

        original = NestedClass()
        encoded = context.encode_python_value(original, {})
        decoded = context.decode_and_sync_python_value(encoded, {})

        assert decoded.data == original.data
        assert isinstance(decoded, NestedClass)


class TestSynchronization:
    """Test synchronization behavior with original objects."""

    def test_list_synchronization(self):
        """Test that changes to encoded lists are synchronized back to original."""
        context = create_test_context()

        original = [1, 2, 3]
        encoded = context.encode_python_value(original, {})
        assert isinstance(encoded, Ref)

        # Modify the encoded list
        encoded_list = context.heap.get(encoded)
        assert isinstance(encoded_list, list)
        encoded_list.append(4)
        encoded_list[0] = 0

        # Decode and sync
        decoded = context.decode_and_sync_python_value(encoded, {})

        # Original should be updated
        assert original == [0, 2, 3, 4]
        assert decoded is original

    def test_dict_synchronization(self):
        """Test that changes to encoded dicts are synchronized back to original."""
        context = create_test_context()

        original = {"a": 1, "b": 2}
        encoded = context.encode_python_value(original, {})
        assert isinstance(encoded, Ref)

        # Modify the encoded dict
        encoded_dict = context.heap.get(encoded)
        assert isinstance(encoded_dict, dict)
        encoded_dict["c"] = 3
        encoded_dict["a"] = 0
        del encoded_dict["b"]

        # Decode and sync
        decoded = context.decode_and_sync_python_value(encoded, {})

        # Original should be updated
        assert original == {"a": 0, "c": 3}
        assert decoded is original

    def test_set_synchronization(self):
        """Test that changes to encoded sets are synchronized back to original."""
        context = create_test_context()

        original = {1, 2, 3}
        encoded = context.encode_python_value(original, {})
        assert isinstance(encoded, Ref)

        # Modify the encoded set
        encoded_set = context.heap.get(encoded)
        assert isinstance(encoded_set, set)
        encoded_set.add(4)
        encoded_set.remove(1)

        # Decode and sync
        decoded = context.decode_and_sync_python_value(encoded, {})

        # Original should be updated
        assert original == {2, 3, 4}
        assert decoded is original

    def test_object_synchronization(self):
        """Test that changes to encoded objects are synchronized back to original."""
        context = create_test_context()

        class TestClass:
            def __init__(self):
                self.value = 42
                self.name = "test"

        original = TestClass()
        encoded = context.encode_python_value(original, {})
        assert isinstance(encoded, Ref)

        # Modify the encoded object
        encoded_obj = context.heap.get(encoded)
        assert isinstance(encoded_obj, Object)
        encoded_obj.attributes["value"] = 100
        encoded_obj.attributes["new_attr"] = "new_value"

        # Decode and sync
        decoded = context.decode_and_sync_python_value(encoded, {})

        # Original should be updated
        assert original.value == 100
        assert original.new_attr == "new_value"  # type: ignore
        assert decoded is original

    def test_class_synchronization(self):
        """Test that changes to encoded class objects are synchronized back to original."""
        context = create_test_context()

        class TestClass:
            class_attr = "original_value"

            def __init__(self):
                self.instance_attr = "instance_value"

        # Test with class object (not instance)
        original_class = TestClass
        encoded = context.encode_python_value(original_class, {})
        assert isinstance(encoded, Ref)

        # Modify the encoded class object
        encoded_class = context.heap.get(encoded)
        assert isinstance(encoded_class, Class)

        # Modify class attributes
        if "class_attr" in encoded_class.attributes:
            encoded_class.attributes["class_attr"] = "modified_value"
        encoded_class.attributes["new_class_attr"] = "new_class_value"

        # Decode and sync
        decoded = context.decode_and_sync_python_value(encoded, {})

        # Original class should be updated
        assert decoded is original_class
        assert hasattr(original_class, "new_class_attr")
        assert original_class.new_class_attr == "new_class_value"  # type: ignore


class TestFuncEncoding:
    """Test encoding and decoding of Func objects."""

    def test_simple_func_encoding(self):
        """Test that simple functions are handled correctly."""
        context = create_test_context()

        # Create a simple Python function
        def add(a, b):
            return a + b

        # Encode the function
        func_ref = context.encode_python_value(val=add, enc_memo={})

        assert isinstance(func_ref, Ref)

        encoded_func = context.deref(func_ref)

        # Verify it's a Func object with correct signature
        assert isinstance(encoded_func, Func)
        assert encoded_func.name == "add"
        assert len(encoded_func.signature.params) == 2

        # Check the parameters
        assert encoded_func.signature.params[0].name == "a"
        assert encoded_func.signature.params[1].name == "b"

        # Test that the function can be called
        result = encoded_func(2, 3)
        assert result == 5

        # Decode it back
        decoded_func = context.decode_and_sync_python_value(val=func_ref, dec_memo={})

        # Test that the function can be called
        result = decoded_func(2, 3)
        assert result == 5

    def test_func_with_defaults_encoding(self):
        """Test that functions with default parameters are handled correctly."""
        context = create_test_context()

        # Create a Python function with default parameters
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        # Encode the function
        func_ref = context.encode_python_value(val=greet, enc_memo={})

        assert isinstance(func_ref, Ref)

        encoded_func = context.deref(func_ref)

        # Verify it's a Func object with correct signature
        assert isinstance(encoded_func, Func)
        assert len(encoded_func.signature.params) == 2

        # Check the parameters
        name_param = encoded_func.signature.params[0]
        greeting_param = encoded_func.signature.params[1]

        assert name_param.name == "name"
        assert name_param.default is None

        assert greeting_param.name == "greeting"
        assert greeting_param.default == "Hello"

        # Decode it back
        decoded_func = context.decode_and_sync_python_value(val=func_ref, dec_memo={})

        # Test that the function can be called
        result = decoded_func("World")
        assert result == "Hello, World!"

        result = decoded_func("World", "Hi")
        assert result == "Hi, World!"

    def test_func_in_collection(self):
        """Test that functions can be stored in collections."""
        context = create_test_context()

        # Create a Python function
        def multiply(x, y):
            return x * y

        # Store function in a list
        func_list = [multiply, "other_value", 42]
        list_ref = context.encode_python_value(val=func_list, enc_memo={})

        # Retrieve and verify
        assert isinstance(list_ref, Ref)
        retrieved_list = context.deref(list_ref)
        assert isinstance(retrieved_list, list)
        assert len(retrieved_list) == 3

        # First item should be a Ref to the function
        assert isinstance(retrieved_list[0], Ref)
        func_from_list = context.deref(retrieved_list[0])
        assert isinstance(func_from_list, Func)

        # Decode the list back
        decoded_list = context.decode_and_sync_python_value(val=list_ref, dec_memo={})
        assert isinstance(decoded_list, list)
        assert len(decoded_list) == 3
        assert decoded_list[0](3, 4) == 12
        assert decoded_list[1] == "other_value"
        assert decoded_list[2] == 42

    def test_object_with_methods_encoding(self):
        """Test that objects with methods can be encoded and decoded, and methods can be called."""
        context = create_test_context()

        # Create a class with methods
        class Calculator:
            def __init__(self, initial_value=0):
                self.value = initial_value

            def add(self, x):
                self.value += x
                return self.value

            def multiply(self, x):
                self.value *= x
                return self.value

            def get_value(self):
                return self.value

        # Create an instance
        calc = Calculator(initial_value=10)

        # Encode the object
        obj_ref = context.encode_python_value(val=calc, enc_memo={})

        # Verify it's a reference
        assert isinstance(obj_ref, Ref)

        # Decode it back
        decoded_calc = context.decode_and_sync_python_value(val=obj_ref, dec_memo={})

        # Verify the decoded object has the correct initial value
        assert decoded_calc.get_value() == 10

        # Test calling methods on the decoded object
        result = decoded_calc.add(5)
        assert result == 15
        assert decoded_calc.get_value() == 15

        result = decoded_calc.multiply(2)
        assert result == 30
        assert decoded_calc.get_value() == 30


class TestErrorHandling:
    """Test error handling for unsupported types and edge cases."""


class TestRoundTripConsistency:
    """Test that encoding and decoding are consistent."""

    def test_round_trip_consistency(self):
        """Test that encoding and decoding preserves the original value."""
        context = create_test_context()

        class TestClass:
            def __init__(self, value: int):
                self.value = value
                self.nested = {"list": [1, 2, 3], "set": {1, 2, 3}}

        test_values = [
            None,
            "hello",
            42,
            3.14,
            True,
            (1, 2, 3),
            [1, 2, 3],
            {"a": 1, "b": 2},
            {1, 2, 3},
            TestClass(42),
            # Complex nested structure
            {
                "primitives": [1, "string", 3.14, True, None],
                "collections": {
                    "list": [1, 2, 3],
                    "dict": {"nested": "value"},
                    "set": {1, 2, 3},
                    "tuple": (1, 2, 3),
                },
                "object": TestClass(100),
            },
        ]

        for original in test_values:
            encoded = context.encode_python_value(original, {})
            decoded = context.decode_and_sync_python_value(encoded, {})

            if isinstance(original, dict):
                assert decoded == original, f"Round trip failed for {original}"
            elif hasattr(original, "__dict__"):
                assert decoded.__dict__ == original.__dict__, f"Round trip failed for {original}"
            else:
                assert decoded == original, f"Round trip failed for {original}"


if __name__ == "__main__":
    pytest.main([__file__])

import pytest

import nightjarpy


@nightjarpy.fn()
def deref_list():
    items = [1, 2, 3, 4, 5]
    """natural
    Save the first value in <items> as <:x>
    """
    return x


@nightjarpy.fn()
def deref_dict():
    data = {"name": "Alice", "age": 30, "city": "New York"}
    """natural
    Get the value of city in <data> as <:y>
    """
    return y


@nightjarpy.fn()
def deref_tuple():
    coordinates = (10, 20, 30)
    """natural
    Get the z value of <coordinates> as <:z>
    """
    return z


@nightjarpy.fn()
def deref_set():
    unique_numbers = {1, 2, 3, 4, 5}
    """natural
    Get the largest number in <unique_numbers> as <:num>
    """
    return num


@nightjarpy.fn()
def deref_nested_collection():
    nested = {"list": [1, 2, 3], "tuple": (4, 5, 6), "inner_dict": {"a": 1, "b": 2}}
    """natural
    Get the value of b in inner_dict in <nested> as <:n>
    """
    return n


@nightjarpy.fn()
def deref_object_attribute():
    class Car:
        def __init__(self, make, model, year):
            self.make = make
            self.model = model
            self.year = year

    car = Car("Toyota", "Camry", 2020)
    """natural
    Get the make attribute of <car> as <:make>
    """
    return make


def test_deref_list():
    """Test that dereferencing a list works correctly."""
    result = deref_list()
    assert result == 1


def test_deref_dict():
    """Test that dereferencing a dictionary works correctly."""
    result = deref_dict()
    assert result == "New York"


def test_deref_tuple():
    """Test that dereferencing a tuple works correctly."""
    result = deref_tuple()
    assert result == 30


def test_deref_set():
    """Test that dereferencing a set works correctly."""
    result = deref_set()
    assert result == 5


def test_deref_nested_collection():
    """Test that dereferencing a nested collection works correctly."""
    result = deref_nested_collection()
    assert result == 2


def test_deref_object_attribute():
    """Test that dereferencing an object attribute works correctly."""
    result = deref_object_attribute()
    assert result == "Toyota"


@nightjarpy.fn()
def deref_function():
    """Test dereferencing a function stored at a reference."""
    functions = {"add": lambda x, y: x + y, "multiply": lambda x, y: x * y, "subtract": lambda x, y: x - y}
    """natural
    Get the add function from <functions> as <:operation>
    """
    return operation


@nightjarpy.fn()
def deref_function_in_list():
    """Test dereferencing a function from a list."""
    operations = [lambda x: x + 1, lambda x: x * 2, lambda x: x**2]
    """natural
    Get the second function (index 1) from <operations> as <:func>
    """
    return func


# @nightjarpy.fn()
# def deref_function_and_call():
#     """Test dereferencing a function and calling it."""
#     calc = {"square": lambda n: n**2, "double": lambda n: n * 2}
#     x = 5
#     """natural
#     Get the square function from <calc> as <:square_func>
#     Call <square_func> with <x> and save the result as <:result>
#     """
#     return result


def test_deref_function():
    """Test that dereferencing a function from a dict works correctly."""
    result = deref_function()
    # The function should be the add function
    assert callable(result)
    assert result(3, 4) == 7


def test_deref_function_in_list():
    """Test that dereferencing a function from a list works correctly."""
    result = deref_function_in_list()
    # The second function (index 1) multiplies by 2
    assert callable(result)
    assert result(5) == 10


# Not implemented yet for base effects
# def test_deref_function_and_call():
#     """Test that dereferencing and calling a function works correctly."""
#     result = deref_function_and_call()
#     # Should be 5 squared = 25
#     assert result == 25

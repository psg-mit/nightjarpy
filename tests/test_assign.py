from dataclasses import dataclass

import pytest

import nightjarpy
from nightjarpy.configs import INTERPRETER_BASE_NOREG_JSON_CONFIG


@nightjarpy.fn()
def assign_x_to_10():
    x = 5
    """natural
    Set <:x> as 10
    """
    return x


@nightjarpy.fn()
def assign_y_to_x_plus_10():
    x = 10
    """natural
    Set <:y> as <x> + 10
    """
    return y


@nightjarpy.fn()
def assign_y_to_x():
    x = 10
    """natural
    Set <:y> as <x>
    """
    return y


@nightjarpy.fn(config=INTERPRETER_BASE_NOREG_JSON_CONFIG.disable_cache())
def assign_object_attribute():
    @dataclass
    class Person:
        name: str
        age: int

    """natural
    Create <Person> for Sarah born 25 years ago as <:person>
    """
    return person, Person


def test_assign_x_to_10():
    """Test that natural language assignment sets x to 10."""
    result = assign_x_to_10()
    assert result == 10


def test_assign_y_to_x_plus_10():
    """Test that natural language assignment sets y to x + 10."""
    result = assign_y_to_x_plus_10()
    assert result == 20


def test_assign_y_to_x():
    """Test that natural language assignment sets y to x."""
    result = assign_y_to_x()
    assert result == 10


def test_assign_object_attribute():
    """Test that natural language assignment works with object attributes."""
    result, Person = assign_object_attribute()
    assert isinstance(result, Person)
    assert result.name == "Sarah"
    assert result.age == 25

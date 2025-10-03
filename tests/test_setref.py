import pytest

import nightjarpy


@nightjarpy.fn()
def setref_list():
    items = [1, 2, 3]
    """natural
    Update the first item in <items> to 0
    """
    return items


@nightjarpy.fn()
def setref_dict():
    data = {"a": 1}
    """natural
    Set all entries in <data> dict have value = 0
    """
    return data


@nightjarpy.fn()
def setref_nested_collection():
    nested = {"a": 1}
    """natural
    Assign key "b" to value [0,0] in <nested>
    """
    return nested


@nightjarpy.fn()
def setref_object_attribute():
    class Student:
        def __init__(self, name, grade):
            self.name = name
            self.grade = grade

    student = Student("Bob", "B")
    """natural
    Update the grade attribute of <student> to "A"
    """
    return student.grade


def test_setref_list():
    """Test that updating a reference to a list works correctly."""
    result = setref_list()
    assert result == [0, 2, 3]


def test_setref_dict():
    """Test that updating a reference to a dict works correctly."""
    result = setref_dict()
    assert result["a"] == 0


def test_setref_nested_collection():
    """Test that updating a reference to a nested collection works correctly."""
    result = setref_nested_collection()
    assert result["b"] == [0, 0]


def test_setref_object_attribute():
    """Test that updating an object attribute works correctly."""
    result = setref_object_attribute()
    assert result == "A"

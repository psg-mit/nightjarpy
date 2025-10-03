import pytest

import nightjarpy


@nightjarpy.fn()
def ref_list():
    """natural
    Create a list [1, 2, 3] as <:l>
    """
    return l


@nightjarpy.fn()
def ref_list_update():
    """natural
    Create a list [1, 2, 3] as <:l>
    """

    l[0] = 0
    return l


@nightjarpy.fn()
def ref_list_update2():
    x = [1, 2, 3]
    y = x
    """natural
    Append 4 to <x>
    """

    return y


@nightjarpy.fn()
def ref_dict():
    """natural
    Create a dictionary where "a" = 1 as <:d>
    """
    return d


def test_ref_list():
    """Test that creating a list works correctly."""
    result = ref_list()
    assert result == [1, 2, 3]


def test_ref_list_update():
    """Test that created list is mutable."""
    result = ref_list_update()
    assert result == [0, 2, 3]


def test_ref_list_update2():
    """Test that mutable update works."""
    result = ref_list_update2()
    assert result == [1, 2, 3, 4]


def test_ref_dict():
    result = ref_dict()
    assert result["a"] == 1

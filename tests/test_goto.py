import pytest

import nightjarpy


@nightjarpy.fn()
def goto_break():
    for i in range(3):
        """natural
        |break| from loop
        """
    return i


@nightjarpy.fn()
def goto_continue():
    result = []
    for i in range(5):
        if i == 2:
            """natural
            |continue| to next iteration
            """
        result.append(i)
    return result


@nightjarpy.fn()
def goto_return_value():
    x = 42
    """natural
    |return| <x>
    """
    return "unreachable"


@nightjarpy.fn()
def goto_raise_value_error():
    # TODO: once objects are implemented, change this to let natural code create the error
    e = ValueError("Test error")
    """natural
    |raise| <e>
    """
    return "unreachable"


@nightjarpy.fn()
def goto_continue_in_nested_loop():
    result = []
    for i in range(3):
        for j in range(3):
            if j == 1:
                """natural
                |continue| to next iteration of inner loop
                """
            result.append((i, j))
    return result


@nightjarpy.fn()
def goto_break_in_nested_loop():
    result = []
    for i in range(3):
        for j in range(3):
            if j == 1:
                """natural
                |break| from inner loop
                """
            result.append((i, j))
    return result


@nightjarpy.fn()
def goto_return_in_loop():
    for i in range(5):
        if i == 2:
            """natural
            |return| <i>
            """
    return "unreachable"


def test_goto_break():
    """Test that break works correctly in a loop."""
    result = goto_break()
    assert result == 0


def test_goto_continue():
    """Test that continue works correctly in a loop."""
    result = goto_continue()
    assert result == [0, 1, 3, 4]  # Should skip index 2


def test_goto_return_value():
    """Test that return with a value works correctly."""
    result = goto_return_value()
    assert result == 42


def test_goto_raise_value_error():
    """Test that raise with ValueError works correctly."""
    with pytest.raises(ValueError, match="Test error"):
        goto_raise_value_error()


def test_goto_continue_in_nested_loop():
    """Test that continue works correctly in nested loops."""
    result = goto_continue_in_nested_loop()
    # Should skip j=1 in each iteration
    expected = [(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2)]
    assert result == expected


def test_goto_break_in_nested_loop():
    """Test that break works correctly in nested loops."""
    result = goto_break_in_nested_loop()
    # Should break from inner loop when j=1, so only j=0 for each i
    expected = [(0, 0), (1, 0), (2, 0)]
    assert result == expected


def test_goto_return_in_loop():
    """Test that return works correctly inside a loop."""
    result = goto_return_in_loop()
    assert result == 2

import pytest

import nightjarpy


@nightjarpy.fn
def lookup_variable():
    x = 5
    """natural
    Lookup <x>
    """
    return x


def test_lookup_variable():
    """Test that natural language lookup works correctly."""
    result = lookup_variable()
    assert result == 5, "Lookup should return the value of x"

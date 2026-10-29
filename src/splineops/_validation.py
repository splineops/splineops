"""Shared scalar validation for shapes, axes, and integer configuration."""

import operator

import numpy as np


def as_index(value) -> int:
    """Accept integer scalars without silently interpreting booleans as 0 or 1."""
    # Check before operator.index: Python's bool would otherwise become an int,
    # and NumPy's bool conversion differs between supported NumPy versions.
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("Expected an integer, not a boolean.")
    return operator.index(value)

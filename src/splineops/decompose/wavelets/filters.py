"""
filters.py
----------
Holds numeric arrays for various spline filters used in wavelets transformations.
"""

import numpy as np

class SplineFilter:
    """
    Example container that, given an order, holds the lowpass (h) and highpass (g).
    """
    def __init__(self, order:int):
        self.order = order
        # define h,g for each order
        if order == 1:
            self.h = np.array([
                # Some values, e.g. from your SplineFilter(1) ...
                0.8176464, 0.3972970, -0.0691009, ...
            ])
            self.g = np.array([
                # Possibly g is derived from h by +/- patterns ...
                # Or fill them directly
            ])
        elif order == 3:
            self.h = ...
            self.g = ...
        else:
            raise ValueError(f"SplineFilter order={order} not yet implemented.")

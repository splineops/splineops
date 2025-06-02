def bspline3(x: float) -> float:
    """Uniform cubic B-spline basis (|x| ≤ 2)."""
    X = abs(x)
    if X < 1.0:
        return 0.5 * X * X * (X - 2.0) + 2.0 / 3.0
    if X < 2.0:
        X -= 2.0
        return -X**3 / 6.0
    return 0.0

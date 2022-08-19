def is_cupy_type(x) -> bool:
    # TODO: avoid explicit reference to CuPy
    return "cupy" in str(type(x))

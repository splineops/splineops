# splineops/src/splineops/spline_interpolation/bases/utils.py

from typing import Union, Mapping, Type

from .spline_basis import SplineBasis
from .bspline0_basis import BSpline0Basis, BSpline0SymBasis
from .bspline1_basis import BSpline1Basis
from .bspline2_basis import BSpline2Basis
from .bspline3_basis import BSpline3Basis
from .bspline4_basis import BSpline4Basis
from .bspline5_basis import BSpline5Basis
from .bspline6_basis import BSpline6Basis
from .bspline7_basis import BSpline7Basis
from .bspline8_basis import BSpline8Basis
from .bspline9_basis import BSpline9Basis
from .omoms0_basis import OMOMS0Basis, OMOMS0SymBasis
from .omoms1_basis import OMOMS1Basis
from .omoms2_basis import OMOMS2Basis, OMOMS2SymBasis
from .omoms3_basis import OMOMS3Basis
from .omoms4_basis import OMOMS4Basis, OMOMS4SymBasis
from .omoms5_basis import OMOMS5Basis
from .nearest_neighbor_basis import NearestNeighborBasis, NearestNeighborSymBasis
from .linear_basis import LinearBasis
from .keys_basis import KeysBasis

basis_map: Mapping[str, Type[SplineBasis]] = {
    "bspline0": BSpline0Basis,
    "bspline0-sym": BSpline0SymBasis,
    "bspline1": BSpline1Basis,
    "bspline2": BSpline2Basis,
    "bspline3": BSpline3Basis,
    "bspline4": BSpline4Basis,
    "bspline5": BSpline5Basis,
    "bspline6": BSpline6Basis,
    "bspline7": BSpline7Basis,
    "bspline8": BSpline8Basis,
    "bspline9": BSpline9Basis,
    "omoms0": OMOMS0Basis,
    "omoms0-sym": OMOMS0SymBasis,
    "omoms1": OMOMS1Basis,
    "omoms2": OMOMS2Basis,
    "omoms2-sym": OMOMS2SymBasis,
    "omoms3": OMOMS3Basis,
    "omoms4": OMOMS4Basis,
    "omoms4-sym": OMOMS4SymBasis,
    "omoms5": OMOMS5Basis,
    "nearest": NearestNeighborBasis,
    "nearest-sym": NearestNeighborSymBasis,
    "linear": LinearBasis,
    "keys": KeysBasis,
}


def create_basis(name: str) -> SplineBasis:
    # Check if valid basis name
    valid_names = basis_map.keys()
    if name not in valid_names:
        valid_name_str = ", ".join([f"'{b}'" for b in valid_names])
        raise ValueError(
            f"Unsupported basis '{name}'. " f"Supported: {valid_name_str}."
        )

    basis = basis_map[name]()  # type: ignore
    # TODO(dperdios): not easy to get this mapping to work with mypy.
    #  The type annotation `basis_map: Mapping[str, Type[SplineBasis]]` helps
    #  (as in https://stackoverflow.com/a/54243383) but the combination with
    #  mandatory inputs (e.g., `support`) in the constructor may not be
    #  straightforward.

    return basis


def asbasis(basis: Union[str, SplineBasis]) -> SplineBasis:
    if isinstance(basis, str):
        return create_basis(name=basis)
    elif isinstance(basis, SplineBasis):
        return basis
    else:
        raise TypeError(f"Must be a 'str' or a '{SplineBasis.__name__}.'")

from typing import Union, Mapping, Type
from splineops.bases.splinebasis import SplineBasis
from splineops.bases.bspline0basis import BSpline0Basis, BSpline0SymBasis
from splineops.bases.bspline1basis import BSpline1Basis
from splineops.bases.bspline2basis import BSpline2Basis
from splineops.bases.bspline3basis import BSpline3Basis
from splineops.bases.bspline4basis import BSpline4Basis
from splineops.bases.bspline5basis import BSpline5Basis
from splineops.bases.bspline6basis import BSpline6Basis
from splineops.bases.bspline7basis import BSpline7Basis
from splineops.bases.bspline8basis import BSpline8Basis
from splineops.bases.bspline9basis import BSpline9Basis
from splineops.bases.omoms0basis import OMOMS0Basis, OMOMS0SymBasis
from splineops.bases.omoms1basis import OMOMS1Basis
from splineops.bases.omoms2basis import OMOMS2Basis, OMOMS2SymBasis
from splineops.bases.omoms3basis import OMOMS3Basis
from splineops.bases.omoms4basis import OMOMS4Basis, OMOMS4SymBasis
from splineops.bases.omoms5basis import OMOMS5Basis
from splineops.bases.nearestneighborbasis import NearestNeighborBasis
from splineops.bases.nearestneighborbasis import NearestNeighborSymBasis
from splineops.bases.linearbasis import LinearBasis
from splineops.bases.keysbasis import KeysBasis

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
    # TODO(dperdios): not easy go get this mapping to work with mypy.
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

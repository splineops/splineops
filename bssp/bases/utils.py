from typing import Union
from bssp.bases.splinebasis import SplineBasis
from bssp.bases.bspline0basis import BSpline0Basis
from bssp.bases.bspline1basis import BSpline1Basis
from bssp.bases.bspline2basis import BSpline2Basis
from bssp.bases.bspline3basis import BSpline3Basis
from bssp.bases.bspline4basis import BSpline4Basis
from bssp.bases.bspline5basis import BSpline5Basis
from bssp.bases.bspline6basis import BSpline6Basis
from bssp.bases.bspline7basis import BSpline7Basis
from bssp.bases.bspline8basis import BSpline8Basis
from bssp.bases.bspline9basis import BSpline9Basis
from bssp.bases.omoms0basis import OMOMS0Basis
from bssp.bases.omoms1basis import OMOMS1Basis
from bssp.bases.omoms2basis import OMOMS2Basis
from bssp.bases.omoms3basis import OMOMS3Basis
from bssp.bases.omoms4basis import OMOMS4Basis
from bssp.bases.omoms5basis import OMOMS5Basis

basis_map = {
    'bspline0': BSpline0Basis,
    'bspline1': BSpline1Basis,
    'bspline2': BSpline2Basis,
    'bspline3': BSpline3Basis,
    'bspline4': BSpline4Basis,
    'bspline5': BSpline5Basis,
    'bspline6': BSpline6Basis,
    'bspline7': BSpline7Basis,
    'bspline8': BSpline8Basis,
    'bspline9': BSpline9Basis,
    'omoms0': OMOMS0Basis,
    'omoms1': OMOMS1Basis,
    'omoms2': OMOMS2Basis,
    'omoms3': OMOMS3Basis,
    'omoms4': OMOMS4Basis,
    'omoms5': OMOMS5Basis,
}


def create_basis(name: str) -> SplineBasis:

    # Check if valid basis name
    valid_names = basis_map.keys()
    if name not in valid_names:
        valid_name_str = ', '.join([f"'{b}'" for b in valid_names])
        raise ValueError(
            f"Unsupported basis '{name}'. "
            f"Supported: {valid_name_str}.")

    return basis_map[name]()


def asbasis(basis: Union[str, SplineBasis]) -> SplineBasis:

    if isinstance(basis, str):
        return create_basis(name=basis)
    elif isinstance(basis, SplineBasis):
        return basis
    else:
        raise TypeError(f"Must be a 'str' or a '{SplineBasis.__name__}.'")

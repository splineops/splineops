from typing import Union, Mapping, Type
from splineops.modes.extensionmode import ExtensionMode
from splineops.modes.narrowmirroring import NarrowMirroring
from splineops.modes.finitesupportcoefficients import FiniteSupportCoefficients

mode_map: Mapping[str, Type[ExtensionMode]] = {
    "zero": FiniteSupportCoefficients,
    "mirror": NarrowMirroring,
}


def create_mode(name: str) -> ExtensionMode:
    # Check if valid mode name
    valid_names = mode_map.keys()
    if name not in valid_names:
        valid_name_str = ", ".join([f"'{b}'" for b in valid_names])
        raise ValueError(f"Unsupported mode '{name}'. " f"Supported: {valid_name_str}.")

    mode = mode_map[name]()  # type: ignore
    # TODO(dperdios): not easy go get this mapping to work with mypy.
    #  The type annotation `basis_map: Mapping[str, Type[SplineBasis]]` helps
    #  (as in https://stackoverflow.com/a/54243383) but the combination with
    #  mandatory inputs (e.g., `support`) in the constructor may not be
    #  straightforward.

    return mode


def asmode(mode: Union[str, ExtensionMode]) -> ExtensionMode:
    if isinstance(mode, str):
        return create_mode(name=mode)
    elif isinstance(mode, ExtensionMode):
        return mode
    else:
        raise TypeError(f"Must be a 'str' or a '{ExtensionMode.__name__}.'")

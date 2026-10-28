"""Validate release contents and, optionally, the complete supported wheel matrix."""

from __future__ import annotations

import argparse
from email.parser import Parser
from itertools import product
from pathlib import Path
import tarfile
import zipfile

from packaging.utils import parse_wheel_filename
from packaging.version import Version

PYTHONS = ("cp311", "cp312", "cp313")
PLATFORMS = ("linux-x86_64", "windows-amd64", "macos-x86_64", "macos-arm64")
RESOURCE = "splineops/demos/data/selma3d_vessels_summary.json"


def wheel_target(tag):
    platform = tag.platform
    if platform.startswith("manylinux") and platform.endswith("_x86_64"):
        platform = "linux-x86_64"
    elif platform == "win_amd64":
        platform = "windows-amd64"
    elif platform.startswith("macosx_") and platform.endswith("_x86_64"):
        platform = "macos-x86_64"
    elif platform.startswith("macosx_") and platform.endswith("_arm64"):
        platform = "macos-arm64"
    else:
        raise ValueError(f"Unexpected wheel platform: {tag.platform}")
    if tag.interpreter not in PYTHONS or tag.abi != tag.interpreter:
        raise ValueError(f"Unexpected Python/ABI tag: {tag}")
    return tag.interpreter, platform


def audit(directory: Path, expected_version: str, *, complete: bool = False):
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    if not wheels or len(sdists) != 1:
        raise ValueError("Expected at least one wheel and exactly one source archive")
    version = Version(expected_version)
    targets = set()
    for wheel in wheels:
        name, wheel_version, _, tags = parse_wheel_filename(wheel.name)
        if name != "splineops" or wheel_version != version:
            raise ValueError(f"Unexpected package/version: {wheel.name}")
        wheel_targets = {wheel_target(tag) for tag in tags}
        if len(wheel_targets) != 1 or targets.intersection(wheel_targets):
            raise ValueError(f"Ambiguous or duplicate wheel target: {wheel.name}")
        targets.update(wheel_targets)
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            native = [
                name
                for name in names
                if name.startswith("splineops/_lsresize.")
                and name.endswith((".so", ".pyd"))
            ]
            if len(native) != 1 or RESOURCE not in names:
                raise ValueError(f"Missing native extension or resource: {wheel.name}")
            metadata_names = [
                name for name in names if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise ValueError(f"Expected one wheel metadata file: {wheel.name}")
            metadata = Parser().parsestr(
                archive.read(metadata_names[0]).decode("utf-8")
            )
            if (
                metadata["Name"] != "splineops"
                or Version(metadata["Version"]) != version
            ):
                raise ValueError(
                    f"Wheel metadata disagrees with filename: {wheel.name}"
                )
    if complete and targets != set(product(PYTHONS, PLATFORMS)):
        raise ValueError(f"Incomplete release wheel matrix: {sorted(targets)}")
    expected_name = f"splineops-{version}.tar.gz"
    if sdists[0].name != expected_name:
        raise ValueError(f"Expected source archive {expected_name}")
    with tarfile.open(sdists[0]) as archive:
        names = archive.getnames()
    if any("__pycache__" in name or name.endswith(".pyc") for name in names):
        raise ValueError("Source archive contains Python cache files")
    for required in (
        f"src/{RESOURCE}",
        "scripts/check_installed_wheel.py",
        "CMakeLists.txt",
    ):
        if not any(name.endswith(f"/{required}") for name in names):
            raise ValueError(f"Source archive is missing {required}")
    return len(wheels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("dist"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    count = audit(args.directory, args.version, complete=args.complete)
    print(
        f"Validated {count} native wheel(s) and the source archive for {args.version}."
    )


if __name__ == "__main__":
    main()

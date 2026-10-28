"""Release gates must reject missing, duplicated, or mismatched artifacts."""

import importlib.util
import io
from itertools import product
from pathlib import Path
import tarfile
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_release_artifacts.py"
SPEC = importlib.util.spec_from_file_location("release_audit", SCRIPT)
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)

PLATFORMS = (
    "manylinux_2_17_x86_64.manylinux2014_x86_64",
    "win_amd64",
    "macosx_10_13_x86_64",
    "macosx_11_0_arm64",
)
VERSION = "2.3.0rc1"


def make_wheel(
    directory,
    python="cp312",
    platform="win_amd64",
    *,
    metadata_version=VERSION,
    native=True,
    resource=True,
):
    path = directory / f"splineops-{VERSION}-{python}-{python}-{platform}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"splineops-{VERSION}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: splineops\nVersion: {metadata_version}\n",
        )
        if native:
            archive.writestr(
                "splineops/_lsresize.test.pyd", b"native stub for archive audit"
            )
        if resource:
            archive.writestr(audit_module.RESOURCE, "{}")
    return path


def make_sdist(directory, *, extra=()):
    with tarfile.open(directory / f"splineops-{VERSION}.tar.gz", "w:gz") as archive:
        for name in (
            f"src/{audit_module.RESOURCE}",
            "scripts/check_installed_wheel.py",
            "CMakeLists.txt",
            *extra,
        ):
            info = tarfile.TarInfo(f"splineops-{VERSION}/{name}")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))


def test_complete_matrix_accepts_all_supported_targets(tmp_path):
    for python, platform in product(audit_module.PYTHONS, PLATFORMS):
        make_wheel(tmp_path, python, platform)
    make_sdist(tmp_path)
    assert audit_module.audit(tmp_path, VERSION, complete=True) == 12


def test_local_build_is_valid_but_incomplete_for_publication(tmp_path):
    make_wheel(tmp_path)
    make_sdist(tmp_path)
    assert audit_module.audit(tmp_path, VERSION) == 1
    with pytest.raises(ValueError, match="Incomplete"):
        audit_module.audit(tmp_path, VERSION, complete=True)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"metadata_version": "2.2.0"}, "metadata"),
        ({"native": False}, "native"),
        ({"resource": False}, "resource"),
    ],
)
def test_corrupt_wheel_cannot_pass_release_audit(tmp_path, options, match):
    make_wheel(tmp_path, **options)
    make_sdist(tmp_path)
    with pytest.raises(ValueError, match=match):
        audit_module.audit(tmp_path, VERSION)


def test_duplicate_platform_cannot_substitute_for_a_missing_wheel(tmp_path):
    make_wheel(tmp_path, platform="macosx_10_13_x86_64")
    make_wheel(tmp_path, platform="macosx_11_0_x86_64")
    make_sdist(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        audit_module.audit(tmp_path, VERSION)


def test_source_cache_files_are_rejected(tmp_path):
    make_wheel(tmp_path)
    make_sdist(tmp_path, extra=("src/splineops/__pycache__/cached.pyc",))
    with pytest.raises(ValueError, match="cache"):
        audit_module.audit(tmp_path, VERSION)


def test_wrong_release_version_is_rejected(tmp_path):
    make_wheel(tmp_path)
    make_sdist(tmp_path)
    with pytest.raises(ValueError, match="package/version"):
        audit_module.audit(tmp_path, "2.3.0")

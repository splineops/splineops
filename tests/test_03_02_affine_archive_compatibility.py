"""Permanent compatibility checks for historical affine coefficient archives."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import numpy as np
import pytest

from splineops.affine import AffinePlan

FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "affine_coefficients"
ARCHIVE_SHA256 = {
    1: "79e3da927b4928fa5555e13258e4c63e672b258f1ca79151dbab02a76be7dd92",
    2: "95b4e5b1a2456fc48bb6521084d35978dd2a054a6533921780851d6807b99728",
}


@pytest.mark.parametrize("schema_version", (1, 2))
def test_historical_affine_archive_remains_readable(schema_version, tmp_path):
    encoded = (
        (FIXTURE_DIRECTORY / f"schema-{schema_version}-float64.npz.b64")
        .read_bytes()
        .strip()
    )
    decoded = base64.b64decode(encoded, validate=True)
    assert hashlib.sha256(decoded).hexdigest() == ARCHIVE_SHA256[schema_version]
    archive = tmp_path / f"schema-{schema_version}.npz"
    archive.write_bytes(decoded)
    plan = AffinePlan((2, 3), np.eye(2), degree=1, mode="mirror")

    field = plan.load_coefficients(archive)

    np.testing.assert_equal(field.values, np.arange(6, dtype=np.float64).reshape(2, 3))
    np.testing.assert_equal(plan.apply_coefficients(field), field.values)


def test_historical_schema_one_can_be_migrated_without_value_changes(tmp_path):
    encoded = (FIXTURE_DIRECTORY / "schema-1-float64.npz.b64").read_bytes().strip()
    legacy_archive = tmp_path / "legacy.npz"
    legacy_archive.write_bytes(base64.b64decode(encoded, validate=True))
    plan = AffinePlan((2, 3), np.eye(2), degree=1, mode="mirror")
    legacy = plan.load_coefficients(legacy_archive)
    migrated_archive = tmp_path / "migrated.npz"

    legacy.save(migrated_archive)
    migrated = plan.load_coefficients(migrated_archive)

    assert migrated.configuration == legacy.configuration
    np.testing.assert_equal(migrated.values, legacy.values)

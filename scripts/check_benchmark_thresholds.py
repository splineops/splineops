"""Check machine-relative benchmark evidence against a stored policy.

The policy intentionally favors within-run ratios and numerical errors over
absolute seconds.  This makes it useful across developer and CI machines while
still detecting large execution regressions or broken equivalence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _resolve_path(value, path):
    values = [value]
    for token in path.split("."):
        next_values = []
        for current in values:
            if token == "*":
                if not isinstance(current, list):
                    raise ValueError("'*' path entries require a JSON list.")
                next_values.extend(current)
            elif isinstance(current, dict) and token in current:
                next_values.append(current[token])
            else:
                raise ValueError(f"JSON path component {token!r} was not found.")
        values = next_values
    return values


def check_policy(policy_path: Path, artifacts_dir: Path) -> list[str]:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark threshold schema.")
    failures = []
    for check in policy.get("checks", []):
        artifact_path = artifacts_dir / check["artifact"]
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        values = _resolve_path(payload, check["json_path"])
        if not values:
            failures.append(f"{check['name']}: JSON path selected no values")
            continue
        for index, raw_value in enumerate(values):
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                failures.append(
                    f"{check['name']}[{index}]: expected numeric value, "
                    f"received {raw_value!r}"
                )
                continue
            value = float(raw_value)
            if not math.isfinite(value):
                failures.append(f"{check['name']}[{index}]: value is not finite")
            if "minimum" in check and value < float(check["minimum"]):
                failures.append(
                    f"{check['name']}[{index}]: {value:.6g} is below "
                    f"{float(check['minimum']):.6g}"
                )
            if "maximum" in check and value > float(check["maximum"]):
                failures.append(
                    f"{check['name']}[{index}]: {value:.6g} exceeds "
                    f"{float(check['maximum']):.6g}"
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    failures = check_policy(args.policy, args.artifacts_dir)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("All benchmark thresholds passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

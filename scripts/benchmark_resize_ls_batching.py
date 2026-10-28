"""Compare automatic 3-D cubic LS dispatch with the scalar execution path.

For cubic equal-degree projection, disabling batching reproduces the automatic
3-D dispatch at parent a148a9e. Timings exclude plan construction and use
randomized method order. Numerical differences are parity checks, not quality
metrics. Run in a fresh process, with no competing performance measurements.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from splineops import __version__
from splineops.resize import ResizePlan

CASES = (
    ((64, 96, 128), (32, 48, 64)),
    ((48, 256, 256), (48, 32, 32)),
    ((48, 56, 64), (18, 34, 24)),
    ((24, 32, 40), (36, 48, 60)),
    ((8, 9, 10), (3, 4, 5)),
    ((96, 128, 128), (48, 64, 64)),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    from splineops import _lsresize  # Require the compiled extension.

    rng = np.random.default_rng(41218)
    rows = []
    saved = {
        key: os.environ.get(key)
        for key in ("LSRESIZE_NUM_THREADS", "LSRESIZE_BATCHED_AXIS")
    }
    try:
        for threads in ("1", "default"):
            if threads == "default":
                os.environ.pop("LSRESIZE_NUM_THREADS", None)
            else:
                os.environ["LSRESIZE_NUM_THREADS"] = threads
            for shape, target in CASES:
                for dtype in ("float32", "float64"):
                    values = rng.standard_normal(shape).astype(dtype)
                    plan = ResizePlan.from_degrees(
                        shape,
                        output_size=target,
                        interp_degree=3,
                        analy_degree=3,
                        synthe_degree=3,
                    )
                    if plan._native_plan is None:
                        raise RuntimeError("Native acceleration must be enabled")
                    outputs = {}
                    timings = {mode: [] for mode in ("off", "auto")}
                    for mode in timings:
                        os.environ["LSRESIZE_BATCHED_AXIS"] = mode
                        for _ in range(3):
                            outputs[mode] = plan(values)
                    for _ in range(args.repeats):
                        for mode in rng.permutation(list(timings)):
                            os.environ["LSRESIZE_BATCHED_AXIS"] = mode
                            started = time.perf_counter()
                            outputs[mode] = plan(values)
                            timings[mode].append(time.perf_counter() - started)
                    tolerance = 2e-6 if dtype == "float32" else 2e-12
                    np.testing.assert_allclose(
                        outputs["auto"],
                        outputs["off"],
                        atol=tolerance,
                        rtol=tolerance,
                    )
                    medians = {
                        mode: float(np.median(samples) * 1000)
                        for mode, samples in timings.items()
                    }
                    row = {
                        "shape": shape,
                        "target": target,
                        "dtype": dtype,
                        "threads": threads,
                        "median_ms": medians,
                        "samples_ms": {
                            mode: [value * 1000 for value in samples]
                            for mode, samples in timings.items()
                        },
                        "speedup": medians["off"] / medians["auto"],
                        "max_abs_difference": float(
                            np.max(np.abs(outputs["off"] - outputs["auto"]))
                        ),
                    }
                    rows.append(row)
                    print(
                        f"{shape} -> {target} {dtype} threads={threads}: "
                        f"{row['speedup']:.2f}x, "
                        f"max difference={row['max_abs_difference']:.3g}",
                        flush=True,
                    )
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    payload = {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "splineops": __version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "native_extension": _lsresize.__file__,
        },
        "method": "cubic equal-degree LS (3,3,3)",
        "timing": "warm plans; 3 warmups; randomized order; no allocation tracing",
        "repeats": args.repeats,
        "results": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

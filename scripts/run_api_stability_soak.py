"""Exercise persisted and buffered public workflows over repeated API cycles.

This is a correctness and resource soak, not a timing benchmark.  It verifies
the same numerical contracts after archive replacement, thread fan-out, fresh
spawned-process loads, caller-buffer reuse, and rejected corrupt copies.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import cProfile
import hashlib
import json
import multiprocessing
import platform
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

from splineops import __version__
from splineops.affine import AffinePlan
from splineops.differentials import DifferentialPlan, DifferentialResult

PROFILES = {
    "smoke": {
        "cycles": 2,
        "frames": 2,
        "image_shape": (24, 32),
        "volume_shape": (6, 8, 10),
        "threads": 2,
        "processes": 2,
    },
    "standard": {
        "cycles": 12,
        "frames": 3,
        "image_shape": (64, 80),
        "volume_shape": (12, 16, 20),
        "threads": 3,
        "processes": 2,
    },
    "extended": {
        "cycles": 100,
        "frames": 4,
        "image_shape": (128, 160),
        "volume_shape": (24, 32, 40),
        "threads": 4,
        "processes": 4,
    },
}
ANGLES = (7.0, -11.0, 18.0)


def _rotation_2d(shape, angle):
    radians = np.radians(-angle)
    matrix = np.array(
        [
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ]
    )
    center = (np.asarray(shape) - 1.0) / 2.0
    return matrix, center - matrix @ center


def _rotation_3d(shape, angle):
    radians = np.radians(-angle)
    matrix = np.array(
        [
            [np.cos(radians), -np.sin(radians), 0.0],
            [np.sin(radians), np.cos(radians), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    center = (np.asarray(shape) - 1.0) / 2.0
    return matrix, center - matrix @ center


def _process_archive_fanout(task):
    """Load one archive in a spawned worker and apply all geometry specs."""

    archive, shape, specifications = task
    results = []
    for matrix, offset in specifications:
        plan = AffinePlan(shape, matrix, offset, degree=3, mode="mirror")
        field = plan.load_coefficients(archive)
        results.append(plan.apply_coefficients(field))
    return tuple(results)


def _max_difference(first, second):
    if isinstance(first, tuple):
        return max(_max_difference(a, b) for a, b in zip(first, second))
    return float(np.max(np.abs(first - second)))


def _buffer_ids(result):
    return tuple(id(item) for item in result.gradient) + (id(result.laplacian),)


def _run_soak(configuration, seed, directory):
    rng = np.random.default_rng(seed)
    image_shape = tuple(configuration["image_shape"])
    frames = rng.standard_normal((configuration["frames"],) + image_shape)
    frame_copies = frames.copy()
    specifications = tuple(_rotation_2d(image_shape, angle) for angle in ANGLES)
    affine_plans = tuple(
        AffinePlan(image_shape, matrix, offset, degree=3, mode="mirror")
        for matrix, offset in specifications
    )
    reference_registration = tuple(
        tuple(plan(frame) for plan in affine_plans) for frame in frames
    )

    volume_shape = tuple(configuration["volume_shape"])
    volumes = rng.standard_normal((2,) + volume_shape).astype(np.float32)
    volume_copy = volumes.copy()
    matrix, offset = _rotation_3d(volume_shape, 6.0)
    volume_affine = AffinePlan(
        volume_shape, matrix, offset, degree=3, mode="mirror", dtype=np.float32
    )
    differentials = DifferentialPlan(volume_shape, spacing=(0.8, 0.8, 1.5))
    warped_buffer = np.empty_like(volumes)
    feature_buffer = DifferentialResult(
        tuple(np.empty_like(volumes) for _ in range(3)),
        None,
        np.empty_like(volumes),
    )
    feature_buffer_ids = _buffer_ids(feature_buffer)
    reference_features = []
    for volume in volumes:
        warped = volume_affine(volume)
        result = differentials(warped, gradient=True, hessian=False, laplacian=True)
        reference_features.append(result.gradient + (result.laplacian,))
    reference_features = tuple(
        np.stack([components[index] for components in reference_features])
        for index in range(4)
    )

    maximum_thread_difference = 0.0
    maximum_process_difference = 0.0
    maximum_buffer_difference = 0.0
    corruption_rejections = 0
    archive_digests = set()
    archive_sizes = set()
    started = time.perf_counter()
    tracemalloc.start()
    try:
        spawn_context = multiprocessing.get_context("spawn")
        for cycle in range(configuration["cycles"]):
            process_tasks = []
            with ThreadPoolExecutor(
                max_workers=configuration["threads"]
            ) as thread_executor:
                for frame_index, frame in enumerate(frames):
                    field = affine_plans[0].prepare_coefficients(frame)
                    archive = directory / f"frame-{frame_index}.npz"
                    field.save(archive)
                    archive_bytes = archive.read_bytes()
                    archive_digests.add(hashlib.sha256(archive_bytes).hexdigest())
                    archive_sizes.add(len(archive_bytes))
                    restored = affine_plans[-1].load_coefficients(archive)
                    threaded = tuple(
                        thread_executor.map(
                            lambda plan: plan.apply_coefficients(restored),
                            affine_plans,
                        )
                    )
                    expected = reference_registration[frame_index]
                    maximum_thread_difference = max(
                        maximum_thread_difference,
                        _max_difference(threaded, expected),
                    )
                    process_tasks.append((str(archive), image_shape, specifications))

            # Each task gets a newly spawned interpreter.  This deliberately
            # checks on-disk restoration rather than inherited process state.
            with ProcessPoolExecutor(
                max_workers=configuration["processes"],
                mp_context=spawn_context,
                max_tasks_per_child=1,
            ) as process_executor:
                processed = tuple(
                    process_executor.map(_process_archive_fanout, process_tasks)
                )
            for frame_index, outputs in enumerate(processed):
                maximum_process_difference = max(
                    maximum_process_difference,
                    _max_difference(outputs, reference_registration[frame_index]),
                )

            corrupt = directory / "corrupt-copy.npz"
            good_bytes = (directory / "frame-0.npz").read_bytes()
            corrupt.write_bytes(good_bytes[: max(1, len(good_bytes) // 2)])
            try:
                affine_plans[0].load_coefficients(corrupt)
            except ValueError:
                corruption_rejections += 1
            else:
                raise AssertionError("A truncated affine archive was accepted.")

            returned = volume_affine(volumes, spatial_axes=(1, 2, 3), out=warped_buffer)
            if returned is not warped_buffer:
                raise AssertionError("AffinePlan did not return the caller buffer.")
            result = differentials(
                warped_buffer,
                gradient=True,
                hessian=False,
                laplacian=True,
                spatial_axes=(1, 2, 3),
                out=feature_buffer,
            )
            if (
                result is not feature_buffer
                or _buffer_ids(result) != feature_buffer_ids
            ):
                raise AssertionError("DifferentialPlan replaced a caller buffer.")
            maximum_buffer_difference = max(
                maximum_buffer_difference,
                _max_difference(
                    result.gradient + (result.laplacian,), reference_features
                ),
            )
            np.testing.assert_array_equal(frames, frame_copies)
            np.testing.assert_array_equal(volumes, volume_copy)
            if corruption_rejections != cycle + 1:
                raise AssertionError("Corrupt-archive rejection count changed.")
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    elapsed_seconds = time.perf_counter() - started
    return {
        "cycles_completed": configuration["cycles"],
        "archive_replacements": configuration["cycles"] * configuration["frames"],
        "thread_applications": (
            configuration["cycles"] * configuration["frames"] * len(affine_plans)
        ),
        "spawned_process_tasks": configuration["cycles"] * configuration["frames"],
        "corrupt_archives_rejected": corruption_rejections,
        "buffer_reuses": configuration["cycles"],
        "elapsed_seconds": elapsed_seconds,
        "tracemalloc_peak_bytes": peak_bytes,
        "retained_plan_bytes": (
            sum(plan.retained_bytes for plan in affine_plans)
            + volume_affine.retained_bytes
            + differentials.retained_bytes
        ),
        "unique_archive_digests": len(archive_digests),
        "archive_size_min_bytes": min(archive_sizes),
        "archive_size_max_bytes": max(archive_sizes),
        "max_abs_difference": {
            "thread_fanout": maximum_thread_difference,
            "spawned_process_fanout": maximum_process_difference,
            "buffered_volume_features": maximum_buffer_difference,
        },
        "source_arrays_unchanged": True,
        "caller_buffer_identities_preserved": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--cycles", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--processes", type=int)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--profile-output",
        type=Path,
        help="write a cProfile record for parent-process orchestration",
    )
    args = parser.parse_args()

    configuration = dict(PROFILES[args.profile])
    for name in ("cycles", "threads", "processes"):
        value = getattr(args, name)
        if value is not None:
            configuration[name] = value
        if configuration[name] <= 0:
            parser.error(f"{name} must be positive")

    profiler = cProfile.Profile() if args.profile_output else None
    if profiler is not None:
        profiler.enable()
    try:
        with tempfile.TemporaryDirectory(prefix="splineops-api-soak-") as directory:
            results = _run_soak(configuration, args.seed, Path(directory))
    finally:
        if profiler is not None:
            profiler.disable()
            args.profile_output.parent.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(args.profile_output)
    maximum_difference = max(results["max_abs_difference"].values())
    if maximum_difference != 0.0:
        raise AssertionError(
            f"Stability-soak numerical drift was {maximum_difference:.3e}."
        )

    payload = {
        "schema_version": 1,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "splineops": __version__,
            "process_start_method": "spawn",
        },
        "profile": args.profile,
        "configuration": configuration,
        "semantics": {
            "timing": "complete correctness soak; not a performance benchmark",
            "persistence": "atomic schema-2 replacement and checked restoration",
            "concurrency": "shared thread fields and independent spawned-process loads",
            "buffers": "caller-owned affine and differential output reuse",
        },
        "results": results,
    }
    print(
        f"cycles={results['cycles_completed']} "
        f"archives={results['archive_replacements']} "
        f"process-tasks={results['spawned_process_tasks']} "
        f"elapsed={results['elapsed_seconds']:.3f}s "
        f"max-error={maximum_difference:.3e}"
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import textwrap
import time

import numpy as np
import pytest

from splineops.resize import ResizePlan, resize, resize_degrees

try:
    import splineops._lsresize as _native_resize
except ImportError:  # pragma: no cover - exercised by fallback-only builds
    _native_resize = None


requires_native = pytest.mark.skipif(
    _native_resize is None,
    reason="native resize extension is not available",
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
@pytest.mark.parametrize("method", ["linear", "cubic-antialiasing"])
@requires_native
def test_direct_output_matches_allocating_path_bitwise(dtype, method):
    rng = np.random.default_rng(100)
    data = (rng.standard_normal((47, 39)) * 8).astype(dtype)
    output_size = (31, 23)

    expected = resize(data, output_size=output_size, method=method)
    expected_dtype = np.dtype(np.float32 if dtype == np.float32 else np.float64)
    assert expected.dtype == expected_dtype

    output = np.empty(output_size, dtype=expected_dtype)
    result = resize(
        data,
        output_size=output_size,
        method=method,
        output=output,
    )

    assert result is output
    assert np.array_equal(result, expected)


@requires_native
def test_direct_output_identity_preserves_every_bit():
    bits = np.arange(64, dtype=np.uint32).reshape(8, 8) * np.uint32(2654435761)
    data = bits.view(np.float32)
    original = data.copy()
    output = np.empty_like(data)

    result = resize(data, output_size=data.shape, method="cubic", output=output)

    assert result is output
    assert np.array_equal(output.view(np.uint32), bits)
    assert np.array_equal(data.view(np.uint32), original.view(np.uint32))


@requires_native
def test_output_cast_and_noncontiguous_fallbacks_remain_supported():
    data = np.arange(35, dtype=np.int16).reshape(7, 5)
    expected = resize(data, output_size=(9, 8), method="cubic")

    integer_output = np.empty(expected.shape, dtype=np.int16)
    assert (
        resize(
            data,
            output_size=expected.shape,
            method="cubic",
            output=integer_output,
        )
        is integer_output
    )
    assert np.array_equal(integer_output, expected.astype(np.int16))

    backing = np.empty(expected.shape[::-1], dtype=np.float64)
    noncontiguous_output = backing.T
    assert not noncontiguous_output.flags.c_contiguous
    assert (
        resize(
            data,
            output_size=expected.shape,
            method="cubic",
            output=noncontiguous_output,
        )
        is noncontiguous_output
    )
    assert np.array_equal(noncontiguous_output, expected)


@requires_native
def test_public_alias_uses_temporary_and_native_direct_rejects_overlap():
    data = np.arange(64, dtype=np.float64).reshape(8, 8)
    plan = ResizePlan(data.shape, output_size=data.shape, method="cubic")
    expected = data.copy()

    assert resize(data, output_size=data.shape, method="cubic", output=data) is data
    assert np.array_equal(data, expected)
    assert plan(data, output=data) is data
    assert np.array_equal(data, expected)
    with pytest.raises(ValueError, match="overlap"):
        _native_resize.resize_nd_into(data, data, [1.0, 1.0], 3, -1, 3)


@requires_native
@pytest.mark.parametrize("use_plan", [False, True])
def test_native_direct_rejects_output_that_would_require_a_temporary(use_plan):
    data = np.arange(35, dtype=np.float64).reshape(7, 5)
    output_shape = (9, 8)
    zoom = [output_shape[0] / data.shape[0], output_shape[1] / data.shape[1]]
    native_plan = _native_resize.ResizePlan(list(data.shape), zoom, 3, -1, 3)

    def apply(output):
        if use_plan:
            return native_plan.apply_into(data, output)
        return _native_resize.resize_nd_into(data, output, zoom, 3, -1, 3)

    wrong_dtype = np.full(output_shape, -99, dtype=np.float32)
    with pytest.raises(TypeError, match="output dtype"):
        apply(wrong_dtype)
    assert np.all(wrong_dtype == -99)

    fortran_output = np.full(output_shape, -99, dtype=np.float64, order="F")
    with pytest.raises(ValueError, match="C-contiguous"):
        apply(fortran_output)
    assert np.all(fortran_output == -99)

    storage = np.empty(np.prod(output_shape) * 8 + 1, dtype=np.uint8)
    unaligned_output = np.ndarray(
        output_shape, dtype=np.float64, buffer=storage, offset=1
    )
    assert unaligned_output.flags.c_contiguous
    assert not unaligned_output.flags.aligned
    unaligned_output.fill(-99)
    with pytest.raises(ValueError, match="aligned"):
        apply(unaligned_output)
    assert np.all(unaligned_output == -99)


@requires_native
@pytest.mark.parametrize("use_plan", [False, True])
def test_native_input_is_copied_when_not_naturally_aligned(use_plan):
    shape = (7, 5)
    output_shape = (9, 8)
    zoom = [output_shape[0] / shape[0], output_shape[1] / shape[1]]
    storage = np.empty(np.prod(shape) * 8 + 1, dtype=np.uint8)
    data = np.ndarray(shape, dtype=np.float64, buffer=storage, offset=1)
    data[...] = np.arange(np.prod(shape)).reshape(shape)
    assert data.flags.c_contiguous
    assert not data.flags.aligned

    expected = _native_resize.resize_nd(np.array(data, copy=True), zoom, 3, -1, 3)
    if use_plan:
        native_plan = _native_resize.ResizePlan(list(shape), zoom, 3, -1, 3)
        result = native_plan.apply(data)
    else:
        result = _native_resize.resize_nd(data, zoom, 3, -1, 3)

    assert np.array_equal(result, expected)


@requires_native
def test_public_output_falls_back_safely_when_not_naturally_aligned():
    data = np.arange(35, dtype=np.float64).reshape(7, 5)
    output_shape = (9, 8)
    expected = resize(data, output_size=output_shape, method="cubic")
    storage = np.empty(np.prod(output_shape) * 8 + 1, dtype=np.uint8)
    output = np.ndarray(output_shape, dtype=np.float64, buffer=storage, offset=1)
    assert output.flags.c_contiguous
    assert not output.flags.aligned

    assert (
        resize(
            data,
            output_size=output_shape,
            method="cubic",
            output=output,
        )
        is output
    )
    assert np.array_equal(output, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", ["linear", "cubic", "cubic-antialiasing"])
@requires_native
def test_plan_allocating_and_direct_paths_are_bitwise_identical(dtype, method):
    data = np.random.default_rng(202).standard_normal((43, 37)).astype(dtype)
    plan = ResizePlan(data.shape, output_size=(29, 21), method=method)

    expected = resize(data, output_size=plan.output_shape, method=method)
    allocated = plan(data)
    output = np.empty(plan.output_shape, dtype=dtype)
    direct = plan(data, output=output)

    assert direct is output
    assert np.array_equal(allocated, expected)
    assert np.array_equal(direct, allocated)


@requires_native
def test_two_axis_plan_retains_only_one_intermediate_per_dtype(monkeypatch):
    monkeypatch.delenv("LSRESIZE_WORKSPACE_CACHE_BYTES", raising=False)
    shape = (128, 112)
    zoom = (91 / shape[0], 73 / shape[1])
    plan = _native_resize.ResizePlan(shape, zoom, 3, 1, 3)

    data64 = np.ones(shape, dtype=np.float64)
    for _ in range(3):
        plan.apply(data64)
    info = plan._workspace_cache_info
    assert info["limit_bytes"] == 128 * 1024 * 1024
    assert info["retained_count"] == info["float64_count"] == 1
    assert info["float64_prev_bytes"] > 0
    assert info["float64_scratch_bytes"] == 0
    assert info["retained_bytes"] == info["float64_prev_bytes"]

    plan.apply(data64.astype(np.float32))
    info = plan._workspace_cache_info
    assert info["retained_count"] == 2
    assert info["float32_count"] == info["float64_count"] == 1
    assert info["float32_prev_bytes"] > 0
    assert info["float32_scratch_bytes"] == 0
    assert info["float64_scratch_bytes"] == 0

    with pytest.raises(AttributeError):
        plan._workspace_cache_info = {}


@requires_native
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0", 0),
        ("4096", 4096),
        (" 8192 ", 8192),
        ("", 128 * 1024 * 1024),
        ("invalid", 128 * 1024 * 1024),
        ("-1", 128 * 1024 * 1024),
        ("+1", 128 * 1024 * 1024),
        ("184467440737095516160", 128 * 1024 * 1024),
    ],
)
def test_workspace_cache_byte_limit_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("LSRESIZE_WORKSPACE_CACHE_BYTES", raw)
    plan = _native_resize.ResizePlan((8, 7), (0.75, 0.75), 3, 1, 3)

    assert plan._workspace_cache_info["limit_bytes"] == expected


@requires_native
def test_zero_workspace_cache_limit_retains_only_primary(monkeypatch):
    monkeypatch.setenv("LSRESIZE_WORKSPACE_CACHE_BYTES", "0")
    plan = _native_resize.ResizePlan((128, 112), (91 / 128, 73 / 112), 3, 1, 3)
    data = np.ones((128, 112), dtype=np.float64)

    plan.apply(data)
    plan.apply(data.astype(np.float32))

    info = plan._workspace_cache_info
    assert info["limit_bytes"] == 0
    assert info["retained_count"] == 1
    assert info["retained_bytes"] > info["limit_bytes"]
    assert info["float32_count"] + info["float64_count"] == 1


@requires_native
def test_mixed_dtype_concurrency_obeys_shared_workspace_count(monkeypatch):
    monkeypatch.setenv("LSRESIZE_NUM_THREADS", "1")
    monkeypatch.setenv("LSRESIZE_WORKSPACE_CACHE_BYTES", str(64 * 1024 * 1024))
    shape = (512, 448)
    zoom = (379 / shape[0], 331 / shape[1])
    plan = _native_resize.ResizePlan(shape, zoom, 3, 1, 3)
    inputs = [
        np.ones(shape, dtype=np.float32 if index % 2 else np.float64)
        for index in range(8)
    ]
    barrier = threading.Barrier(len(inputs))

    def apply(data):
        barrier.wait(timeout=10)
        return float(np.sum(plan.apply(data)))

    with ThreadPoolExecutor(max_workers=len(inputs)) as executor:
        results = list(executor.map(apply, inputs))

    assert all(np.isfinite(value) for value in results)
    info = plan._workspace_cache_info
    assert info["retained_count"] == (info["float32_count"] + info["float64_count"])
    # Thread scheduling controls how many leases overlap and which dtype returns
    # first, so the cache need not fill every slot or retain a particular dtype
    # mix. The contract is one useful primary allocation plus a shared cap.
    assert 1 <= info["retained_count"] <= info["max_retained_count"] == 4
    assert info["retained_bytes"] <= info["limit_bytes"] == 64 * 1024 * 1024
    assert info["retained_bytes"] == (info["float32_bytes"] + info["float64_bytes"])
    assert info["float32_scratch_bytes"] == 0
    assert info["float64_scratch_bytes"] == 0


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="RSS regression reads Linux /proc status",
)
@requires_native
def test_workspace_burst_retained_rss_is_bounded_in_subprocess():
    script = textwrap.dedent("""
        import ctypes
        import gc
        import json
        import os
        import threading
        from concurrent.futures import ThreadPoolExecutor

        import numpy as np
        import splineops._lsresize as native

        def rss_bytes():
            with open('/proc/self/status', encoding='ascii') as stream:
                for line in stream:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) * 1024
            raise RuntimeError('VmRSS is unavailable')

        os.environ['LSRESIZE_NUM_THREADS'] = '1'
        os.environ['LSRESIZE_WORKSPACE_CACHE_BYTES'] = str(16 * 1024 * 1024)
        shape = (1280, 1024)
        zoom = (960 / shape[0], 768 / shape[1])
        plan = native.ResizePlan(shape, zoom, 3, 1, 3)
        data64 = np.ones(shape, dtype=np.float64)
        data32 = np.ones(shape, dtype=np.float32)
        before = rss_bytes()

        def run_wave():
            barrier = threading.Barrier(8)
            def apply(index):
                barrier.wait(timeout=30)
                data = data32 if index % 2 else data64
                return float(np.sum(plan.apply(data)))
            with ThreadPoolExecutor(max_workers=8) as executor:
                values = list(executor.map(apply, range(8)))
            assert all(np.isfinite(value) for value in values)

        run_wave()
        run_wave()
        gc.collect()
        libc = ctypes.CDLL(None)
        if hasattr(libc, 'malloc_trim'):
            libc.malloc_trim(0)
        after = rss_bytes()
        print(json.dumps({
            'rss_before': before,
            'rss_after': after,
            'info': plan._workspace_cache_info,
        }))
        """)
    environment = os.environ.copy()
    environment["MALLOC_ARENA_MAX"] = "2"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    info = result["info"]
    rss_growth = max(0, result["rss_after"] - result["rss_before"])

    assert info["retained_count"] <= info["max_retained_count"] == 4
    assert info["retained_bytes"] <= info["limit_bytes"] == 16 * 1024 * 1024
    # Account for allocator/page granularity and extension bookkeeping while
    # tying the dominant retained growth to the explicit workspace budget.
    assert rss_growth <= info["retained_bytes"] + 32 * 1024 * 1024


@pytest.mark.parametrize(
    ("attribute", "replacement"),
    [
        ("input_shape", (99, 99)),
        ("output_shape", (99, 99)),
        ("zoom_factors", (1.0, 1.0)),
        ("axes", (1,)),
        ("interp_degree", 0),
        ("analy_degree", 0),
        ("synthe_degree", 0),
        ("method", "fast"),
    ],
)
def test_plan_configuration_is_read_only(attribute, replacement):
    plan = ResizePlan((17, 13), output_size=(11, 9), method="cubic")
    original = getattr(plan, attribute)

    with pytest.raises(AttributeError):
        setattr(plan, attribute, replacement)

    assert getattr(plan, attribute) == original
    assert plan(np.arange(17 * 13).reshape(17, 13)).shape == (11, 9)


@requires_native
@pytest.mark.parametrize("target_length", [9, 6])
def test_native_unselected_axes_are_never_filtered_by_custom_degrees(
    target_length,
):
    # target_length=6 also proves that a selected same-grid custom-synthesis
    # pass remains active; selection and geometric identity are independent.
    data = np.random.default_rng(52).standard_normal((6, 7))
    kwargs = {
        "output_size": (target_length,),
        "axes": (0,),
        "interp_degree": 3,
        "analy_degree": 1,
        "synthe_degree": 1,
    }
    expected = np.stack(
        [
            resize_degrees(
                data[:, column],
                output_size=(target_length,),
                interp_degree=3,
                analy_degree=1,
                synthe_degree=1,
            )
            for column in range(data.shape[1])
        ],
        axis=1,
    )

    actual = resize_degrees(data, **kwargs)
    planned = ResizePlan.from_degrees(data.shape, **kwargs)(data)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(planned, expected, rtol=0.0, atol=2e-12)


@requires_native
def test_same_plan_is_safe_under_concurrent_repeated_calls(monkeypatch):
    monkeypatch.setenv("LSRESIZE_NUM_THREADS", "4")
    plan = ResizePlan(
        (128, 112),
        output_size=(91, 73),
        method="cubic-antialiasing",
    )
    inputs = [
        np.random.default_rng(seed).standard_normal(plan.input_shape).astype(np.float32)
        for seed in range(8)
    ]
    expected = [plan(data) for data in inputs]

    def apply(index: int) -> np.ndarray:
        data = inputs[index % len(inputs)]
        if index % 2:
            output = np.empty(plan.output_shape, dtype=np.float32)
            assert plan(data, output=output) is output
            return output
        return plan(data)

    # Repeated rounds exercise workspace return/re-lease as well as many
    # persistent-executor batch completions from independent Python callers.
    for _ in range(3):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(apply, range(16)))
        for index, result in enumerate(results):
            assert np.array_equal(result, expected[index % len(inputs)])


@requires_native
def test_independent_one_shot_calls_complete_concurrently(monkeypatch):
    monkeypatch.setenv("LSRESIZE_NUM_THREADS", "4")
    inputs = [
        np.random.default_rng(seed + 80).standard_normal((96, 88)) for seed in range(6)
    ]
    expected = [
        resize(data, output_size=(67, 59), method="cubic-antialiasing")
        for data in inputs
    ]

    def apply(index: int) -> np.ndarray:
        output = np.empty((67, 59), dtype=np.float64)
        return resize(
            inputs[index],
            output_size=output.shape,
            method="cubic-antialiasing",
            output=output,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(apply, range(len(inputs))))
    for result, reference in zip(results, expected):
        assert np.array_equal(result, reference)


@requires_native
def test_native_compute_releases_the_gil(monkeypatch):
    monkeypatch.setenv("LSRESIZE_NUM_THREADS", "1")
    data = np.random.default_rng(303).standard_normal((768, 768))
    plan = ResizePlan(
        data.shape,
        output_size=(511, 503),
        method="cubic-antialiasing",
    )

    stop = threading.Event()
    counter = [0]

    def python_worker() -> None:
        while not stop.is_set():
            counter[0] += 1
            if counter[0] % 1000 == 0:
                time.sleep(0)

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1.0)
    worker = threading.Thread(target=python_worker)
    worker.start()
    try:
        time.sleep(0.02)
        before = counter[0]
        plan(data)
        progressed = counter[0] - before
    finally:
        stop.set()
        worker.join(timeout=5)
        sys.setswitchinterval(old_interval)

    assert progressed >= 1000


def _fork_resize_child(connection) -> None:
    try:
        os.environ["LSRESIZE_NUM_THREADS"] = "4"
        data = np.arange(128 * 96, dtype=np.float64).reshape(128, 96)
        result = resize(
            data,
            output_size=(91, 67),
            method="cubic-antialiasing",
        )
        connection.send((result.shape, float(np.sum(result))))
    except BaseException as exc:  # pragma: no cover - reported to parent
        connection.send(("error", repr(exc)))
    finally:
        connection.close()


def _fork_inherited_plan_child(connection, plan, data) -> None:
    try:
        result = plan(data)
        connection.send((result.shape, float(np.sum(result))))
    except BaseException as exc:  # pragma: no cover - reported to parent
        connection.send(("error", repr(exc)))
    finally:
        connection.close()


@pytest.mark.skipif(os.name != "posix", reason="fork is POSIX-only")
@requires_native
def test_resize_after_fork_reinitializes_persistent_executor(monkeypatch):
    monkeypatch.setenv("LSRESIZE_NUM_THREADS", "4")
    data = np.arange(128 * 96, dtype=np.float64).reshape(128, 96)
    # Warm the process-wide executor so the child inherits the difficult case.
    resize(data, output_size=(91, 67), method="cubic-antialiasing")

    context = multiprocessing.get_context("fork")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(target=_fork_resize_child, args=(child_connection,))
    process.start()
    child_connection.close()
    process.join(timeout=15)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail("native resize deadlocked after fork")

    assert process.exitcode == 0
    assert parent_connection.poll(1)
    message = parent_connection.recv()
    parent_connection.close()
    assert message[0] != "error", message
    assert message[0] == (91, 67)
    assert np.isfinite(message[1])


@pytest.mark.skipif(os.name != "posix", reason="fork is POSIX-only")
@requires_native
def test_inherited_plan_is_usable_when_fork_occurs_during_apply(monkeypatch):
    # Workspace-pool acquire/return happens with the GIL held, while the
    # workspace itself is outside the pool during native execution.  Repeatedly
    # apply one inherited plan in a parent thread while a child reuses the same
    # plan to exercise that fork boundary.
    monkeypatch.setenv("LSRESIZE_NUM_THREADS", "1")
    data = np.random.default_rng(404).standard_normal((320, 288))
    plan = ResizePlan(
        data.shape,
        output_size=(231, 207),
        method="cubic-antialiasing",
    )

    started = threading.Event()
    stop = threading.Event()
    parent_errors = []

    def parent_worker() -> None:
        started.set()
        try:
            while not stop.is_set():
                plan(data)
        except BaseException as exc:  # pragma: no cover - asserted below
            parent_errors.append(exc)

    worker = threading.Thread(target=parent_worker)
    worker.start()
    assert started.wait(timeout=5)

    context = multiprocessing.get_context("fork")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_fork_inherited_plan_child,
        args=(child_connection, plan, data),
    )
    try:
        process.start()
        child_connection.close()
        process.join(timeout=15)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("inherited resize plan deadlocked after fork")
    finally:
        stop.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert not parent_errors
    assert process.exitcode == 0
    assert parent_connection.poll(1)
    message = parent_connection.recv()
    parent_connection.close()
    assert message[0] != "error", message
    assert message[0] == plan.output_shape
    assert np.isfinite(message[1])


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("interp_degree", 2.5),
        ("interp_degree", True),
        ("analy_degree", np.float64(1.0)),
        ("analy_degree", np.bool_(False)),
        ("synthe_degree", 2.0),
        ("synthe_degree", False),
    ],
)
def test_degrees_require_non_boolean_integers(argument, value):
    kwargs = {
        "interp_degree": 3,
        "analy_degree": 1,
        "synthe_degree": 3,
    }
    kwargs[argument] = value
    with pytest.raises(TypeError, match=rf"{argument} must be an integer"):
        resize_degrees(
            np.arange(16, dtype=np.float64),
            output_size=(9,),
            **kwargs,
        )


def test_numpy_integer_degrees_are_accepted():
    result = resize_degrees(
        np.arange(16, dtype=np.float64),
        output_size=(9,),
        interp_degree=np.int64(3),
        analy_degree=np.int64(1),
        synthe_degree=np.int64(3),
    )
    assert result.shape == (9,)


def test_accel_always_fails_clearly_without_native(monkeypatch):
    module = importlib.import_module("splineops.resize.resize")
    import_error = ImportError("test extension failure")
    monkeypatch.setattr(module, "_ACCEL_ENV", "always")
    monkeypatch.setattr(module, "_HAS_CPP", False)
    monkeypatch.setattr(module, "_NATIVE_IMPORT_ERROR", import_error)

    with pytest.raises(RuntimeError, match="SPLINEOPS_ACCEL=always") as error:
        module.resize(np.arange(8), output_size=(5,))
    assert error.value.__cause__ is import_error

    with pytest.raises(RuntimeError, match="SPLINEOPS_ACCEL=always"):
        module.ResizePlan((8,), output_size=(5,))

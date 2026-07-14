"""Focused regressions for the byte-bounded pure-Python plan cache."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from splineops.resize._pycore import resize_nd as core
from splineops.resize._pycore.params import LSParams


_CACHE_ENV = (
    "SPLINEOPS_PLAN_CACHE",
    "SPLINEOPS_PLAN_CACHE_SIZE",
    "SPLINEOPS_PLAN_CACHE_BYTES",
    "LSRESIZE_PLAN_CACHE_SIZE",
    "LSRESIZE_PLAN_CACHE_BYTES",
)
_PARAMS = LSParams(3, 3, 3, 0.61, 0.0)


@pytest.fixture(autouse=True)
def _empty_plan_cache(monkeypatch):
    for name in _CACHE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SPLINEOPS_PLAN_CACHE_BYTES", "0")
    core._get_plan(31, _PARAMS)
    monkeypatch.setenv("SPLINEOPS_PLAN_CACHE_BYTES", str(128 * 1024 * 1024))
    yield
    monkeypatch.setenv("SPLINEOPS_PLAN_CACHE_BYTES", "0")
    core._get_plan(31, _PARAMS)


def test_byte_limit_skips_oversized_entries_and_zero_clears_cache(monkeypatch):
    first = core._get_plan(256, _PARAMS)
    assert core._get_plan(256, _PARAMS) is first

    monkeypatch.setenv("SPLINEOPS_PLAN_CACHE_BYTES", "1")
    uncached_first = core._get_plan(256, _PARAMS)
    uncached_second = core._get_plan(256, _PARAMS)

    assert uncached_first is not first
    assert uncached_second is not uncached_first


@pytest.mark.parametrize(
    "value",
    ["invalid", "-1", "12bytes", "9" * 5000],
)
def test_invalid_byte_limit_uses_default(monkeypatch, value):
    monkeypatch.setenv("SPLINEOPS_PLAN_CACHE_BYTES", value)

    first = core._get_plan(257, _PARAMS)

    assert core._get_plan(257, _PARAMS) is first


def test_python_byte_override_precedes_native_fallback(monkeypatch):
    monkeypatch.delenv("SPLINEOPS_PLAN_CACHE_BYTES")
    monkeypatch.setenv("LSRESIZE_PLAN_CACHE_BYTES", "0")
    fallback_first = core._get_plan(258, _PARAMS)
    fallback_second = core._get_plan(258, _PARAMS)
    assert fallback_second is not fallback_first

    monkeypatch.setenv("SPLINEOPS_PLAN_CACHE_BYTES", " +1048576 ")
    override_first = core._get_plan(258, _PARAMS)
    assert core._get_plan(258, _PARAMS) is override_first


def test_count_limit_remains_an_independent_lru_bound(monkeypatch):
    monkeypatch.setenv("SPLINEOPS_PLAN_CACHE_SIZE", "1")
    first = core._get_plan(259, _PARAMS)
    core._get_plan(260, _PARAMS)

    assert core._get_plan(259, _PARAMS) is not first

    monkeypatch.setenv("SPLINEOPS_PLAN_CACHE_SIZE", "0")
    disabled_first = core._get_plan(261, _PARAMS)
    assert core._get_plan(261, _PARAMS) is not disabled_first


def test_byte_weight_evicts_least_recent_plan(monkeypatch):
    monkeypatch.setenv("SPLINEOPS_PLAN_CACHE_BYTES", "0")
    first_uncached = core._get_plan(264, _PARAMS)
    second_uncached = core._get_plan(265, _PARAMS)
    budget = max(
        core._plan_memory_bytes(first_uncached),
        core._plan_memory_bytes(second_uncached),
    )

    monkeypatch.setenv("SPLINEOPS_PLAN_CACHE_BYTES", str(budget))
    first = core._get_plan(264, _PARAMS)
    second = core._get_plan(265, _PARAMS)

    assert core._get_plan(265, _PARAMS) is second
    assert core._get_plan(264, _PARAMS) is not first


def test_plan_memory_accounting_covers_numpy_payloads():
    plan = core._get_plan(262, _PARAMS)
    array_bytes = sum(
        value.nbytes
        for value in vars(plan).values()
        if isinstance(value, np.ndarray)
    )

    assert core._plan_memory_bytes(plan) >= array_bytes


def test_racing_builders_converge_on_one_cached_plan(monkeypatch):
    barrier = threading.Barrier(4)
    original = core.make_plan_1d

    def synchronized_build(size, params):
        barrier.wait(timeout=10)
        return original(size, params)

    monkeypatch.setattr(core, "make_plan_1d", synchronized_build)
    with ThreadPoolExecutor(max_workers=4) as executor:
        plans = list(executor.map(lambda _: core._get_plan(1024, _PARAMS), range(4)))
    monkeypatch.setattr(core, "make_plan_1d", original)

    assert all(plan is plans[0] for plan in plans)


@pytest.mark.skipif(os.name != "posix", reason="fork is POSIX-only")
def test_cache_lock_is_reinitialized_in_forked_child():
    inherited = core._get_plan(263, _PARAMS)
    locked = threading.Event()
    release = threading.Event()

    def hold_lock():
        with core._PLAN_CACHE_LOCK:
            locked.set()
            release.wait(timeout=10)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert locked.wait(timeout=10)

    child = os.fork()
    if child == 0:
        try:
            import signal

            signal.alarm(5)
            if core._get_plan(263, _PARAMS) is not inherited:
                os._exit(2)
        except BaseException:
            os._exit(1)
        os._exit(0)

    release.set()
    holder.join(timeout=10)
    _, status = os.waitpid(child, 0)

    assert not holder.is_alive()
    assert os.WIFEXITED(status)
    assert os.WEXITSTATUS(status) == 0

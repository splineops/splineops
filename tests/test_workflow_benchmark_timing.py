"""Exercise timing units and pause resistance without wall-clock assertions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def benchmark():
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark_workflows.py"
    spec = importlib.util.spec_from_file_location("workflow_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_does_not_mistake_one_pause_for_sufficient_work(
    benchmark, monkeypatch
):
    now = 0.0
    paused = False

    def call():
        nonlocal now, paused
        now += 0.1 if not paused else 0.001
        paused = True

    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: now)
    iterations, history = benchmark._calibrate(
        {"reference": call, "optimized": call}, ["reference", "optimized"], 0.05
    )
    assert iterations >= 50
    assert history[0]["block_seconds"]["reference"][0] >= 0.05
    assert all(min(values) >= 0.05 for values in history[-1]["block_seconds"].values())


def test_paired_measurements_report_per_call_units_and_preserve_errors(
    benchmark, monkeypatch
):
    now = 0.0

    def reference():
        nonlocal now
        now += 0.002
        return np.array([1.0])

    def optimized():
        nonlocal now
        now += 0.001
        return np.array([1.125])

    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: now)
    row = benchmark._measure_workflow(
        "controlled",
        reference,
        optimized,
        retained_bytes=0,
        repeats=10,
        warmups=3,
        min_sample_seconds=0.05,
        rng=np.random.default_rng(42),
    )
    assert row["reference_seconds"] == pytest.approx(0.002)
    assert row["optimized_seconds"] == pytest.approx(0.001)
    assert row["speedup"] == pytest.approx(2)
    assert row["paired_speedup_median"] == pytest.approx(2)
    assert row["paired_speedup_relative_iqr"] == pytest.approx(0, abs=1e-10)
    assert row["max_abs_difference"] == 0.125
    timing = row["timing"]
    assert timing["order"].count(["reference", "optimized"]) == 5
    assert timing["order"].count(["optimized", "reference"]) == 5
    for name, cost in (("reference", 0.002), ("optimized", 0.001)):
        assert len(timing["block_seconds"][name]) == 10
        assert min(timing["block_seconds"][name]) >= 0.05
        assert timing["seconds_per_call"][name] == pytest.approx([cost] * 10)
        np.testing.assert_allclose(
            np.asarray(timing["block_seconds"][name]) / row["iterations_per_sample"],
            timing["seconds_per_call"][name],
        )


def test_calibration_has_a_finite_work_limit(benchmark, monkeypatch):
    monkeypatch.setattr(benchmark, "_MAX_BATCH_ITERATIONS", 4)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: 0.0)
    with pytest.raises(RuntimeError, match="Unable to calibrate"):
        benchmark._calibrate(
            {"reference": lambda: None, "optimized": lambda: None},
            ["reference", "optimized"],
            0.05,
        )


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf"])
def test_invalid_calibration_duration_is_rejected(benchmark, monkeypatch, duration):
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["benchmark_workflows.py", "--min-sample-seconds", duration],
    )
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2

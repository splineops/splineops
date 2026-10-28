from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("splineops._lsresize") is None,
    reason="native resize extension is not available",
)
@pytest.mark.parametrize("shape, batched", [((16, 16, 16), True), ((8, 9, 10), False)])
def test_automatic_cubic_ls_uses_batching_only_above_line_guard(shape, batched):
    # Observe execution counters, not wall-clock thresholds. This catches an
    # accidentally unreachable optimized path while retaining the tiny-volume
    # guard that the performance experiment showed was necessary.
    script = f"""
import numpy as np
from splineops.resize import resize_degrees
values = np.ones({shape!r}, dtype=np.float64)
actual = resize_degrees(values, zoom_factors=0.37,
                        interp_degree=3, analy_degree=3, synthe_degree=3)
np.testing.assert_allclose(actual, 1.0, atol=2e-12, rtol=0)
"""
    env = os.environ.copy()
    env.update(
        {
            "LSRESIZE_PROFILE": "1",
            "LSRESIZE_NUM_THREADS": "1",
            "LSRESIZE_BATCHED_AXIS": "auto",
            "SPLINEOPS_ACCEL": "always",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    match = re.search(
        r"^batched\.projection\.total,(\d+),", result.stderr, re.MULTILINE
    )
    calls = int(match.group(1)) if match else 0
    assert (calls > 0) == batched, result.stderr


@pytest.mark.skipif(os.name != "posix", reason="fork is POSIX-only")
@pytest.mark.skipif(
    importlib.util.find_spec("splineops._lsresize") is None,
    reason="native resize extension is not available",
)
def test_profile_counters_are_process_local_after_fork() -> None:
    script = textwrap.dedent("""
        import os
        import sys

        import numpy as np
        from splineops.resize import resize

        data = np.arange(48 * 40, dtype=np.float64).reshape(48, 40)
        kwargs = {
            "output_size": (31, 27),
            "method": "cubic-antialiasing",
        }

        # Populate every profiling phase used by this operation in the parent.
        resize(data, **kwargs)

        child = os.fork()
        if child == 0:
            resize(data, **kwargs)
            sys.exit(0)

        waited, status = os.waitpid(child, 0)
        if waited != child or not os.WIFEXITED(status) or os.WEXITSTATUS(status):
            raise SystemExit(3)
        """)
    env = os.environ.copy()
    env.update(
        {
            "LSRESIZE_PROFILE": "1",
            "LSRESIZE_NUM_THREADS": "1",
            "LSRESIZE_PERSISTENT_THREADS": "0",
            "SPLINEOPS_ACCEL": "always",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    # The child's atexit summary is emitted before the parent exits. Each
    # process ran exactly two axis passes. Inheriting the parent's counters
    # would make the first count four instead of two.
    axis_calls = [
        int(value)
        for value in re.findall(
            r"^nd\.axis\.total,(\d+),", result.stderr, flags=re.MULTILINE
        )
    ]
    assert axis_calls[:2] == [2, 2], result.stderr

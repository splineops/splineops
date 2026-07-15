from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import textwrap

import pytest


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

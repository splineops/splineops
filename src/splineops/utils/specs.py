# splineops/src/splineops/utils/specs.py
"""
splineops.utils.specs
=====================

Lightweight runtime context helper so benchmark tables can be interpreted
in context (Python/OS/CPU/versions/threading env etc).

No hard dependencies: if ``threadpoolctl`` is present, we also report
BLAS/OpenMP thread pools; otherwise we skip that part.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RuntimeContext:
    # Core platform
    python_version: str
    python_impl: str
    os_system: str
    os_release: str
    machine: str
    cpu_name: str
    logical_cores: Optional[int]

    # Libraries
    numpy_version: str
    scipy_version: str
    matplotlib_version: str
    matplotlib_backend: str
    splineops_version: str
    native_present: bool  # splineops._lsresize available?

    # Env vars that affect perf/threading
    env: Dict[str, str]

    # Optional threadpoolctl info
    threadpools: List[Dict[str, str]]


def _safe_import_version(pkg: str) -> str:
    try:
        mod = __import__(pkg)
        return getattr(mod, "__version__", "n/a")
    except Exception:
        return "n/a"


def _matplotlib_info() -> tuple[str, str]:
    try:
        import matplotlib as mpl
        return getattr(mpl, "__version__", "n/a"), mpl.get_backend()
    except Exception:
        return "n/a", "n/a"


def _splineops_info() -> tuple[str, bool]:
    try:
        import importlib.util as _util
        import splineops as _sops
        ver = getattr(_sops, "__version__", "<dev>")
        native = _util.find_spec("splineops._lsresize") is not None
        return ver, native
    except Exception:
        return "n/a", False


def collect_runtime_context(include_threadpools: bool = True) -> RuntimeContext:
    # core platform
    py_ver = platform.python_version()
    py_impl = platform.python_implementation()
    os_sys = platform.system()
    os_rel = platform.release()
    machine = platform.machine() or "unknown"
    cpu_name = platform.processor() or platform.uname().processor or "unknown"
    logical = os.cpu_count()

    # libs
    np_ver = _safe_import_version("numpy")
    sp_ver = _safe_import_version("scipy")
    mpl_ver, mpl_backend = _matplotlib_info()
    so_ver, native = _splineops_info()

    # perf-relevant env
    env_keys = (
        "SPLINEOPS_ACCEL",
        "OMP_NUM_THREADS",
        "OMP_DYNAMIC",
        "OMP_PROC_BIND",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    env = {k: v for k in env_keys if (v := os.environ.get(k)) is not None}

    # optional threadpoolctl
    tps: List[Dict[str, str]] = []
    if include_threadpools:
        try:
            from threadpoolctl import threadpool_info  # type: ignore
            for info in threadpool_info():
                tps.append({
                    "internal_api": str(info.get("internal_api") or ""),
                    "class": str(info.get("class") or ""),
                    "num_threads": str(info.get("num_threads") or ""),
                    "filename": str(info.get("filename") or ""),
                })
        except Exception:
            pass

    return RuntimeContext(
        python_version=py_ver,
        python_impl=py_impl,
        os_system=os_sys,
        os_release=os_rel,
        machine=machine,
        cpu_name=cpu_name,
        logical_cores=logical,
        numpy_version=np_ver,
        scipy_version=sp_ver,
        matplotlib_version=mpl_ver,
        matplotlib_backend=mpl_backend,
        splineops_version=so_ver,
        native_present=native,
        env=env,
        threadpools=tps,
    )


def format_runtime_context(ctx: RuntimeContext) -> str:
    lines = []
    lines.append("Runtime context:")
    lines.append(f"  Python      : {ctx.python_version} ({ctx.python_impl})")
    lines.append(f"  OS          : {ctx.os_system} {ctx.os_release} ({ctx.machine})")
    lines.append(f"  CPU         : {ctx.cpu_name} | logical cores: {ctx.logical_cores}")
    lines.append(f"  NumPy/SciPy : {ctx.numpy_version}/{ctx.scipy_version}")
    lines.append(f"  Matplotlib  : {ctx.matplotlib_version} | backend: {ctx.matplotlib_backend}")
    lines.append(f"  splineops   : {ctx.splineops_version} | native ext present: {ctx.native_present}")
    for k, v in ctx.env.items():
        lines.append(f"  {k}={v}")
    if ctx.threadpools:
        for tp in ctx.threadpools:
            lib = tp.get("internal_api") or tp.get("class") or "threadpool"
            lines.append(
                f"  threadpool  : {lib}  threads={tp.get('num_threads','')}  lib={tp.get('filename','')}"
            )
    return "\n".join(lines)


def print_runtime_context(include_threadpools: bool = True) -> None:
    """Collect and print a compact, human-friendly runtime summary."""
    ctx = collect_runtime_context(include_threadpools=include_threadpools)
    print(format_runtime_context(ctx))


__all__ = [
    "RuntimeContext",
    "collect_runtime_context",
    "format_runtime_context",
    "print_runtime_context",
]

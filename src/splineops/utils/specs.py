# splineops/src/splineops/utils/specs.py
"""
splineops.utils.specs
=====================

Lightweight runtime context helper so benchmark tables can be interpreted
in context (Python/OS/CPU/versions/threading env etc).

No hard dependencies:
- If ``threadpoolctl`` is present, we also report BLAS/OpenMP thread pools.
- If ``psutil`` is present, we report OS-level thread count for this process.
- If optional image/ML libs (Pillow, OpenCV, scikit-image, PyTorch) are present,
  we also report their versions.
"""

from __future__ import annotations

import os
import platform
import threading
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

    # Optional libraries (only meaningful if installed)
    pillow_version: str
    opencv_version: str
    skimage_version: str
    torch_version: str

    # Env vars that affect perf/threading
    env: Dict[str, str]

    # Optional threadpoolctl info (BLAS/OpenMP threadpools)
    threadpools: List[Dict[str, str]]

    # Process-level threading info
    process_pid: int
    python_threads: int           # threading.active_count()
    process_threads: Optional[int]  # OS threads in this process (via psutil, if available)

    # splineops / lsresize-specific config (derived from env)
    lsresize_num_threads: Optional[int]
    lsresize_parallel_threshold: Optional[float]


def _safe_import_version(pkg: str) -> str:
    """
    Import a package by name and return its __version__ attribute if present.
    If import fails or no __version__ is found, return 'n/a'.
    """
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
    # Core platform
    py_ver = platform.python_version()
    py_impl = platform.python_implementation()
    os_sys = platform.system()
    os_rel = platform.release()
    machine = platform.machine() or "unknown"
    cpu_name = platform.processor() or platform.uname().processor or "unknown"
    logical = os.cpu_count()

    # Library versions
    np_ver = _safe_import_version("numpy")
    sp_ver = _safe_import_version("scipy")
    mpl_ver, mpl_backend = _matplotlib_info()
    so_ver, native = _splineops_info()

    # Optional / relevant libs (only meaningful if installed)
    pil_ver = _safe_import_version("PIL")
    cv2_ver = _safe_import_version("cv2")
    skimage_ver = _safe_import_version("skimage")
    torch_ver = _safe_import_version("torch")

    # Perf-relevant env
    env_keys = (
        "SPLINEOPS_ACCEL",
        "LSRESIZE_NUM_THREADS",
        "LSRESIZE_PARALLEL_THRESHOLD",
        "OMP_NUM_THREADS",
        "OMP_DYNAMIC",
        "OMP_PROC_BIND",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    env: Dict[str, str] = {
        k: v for k in env_keys if (v := os.environ.get(k)) is not None
    }

    # Derive lsresize-specific config from env
    lsresize_num_threads: Optional[int] = None
    lsresize_parallel_threshold: Optional[float] = None

    if "LSRESIZE_NUM_THREADS" in env:
        try:
            v = int(env["LSRESIZE_NUM_THREADS"])
            if v > 0:
                lsresize_num_threads = v
        except Exception:
            lsresize_num_threads = None

    if "LSRESIZE_PARALLEL_THRESHOLD" in env:
        try:
            t = float(env["LSRESIZE_PARALLEL_THRESHOLD"])
            if t > 0.0:
                lsresize_parallel_threshold = t
        except Exception:
            lsresize_parallel_threshold = None

    # Optional threadpoolctl info (BLAS/OpenMP pools)
    tps: List[Dict[str, str]] = []
    if include_threadpools:
        try:
            from threadpoolctl import threadpool_info  # type: ignore
            for info in threadpool_info():
                tps.append(
                    {
                        "internal_api": str(info.get("internal_api") or ""),
                        "class": str(info.get("class") or ""),
                        "num_threads": str(info.get("num_threads") or ""),
                        "filename": str(info.get("filename") or ""),
                    }
                )
        except Exception:
            pass

    # Process-level threading
    process_pid = os.getpid()
    python_threads = threading.active_count()
    process_threads: Optional[int] = None
    try:
        import psutil  # type: ignore

        p = psutil.Process(process_pid)
        process_threads = p.num_threads()
    except Exception:
        process_threads = None

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
        pillow_version=pil_ver,
        opencv_version=cv2_ver,
        skimage_version=skimage_ver,
        torch_version=torch_ver,
        env=env,
        threadpools=tps,
        process_pid=process_pid,
        python_threads=python_threads,
        process_threads=process_threads,
        lsresize_num_threads=lsresize_num_threads,
        lsresize_parallel_threshold=lsresize_parallel_threshold,
    )


def format_runtime_context(ctx: RuntimeContext) -> str:
    lines: List[str] = []
    lines.append("Runtime context:")
    lines.append(f"  Python      : {ctx.python_version} ({ctx.python_impl})")
    lines.append(f"  OS          : {ctx.os_system} {ctx.os_release} ({ctx.machine})")
    lines.append(f"  CPU         : {ctx.cpu_name} | logical cores: {ctx.logical_cores}")

    # Process / threading
    proc_line = (
        f"  Process     : pid={ctx.process_pid} | Python threads={ctx.python_threads}"
    )
    if ctx.process_threads is not None:
        proc_line += f" | OS threads={ctx.process_threads}"
    lines.append(proc_line)

    # Libraries
    lines.append(f"  NumPy/SciPy : {ctx.numpy_version}/{ctx.scipy_version}")
    lines.append(
        f"  Matplotlib  : {ctx.matplotlib_version} | backend: {ctx.matplotlib_backend}"
    )
    lines.append(
        f"  splineops   : {ctx.splineops_version} | native ext present: {ctx.native_present}"
    )

    # Optional image / ML libs (only if installed)
    extra_parts: List[str] = []
    if ctx.pillow_version != "n/a":
        extra_parts.append(f"Pillow={ctx.pillow_version}")
    if ctx.opencv_version != "n/a":
        extra_parts.append(f"OpenCV={ctx.opencv_version}")
    if ctx.skimage_version != "n/a":
        extra_parts.append(f"scikit-image={ctx.skimage_version}")
    if ctx.torch_version != "n/a":
        extra_parts.append(f"PyTorch={ctx.torch_version}")
    if extra_parts:
        lines.append("  Extra libs  : " + ", ".join(extra_parts))

    # lsresize-specific config
    if (ctx.lsresize_num_threads is not None) or (
        ctx.lsresize_parallel_threshold is not None
    ):
        parts = []
        if ctx.lsresize_num_threads is not None:
            parts.append(f"threads={ctx.lsresize_num_threads}")
        if ctx.lsresize_parallel_threshold is not None:
            parts.append(f"parallel_threshold={ctx.lsresize_parallel_threshold}")
        lines.append("  lsresize    : " + ", ".join(parts))

    # Environment variables
    for k, v in ctx.env.items():
        lines.append(f"  {k}={v}")

    # BLAS/OpenMP threadpools
    if ctx.threadpools:
        for tp in ctx.threadpools:
            lib = tp.get("internal_api") or tp.get("class") or "threadpool"
            lines.append(
                "  threadpool  : "
                f"{lib}  threads={tp.get('num_threads','')}  lib={tp.get('filename','')}"
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

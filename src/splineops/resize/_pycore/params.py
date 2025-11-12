# splineops/src/splineops/resize/_pycore/params.py
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class LSParams:
    interp_degree: int   # n
    analy_degree:  int   # n1  (-1 for Standard interpolation)
    synthe_degree: int   # n2  (usually = n)
    zoom:          float # a
    shift:         float # b
    inversable:    bool  # size policy

@dataclass
class Plan1D:
    # input/output sizes
    N: int
    outN: int
    out_total: int
    length_total: int
    symmetric_ext: bool  # same meaning as C++

    # padding (single contiguous buffer [LP | ext | RP])
    left_pad:  int
    right_pad: int

    # window metadata (CSR-like signature kept for compatibility)
    kmin:     np.ndarray         # (out_total,) int32
    win_len:  np.ndarray         # (out_total,) int32
    row_ptr:  np.ndarray         # (out_total+1,) int32

    # weights (we use weights2d at runtime; keep fields for API compat)
    weights:    np.ndarray       # (row_ptr[-1],) float64
    win_len_max: int
    idx2d:      np.ndarray       # (out_total, win_len_max) int64 (for gather)
    weights2d:  np.ndarray       # (out_total, win_len_max) float64, zero-padded rows

    lp_dst:   np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.intp))  # positions [0..LP-1]
    lp_src:   np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.intp))  # coeff indices for left pad
    lp_sign:  float      = 1.0
    rp_src:   np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.intp))  # coeff indices for ext[N:]
    rp_sign:  float      = 1.0

@dataclass
class Work1D:
    # Reused scratch buffers to avoid per-column allocations
    coeff:     np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    ext:       np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    ext_full:  np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    gather2d:  np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    y:         np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

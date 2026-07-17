# MiniVess 8x vessel-overview confirmation

This is the single confirmation run defined by `PROTOCOL.md`. It was executed
from commit `25a729e`, whose history contains the separately frozen protocol
(`cd7a2ea`) and runner (`7d1f7a4`). The overall predeclared claim **failed**:
all five quality margins passed, but two of the four speed conditions failed.

Across 62 held-out `512 x 512 x Z -> 64 x 64 x Z` volumes, SplineOps cubic
projection antialiasing achieved mean top-prevalence vessel Dice 0.82027. Its
paired mean improvements and one-sided Bonferroni lower bounds were:

| Comparison | Mean Dice gain | Family-wise lower bound | Quality pass | Runtime ratio | Speed pass |
| --- | ---: | ---: | :---: | ---: | :---: |
| SciPy Gaussian+cubic | +0.03421 | +0.02909 | yes | 3.28x | yes (>=3x) |
| scikit-image cubic AA | +0.03568 | +0.03000 | yes | 3.26x | yes (>=3x) |
| PyTorch area | +0.00983 | +0.00701 | yes | 0.60x | no (>1x required) |
| OpenCV area | +0.00983 | +0.00702 | yes | 0.49x | not required |
| SciPy polyphase FIR | +0.02756 | +0.02251 | yes | 2.45x | no (>=3x required) |

Runtime ratio is comparison median divided by SplineOps median. SplineOps was
therefore about 1.67x slower than PyTorch area and 2.05x slower than OpenCV
area, while remaining 2.45x--3.28x faster than the quality-oriented SciPy,
scikit-image, and polyphase pipelines. The result supports a quality advantage
for this narrow vessel-ranking endpoint, not the frozen joint quality-at-speed
claim and not segmentation, animal-level, or broad resampling superiority.

The run forced one CPU thread for SplineOps, OpenMP/BLAS, OpenCV, and PyTorch,
used one warm-up and three measured public calls per method and volume, and
used the same 20,000 bootstrap resample indices for every comparison. Reproduce
it with:

```shell
python -m pip install -e '.[minivess-study]'
python scripts/benchmark_minivess_overview.py \
  --output-dir benchmarks/minivess
```

`results.json` is the canonical machine-readable decision. The CSV files
retain every score and timing, and `minivess_overview.png` visualizes the two
primary summaries. Source NIfTI volumes are checksum-verified from EBRAINS and
are not redistributed by this repository.

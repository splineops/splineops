# SELMA3D microvessel quality-at-speed confirmation

This is the strongest real-data SplineOps result currently in the repository.
Both frozen decisions passed on 18 held-out expert-labelled light-sheet
microscopy patches.

For lateral `500×500×50 → 250×250×50` coarsening of the WGA microvessel
channel, SplineOps projection achieved mean voxel-ranking ROC AUC 0.92737.
The named comparisons scored 0.92665 for SciPy Gaussian+cubic, 0.92217 for
scikit-image cubic antialiasing, and 0.92218 for PyTorch area. Bonferroni
one-sided paired-bootstrap lower bounds for the three SplineOps differences
were all above zero.

Recorded one-thread median runtimes were 115 ms for SplineOps, 1,441 ms for
SciPy, 1,750 ms for scikit-image, and 157 ms for PyTorch area. The corresponding
speed ratios were 12.49×, 15.17×, and 1.36×.

The result supports superiority only for these named methods, data, voxel-
ranking metric, geometry, and machine. It is not segmentation superiority or
broad resampling superiority. Patch-level specimen grouping is unavailable,
and the source's official metadata conflicts between CC BY and CC BY-NC; the
study uses the stricter CC BY-NC interpretation and does not redistribute data.

Read `PROTOCOL.md` before quoting the result. Reproduce it from the repository
root with:

```shell
python -m pip install -e '.[selma3d-study]'
python scripts/benchmark_selma3d_vessels.py \
    --output-dir benchmarks/selma3d-vessels
```

`results.json` records the full-precision result, decision flags, protocol,
environment, and package versions. The CSV files contain patch-level scores,
timings, summaries, and paired comparisons. No source image is stored here.

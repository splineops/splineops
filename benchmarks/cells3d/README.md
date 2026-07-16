# Public `cells3d` downsampling study

This directory contains the recorded result of the 3-D microscopy case study.
The source TIFF is CC0, downloaded from the scikit-image data repository, and
verified with the SHA-256 recorded in `results.json`. It is not vendored here.

Reproduce the recorded run from the repository root:

```shell
python -m pip install -e '.[study]'
python scripts/benchmark_cells3d_downsampling.py \
    --warmups 1 --repeats 3 \
    --output-dir benchmarks/cells3d
```

Files:

- `results.json`: provenance, environment, metric caveats, and full results;
- `results.csv`: flat method comparison;
- `cells3d_montage.png`: central slices for both channels;
- `cells3d_metrics.png`: runtime and quality summary.

The results are a reproducible snapshot, not golden test thresholds. Runtime is
machine-specific, and the real-data metrics are proxies. The known-target
cosine calibration is the only direct numerical accuracy check in this study.

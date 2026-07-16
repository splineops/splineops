# SELMA3D nuclei confirmation

This was the first fresh SELMA3D confirmation. It passed the frozen quality
non-inferiority rule but failed the frozen 10× CPU speed rule, so its overall
quality-at-speed claim failed.

Across 12 expert-labelled `200³ → 100³` human-brain nuclei patches, SplineOps
projection had mean voxel-ranking ROC AUC 0.98364, versus 0.98305 for SciPy
Gaussian+cubic, 0.97671 for scikit-image, and 0.97680 for PyTorch area. Local
median speedups were 6.63× and 7.40× against SciPy and scikit-image—useful, but
below the predeclared 10× requirement.

This negative decision is retained to show the path to the later anisotropic
microvessel confirmation. Reproduce it with:

```shell
python -m pip install -e '.[selma3d-study]'
python scripts/benchmark_selma3d_nuclei.py \
    --output-dir benchmarks/selma3d
```

The source data is checksum-pinned and downloaded from EMBL-EBI; it is not
redistributed by this repository. Read `PROTOCOL.md` for the license conflict,
failed Otsu pilot, patch-grouping limitation, and exact decision rule.

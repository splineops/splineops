# BBBC050 segmentation validation

The predeclared SplineOps quality-superiority criterion was **not met against
every baseline**. The recorded result is intentionally retained as evidence.

Headline findings:

- SplineOps projection beat SplineOps cubic without antialiasing by 0.0520
  cross-validation Dice.
- Its +0.00387 result over SciPy was below the frozen +0.005 practical margin.
- Its +0.00219 result over scikit-image had a confidence interval crossing
  zero; scikit-image was +0.01185 better on four external embryos.
- SplineOps projection had a 4.16 ms median local resize time, versus 124.38 ms
  for the SciPy pipeline and 110.76 ms for scikit-image.

Read `PROTOCOL.md` before interpreting the result. Reproduce it with:

```shell
python -m pip install -e '.[study]'
python scripts/benchmark_bbbc050_segmentation.py \
    --output-dir benchmarks/bbbc050
```

The BBBC050 archives are downloaded, checksum-verified, and kept outside the
repository. `results.json` records provenance, environment, full-precision
summaries, comparisons, per-embryo results, and the failed conclusion.

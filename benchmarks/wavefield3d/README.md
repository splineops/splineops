# Controlled 3-D spectral-coarsening validation

The frozen numerical-superiority criterion **passed against all six
predeclared generic N-D resize alternatives** on this controlled field class.
It did not survive a post-hoc domain-specialized resampling audit.

Across 72 fresh confirmation cases, SplineOps projection had mean exact-target
NRMSE 0.1449. The nearest predeclared baseline was PyTorch area at 0.3197, a
54.7% relative reduction with a blocked 95% bootstrap interval of 47.0% to
62.0%. SplineOps was better in 70 of 72 individual comparisons against PyTorch
area and all 72 against each other frozen baseline.

After that confirmation, a stronger scientific baseline was identified. A
separable SciPy polyphase FIR pipeline achieved mean NRMSE 0.00218 and beat
SplineOps in all 72 cases. It was about 16 times slower locally. Therefore this
study supports a speed/accuracy trade-off and a comparison against generic
resize APIs—not field-wide scientific-resampling superiority.

This result is narrow by design. It concerns smooth, continuous,
mirror-compatible 3-D cosine fields containing both resolvable and
above-output-Nyquist content. It does not establish superiority for real
seismic data, shocks, conservative regridding, arbitrary boundaries, medical
images, or generic photographs.

The trade-off is visible in the component metrics: with no out-of-band content,
plain cubic interpolation preserved the passband more closely than projection.
PyTorch trilinear was also about five times faster locally, while providing no
volumetric antialiasing in this pipeline. SplineOps was about 12 times faster
than the SciPy Gaussian+cubic pipeline and 13 times faster than scikit-image in
the repeated single-thread CPU timing.

The initial implementation-validation blocks are excluded and identified in
`PROTOCOL.md`; the acceptance rule was unchanged for the fresh confirmation.
Read that protocol before interpreting the result. Reproduce the recorded run
from the repository root with:

```shell
python -m pip install -e '.[wavefield-study]'
python scripts/benchmark_wavefield3d_coarsening.py \
    --output-dir benchmarks/wavefield3d
```

`results.json` contains the protocol hash, full-precision scores, all cases,
blocked bootstrap comparisons, environment, and local timings. The CSV files
provide flat exports of the same evidence.

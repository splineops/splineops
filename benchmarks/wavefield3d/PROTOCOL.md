# Controlled 3-D spectral-coarsening protocol

Status: the field distribution, baselines, metrics, and success rule were frozen
locally on 2026-07-16. This was not registered with an independent
preregistration service. Seeds 0 through 3 were used for feasibility. During
implementation validation, the originally assigned 1000--1007, 1100--1107,
and 1200--1207 blocks were also observed. All of those seeds are excluded from
the final confirmation, intervals, and success decision. The unchanged rule is
applied to fresh confirmation blocks below.

## Claim under test

For endpoint-aligned downsampling of continuous, mirror-compatible 3-D fields
containing both resolvable and above-output-Nyquist cosine modes, SplineOps
cubic projection antialiasing has lower exact-target NRMSE than every frozen
generic N-D resize comparison pipeline.

This is a deliberately narrow numerical claim. It is relevant to repeated
coarsening of smooth simulation or wavefield arrays. It is not a claim about
discontinuities, conservative finite-volume regridding, arbitrary boundary
conditions, categorical data, GPU throughput, or generic image quality.

## Manufactured fields and exact target

The physical domain is `[0, 1]^3`. Source grids include both endpoints. Each
base field is the sum of:

- 12 low-frequency tensor-product cosine modes, with every axis below 55% of
  the corresponding output Nyquist frequency; and
- 12 nuisance modes, each above 115% of output Nyquist on one randomly chosen
  axis and no higher than 95% of source Nyquist.

Cosines make the fields exactly compatible with the mirror boundary assumed by
the tested spline and Gaussian pipelines. Coefficients are deterministic
standard-normal draws, separately normalized to unit Euclidean norm for the
low- and high-frequency mode groups. Nuisance-to-signal coefficient ratios are
0.25, 0.65, and 1.00.

The exact target is the analytical low-frequency component evaluated directly
on each pipeline's documented output grid. No tested resize output is used as
ground truth. Endpoint-aligned methods use points from 0 through 1. Half-pixel
methods use their corresponding interior point locations. For PyTorch `area`,
the half-pixel target deliberately reflects this study's continuous point-grid
objective; regional averages are a different numerical contract.

## Frozen test matrix

Three geometries cover exact twofold and non-integer anisotropic reductions:

1. `(65, 65, 65)` to `(33, 33, 33)`;
2. `(73, 81, 65)` to `(29, 41, 33)`;
3. `(97, 81, 65)` to `(25, 41, 49)`.

Eight base fields per geometry use confirmation seeds 2000 through 2007, 2100
through 2107, and 2200 through 2207 respectively. Combining 24 base fields with
the three nuisance ratios produces 72 confirmation cases. Ratio variants share
modes and are therefore kept together as one block during bootstrap resampling.

## Frozen methods

1. SplineOps endpoint-aligned cubic projection antialiasing.
2. SplineOps endpoint-aligned cubic interpolation without antialiasing.
3. SciPy Gaussian prefilter plus endpoint-aligned cubic sampling. Per-axis
   sigma is `(scale - 1) / 2`, matching the conventional scikit-image default.
4. SciPy endpoint-aligned cubic sampling without antialiasing.
5. scikit-image cubic resize with Gaussian antialiasing and its half-pixel grid.
6. PyTorch trilinear interpolation with `align_corners=True`, without
   antialiasing because volumetric trilinear antialiasing is unavailable.
7. PyTorch area resize on its native regional/half-pixel geometry.

All arrays are float32. Numerical runtimes are restricted to one CPU thread.
No method-specific parameter is fitted on the confirmation cases.

## Metrics and success criterion

Primary error is NRMSE against the exact low-frequency target. For each named
alternative and base-field block, NRMSE is averaged over the three nuisance
ratios. Paired bootstrap resampling draws the 24 blocks with replacement
20,000 times using seed 20260716.

SplineOps numerical superiority is demonstrated against an alternative only
if all of the following hold:

- mean NRMSE is at least 20% lower;
- the lower bound of the paired 95% bootstrap interval for relative NRMSE
  reduction is above 10%;
- SplineOps has lower NRMSE in at least 90% of the 72 individual cases; and
- SplineOps is not worse in mean NRMSE in any of the nine frozen
  geometry-by-nuisance-ratio strata.

The overall claim passes only if every named alternative passes. Passband
distortion, stopband leakage, individual-case errors, and failed comparisons
are published regardless of the result.

## Repeated-workload timing

Timing is secondary and machine-specific. For each geometry, fixed plans or
equivalent precomputed coordinates are built before measurement. Two warm-ups
and seven repetitions are run on eight deterministic random volumes. The
artifact reports plan/setup time and median application time per volume.

A runtime statement is allowed only for a named method when SplineOps is
faster in all three geometries. No runtime superiority against PyTorch is
required for the numerical-quality claim.

## Known limitations

The exact target gives stronger numerical evidence than a round trip, but it is
a manufactured spectral task. The transition band is intentionally separated
from output Nyquist, phases are constrained by mirror-compatible boundaries,
and the fields contain no shocks. A positive result supports the stated field
class; it does not establish better downstream science on real seismic,
fluid-dynamics, medical, or microscopy datasets.

## Post-hoc adversarial audit

After the confirmation passed, a domain-relevant omission was identified:
scientific users can compose `scipy.signal.resample_poly` along the three axes
instead of using an imaging resize API. This method was not part of the frozen
success decision and cannot retroactively change that decision. It is added to
the published artifact as an explicitly post-hoc eighth method, using the exact
endpoint interval ratios and reflect padding.

The broader scientific-resampling claim fails if this audit method is more
accurate. Its scores, timings, and failed comparison must be published beside
the frozen result.

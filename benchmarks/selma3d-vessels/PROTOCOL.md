# Frozen SELMA3D microvessel confirmation protocol

Status: frozen locally on 2026-07-16 before running any resizing method on
patches 006 through 023. This is not an independent preregistration. Patches
000 through 005 were used for feasibility and method selection and are
excluded from all confirmation estimates.

## Claim under test

For twofold lateral coarsening of anisotropic 3-D light-sheet microscopy
patches, SplineOps cubic projection antialiasing preserves expert-labelled
microvessel-versus-background voxel ranking within 0.01 ROC AUC of each named
antialiased comparison, while running at least 10 times faster than the named
SciPy and scikit-image CPU pipelines on this machine.

This is a quality-at-speed claim about an imaging preprocessing step. It is
not a claim of better segmentation, better vascular measurements, GPU
superiority, or state-of-the-art SELMA3D challenge performance.

## Data, channel, and pilot separation

The source is the annotated `VessAP_vessel` subset of BioImage Archive
accession S-BIAD1196. Each patch contains two image channels and an expert
mask. The primary input is channel `_0000`, identified by the dataset as the
WGA microvessel channel. Channel `_0001` and a per-voxel maximum of both
channels were evaluated in the six-patch pilot; channel `_0000` was selected
before confirmation because it ranked the vessel mask substantially better.

The pilot mean ROC AUCs for channel `_0000` were 0.92161 SplineOps, 0.92044
SciPy Gaussian+cubic, 0.91951 scikit-image, and 0.91936 PyTorch area. Pilot
method timing resized both channels and is not reused in the confirmation.

The confirmation set is the remaining 18 patches, IDs 006 through 023. Each
raw channel and mask has spatial shape 500 x 500 x 50 after removing any
singleton channel dimension. The study downloads channel `_0000` and masks
from the official EMBL-EBI archive, verifies a pinned SHA-256 digest for every
file, and does not redistribute source data.

The archive metadata says CC BY 4.0, while the official SELMA3D challenge page
says CC BY-NC. This study follows the stricter CC BY-NC interpretation and
cites the challenge and source dataset. That conflict must remain visible in
published documentation.

The deposit does not expose specimen identity for each patch. Patches are the
bootstrap resampling units, so the interval cannot establish specimen-level
generalization.

## Frozen preprocessing and methods

The WGA channel is clipped and linearly normalized using each patch's 1st and
99.9th percentiles. The first two NIfTI axes are reduced from 500 to 250; the
50-plane third axis is retained. Arrays are float32, and numerical runtimes
use one CPU thread.

1. SplineOps endpoint-aligned cubic projection antialiasing.
2. SplineOps endpoint-aligned cubic interpolation without antialiasing, as a
   negative control rather than a superiority target.
3. SciPy Gaussian prefilter followed by endpoint-aligned cubic sampling. The
   lateral Gaussian sigma is `(scale - 1) / 2`; the retained axis is not
   filtered.
4. scikit-image native half-pixel cubic resize with Gaussian antialiasing.
5. PyTorch area resize on its native regional/half-pixel geometry.

Binary labels are sampled by nearest neighbour at each method's documented
output-grid locations; exact half-grid ties select the higher source index.
No threshold, model, or parameter is fitted on the confirmation patches.

## Metrics and frozen success rule

The primary endpoint is per-patch ROC AUC for ranking labelled vessel voxels
above background voxels on the reduced grid. Ties receive average ranks.
Average precision and top-prevalence Dice are secondary and cannot rescue a
failed primary endpoint.

For each antialiased comparison, 20,000 paired bootstrap resamples draw the 18
patches with replacement using seed 20260716. Quality non-inferiority passes
only when the lower bound of the 95% interval for
`AUC(SplineOps) - AUC(comparison)` is strictly above -0.01. SplineOps must pass
against SciPy, scikit-image, and PyTorch area.

Timing uses two warm-ups and seven repetitions per method and patch. Setup or
plan creation is included. The per-patch median is summarized across the 18
patches. The speed condition passes only if both ratios
`median_time(SciPy or scikit-image) / median_time(SplineOps)` are at least 10.
No speed claim against PyTorch is required.

The overall narrow claim passes only if both quality and speed pass. Every
score, interval, timing, secondary metric, failed comparison, and the
no-antialiasing control is published regardless of outcome.

A stricter, family-wise "superiority" decision is also frozen before
confirmation. For each of the three quality comparisons, the one-sided lower
bound is the 1.667th percentile of its paired bootstrap distribution
(Bonferroni alpha 0.05 / 3). Strict superiority passes only if all three lower
bounds exceed zero, the SciPy and scikit-image speed ratios are at least 10,
and the PyTorch area speed ratio exceeds 1. This secondary rule supports only
the exact named methods, data, metric, geometry, and machine above.

## Known limitations

- The confirmation contains 18 patches with undisclosed specimen grouping.
- The endpoint is voxel ranking, not a deployed segmentation model.
- The comparison covers one exact lateral twofold geometry and CPU execution.
- Native endpoint and half-pixel grids answer slightly different sampling
  contracts; each is scored against labels on its own documented grid.
- The protocol does not compare learned downsamplers or every scientific
  resampling implementation.
- The dataset's conflicting license metadata limits clean commercial use.

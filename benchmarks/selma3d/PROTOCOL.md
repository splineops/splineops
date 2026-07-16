# Frozen SELMA3D nuclei confirmation protocol

Status: frozen locally on 2026-07-16 before running any resizing method on the
confirmation subset. This is not an independent preregistration. The c-Fos
subset was used as a feasibility pilot and is excluded from confirmation
estimates. An earlier label-free Otsu endpoint failed on that pilot (mean Dice
about 0.033 for every method) and is not presented as supporting evidence.

## Claim under test

For twofold coarsening of 200 x 200 x 200 human-brain light-sheet microscopy
patches, SplineOps cubic projection antialiasing preserves expert-labelled
nucleus-versus-background voxel ranking within 0.01 ROC AUC of each named
antialiased comparison, while running at least 10 times faster than the named
SciPy and scikit-image CPU pipelines on this machine.

This is a quality-at-speed claim about a preprocessing step. It is not a claim
of better segmentation, better biological conclusions, GPU superiority, or
state-of-the-art SELMA3D challenge performance.

## Data and separation from the pilot

The confirmation set is all 12 annotated `shannel_cells` patches from
BioImage Archive accession S-BIAD1196. Each raw image and corresponding binary
mask is 200 x 200 x 200. The study downloads files from the official EMBL-EBI
archive and verifies a pinned SHA-256 digest for every file. It does not
redistribute the source data.

The archive metadata says CC BY 4.0, while the official SELMA3D challenge page
says CC BY-NC. This study follows the stricter CC BY-NC interpretation and
cites the challenge and source dataset. That conflict must remain visible in
published documentation.

The 19 annotated `cFos-Active_Neurons` patches were inspected during pilot
development. Their method results, including mean ROC AUC of 0.9614 for
SplineOps, 0.9570 for SciPy Gaussian+cubic, 0.9475 for scikit-image, and 0.9471
for PyTorch area, are not pooled with or used in confirmation inference.

The deposit does not expose specimen identity for each patch. The 12 patches
are therefore the resampling units, and the interval cannot establish
specimen-level generalization.

## Frozen preprocessing and methods

Each image is clipped and linearly normalized using its 1st and 99.9th
percentiles. Each spatial axis is reduced from 200 to 100. Arrays are float32,
and numerical runtimes use one CPU thread.

1. SplineOps endpoint-aligned cubic projection antialiasing.
2. SplineOps endpoint-aligned cubic interpolation without antialiasing, as a
   negative control rather than a superiority target.
3. SciPy Gaussian prefilter followed by endpoint-aligned cubic sampling. The
   per-axis Gaussian sigma is `(scale - 1) / 2`.
4. scikit-image native half-pixel cubic resize with Gaussian antialiasing.
5. PyTorch area resize on its native regional/half-pixel geometry.

Binary labels are sampled by nearest neighbour at each method's documented
output-grid locations; exact half-grid ties select the higher source index.
No threshold, model, or parameter is fitted on the confirmation patches.

## Metrics and frozen success rule

The primary endpoint is per-patch ROC AUC for ranking labelled nucleus voxels
above background voxels on the reduced grid. Ties receive average ranks.
Average precision and top-prevalence Dice (selecting as many highest-score
voxels as there are positives) are secondary and cannot rescue a failed
primary endpoint.

For each antialiased comparison, 20,000 paired bootstrap resamples draw the 12
patches with replacement using seed 20260716. Quality non-inferiority passes
only when the lower bound of the 95% interval for
`AUC(SplineOps) - AUC(comparison)` is strictly above -0.01. SplineOps must pass
against SciPy, scikit-image, and PyTorch area.

Timing uses two warm-ups and seven repetitions per method and patch. Setup or
plan creation is included because this is a one-volume preprocessing case.
The per-patch median is summarized across 12 patches. The speed condition
passes only if both ratios
`median_time(SciPy or scikit-image) / median_time(SplineOps)` are at least 10.
No speed claim against PyTorch is required.

The overall narrow claim passes only if both the quality and speed conditions
pass. Every score, interval, timing, secondary metric, failed comparison, and
the no-antialiasing control is published regardless of outcome.

## Known limitations

- The confirmation contains 12 patches with undisclosed specimen grouping.
- The foreground task is voxel ranking, not a deployed segmentation model.
- The comparison covers one exact twofold geometry and CPU execution only.
- Native endpoint and half-pixel grids answer slightly different sampling
  contracts; each is scored against labels on its own documented grid.
- The protocol does not compare learned downsamplers or every scientific
  resampling implementation.
- The dataset's conflicting license metadata limits clean commercial use.

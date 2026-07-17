# Frozen MiniVess vessel-overview confirmation protocol

Status: frozen in version control on 2026-07-17 before running any resizing
method on the 62 confirmation volumes. This revision supersedes the uncommitted
2026-07-16 draft after a pre-run consistency audit; no confirmation output was
examined. This is a local, time-stamped protocol, not an independent
preregistration.

## Claim under test

For eightfold lateral coarsening of anisotropic 3-D multiphoton microscopy
volumes (512 x 512 x Z to 64 x 64 x Z), SplineOps cubic projection
antialiasing preserves expert-labelled vessel ranking more clearly than each
named comparison while remaining faster than the quality-oriented CPU
pipelines.

The primary quality endpoint is top-prevalence Dice. For each volume and
method, the highest-intensity output voxels are selected until their count
equals the positive count in that method's sampled reference mask. This
threshold-free endpoint tests whether a small overview keeps vessel voxels
near the top of the intensity ranking. It is not a deployed segmentation
score because it uses the reference prevalence.

## Data and pilot separation

MiniVess v1 (DOI `10.25493/HPBE-YHK`) contains 70 raw/mask pairs of rodent
cerebrovasculature acquired with two-photon fluorescence microscopy. The
official EBRAINS data proxy provides raw NIfTI volumes, masks, byte counts,
and archive MD5 digests. The runner downloads from that endpoint, verifies
every file it reads, and does not redistribute source data. The dataset
license is CC BY-NC-SA 4.0.

The seven validation volumes declared in the dataset supplement (06, 28, 35,
38, 44, 48, and 58) were used as pilots. Volume 68 is also a pilot because the
dataset authors publish it in their GitHub repository and it was inspected
during feasibility work. Those eight volumes selected the 8x geometry and the
top-prevalence Dice endpoint. They are excluded from confirmation.

The confirmation set is every other volume: 62 volumes spanning the source
dataset's original train and test labels. Those labels describe a separate
U-Net workflow; no model is fitted here. The supplement identifies species
but not a specimen/animal identifier for each volume. Confidence intervals
therefore resample volumes and cannot establish animal-level independence.

## Frozen methods

All methods reduce only the first two axes and retain Z. Inputs are float32
after per-volume clipping and linear scaling from the 1st to 99.9th intensity
percentiles. Timings force one CPU thread through the native SplineOps,
OpenMP/BLAS, OpenCV, and PyTorch controls. Each method receives one unmeasured
warm-up call followed by three measured public calls. Normal process-local
library caches remain enabled, so the measured calls represent warm fixed-grid
throughput rather than cold plan construction.

1. SplineOps endpoint-aligned cubic projection antialiasing.
2. SplineOps endpoint-aligned cubic interpolation without antialiasing, a
   negative control that is not a superiority target.
3. SciPy Gaussian prefilter plus endpoint-aligned cubic sampling. Gaussian
   sigma is `(endpoint_scale - 1) / 2` on each reduced axis.
4. scikit-image cubic resize with Gaussian antialiasing on its native
   half-pixel grid.
5. PyTorch area resize on its native regional/half-pixel grid.
6. OpenCV `INTER_AREA` on its native regional/half-pixel grid. OpenCV is the
   expected speed counterweight; no speed win over it is required.
7. SciPy polyphase FIR resampling with its native phase-zero grid and linear
   boundary continuation. This is the strong signal-processing countercheck.

Binary labels are sampled by nearest neighbour at each method's documented
output locations. Native grids are deliberately retained because these are
the outputs users receive from the named public calls. Grid differences are a
limitation and remain visible in the result metadata.

## Frozen success rule

The primary paired difference is
`Dice(SplineOps projection) - Dice(comparison)` for each of the five named
comparisons. Twenty thousand paired bootstrap resamples draw the 62 volumes
with replacement using seed 20260716. The same seeded resample indices are used
for all five comparisons. With five comparisons, the one-sided family-wise
lower bound is the 1st percentile (Bonferroni alpha 0.05 / 5).

Quality passes only if that lower bound is strictly above **+0.005 Dice for
every comparison**. This requires evidence for at least a half-point practical
gain, not merely a positive floating-point difference.

Timing uses one warm-up and three measured calls per method and volume. The
per-volume median is summarized across volumes. SplineOps must be at least 3x
faster than SciPy Gaussian+cubic, scikit-image, and SciPy polyphase (a ratio of
exactly 3.0 passes), and strictly faster than PyTorch area. No speed condition
is imposed against OpenCV area.

The narrow claim passes only if all five quality margins and all four speed
conditions pass. ROC AUC, average precision, the no-antialiasing control, all
individual volume scores, and all failed comparisons are reported regardless
of outcome.

## What a pass would and would not mean

A pass would support one application: CPU construction of strongly reduced
3-D microvascular overviews or pyramid levels where thin-vessel ranking and
latency both matter. It would not demonstrate segmentation superiority,
animal-level generalization, GPU superiority, the fastest possible resize,
or universal resampling superiority. OpenCV may remain the better choice when
latency matters more than the measured vessel-ranking loss.

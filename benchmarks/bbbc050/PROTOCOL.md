# BBBC050 semantic-segmentation protocol

Status: frozen locally before the full benchmark run. This was not registered
with an independent preregistration service.

## Claim under test

For twofold XY downsampling of BBBC050 3-D fluorescence volumes followed by a
simple threshold-based nuclei segmentation, SplineOps cubic projection
antialiasing improves held-out embryo-level Dice over each named alternative.

This is a deliberately narrow application claim. It is not a claim about every
segmentation model, microscopy modality, resize geometry, or computer.

## Data

- BBBC050 version 2 `Images.zip`, SHA-256
  `29f100abbfebfb1986b8e87eac091e86d8ec27cd8194f9a1c02c805e76b6dcd8`.
- BBBC050 version 2 `GroundTruth.zip`, SHA-256
  `1f19b308730dccf217c4d4dcf5745ad0fcde4eeb9f9c9306b2c8abd1fe73e5d1`.
- Training partition: 11 embryos and 11 annotated time points per embryo.
- External test partition: four embryos and 11 annotated time points per
  embryo, acquired with a different microscope and fluorophore.
- Ground truth: `GroundTruth_QCANet`, converted to foreground with `label > 0`.

A development check inspected all 11 training volumes at time point `t251` to
choose a feasible threshold range. Those volumes are excluded from the primary
cross-validation, including threshold fitting and evaluation. All 44 external
test volumes remain untouched by protocol development.

## Fixed preprocessing and methods

Each image is clipped to its 1st and 99.9th intensity percentiles and scaled to
`[0, 1]`. Z is retained. Y and X are reduced to `ceil(n / 2)`.

1. SplineOps endpoint-aligned cubic projection antialiasing.
2. SplineOps endpoint-aligned cubic interpolation without antialiasing.
3. SciPy Gaussian prefilter plus endpoint-aligned cubic sampling.
4. scikit-image cubic antialiasing on its documented resize grid.

Endpoint methods use an endpoint-aligned nearest-neighbour ground-truth grid.
scikit-image uses its corresponding order-zero resize grid. This tests each
public pipeline on its internally aligned physical grid; the grids are not
claimed to be numerically identical.

## Segmentation and validation

Segmentation is `resized_image >= threshold`, with no morphology or learned
model. Candidate thresholds are 0.040 through 0.300 in steps of 0.005.

For the primary analysis, leave one training embryo out:

1. choose each method's threshold that maximizes mean embryo-level Dice on the
   other ten embryos;
2. evaluate that threshold on the held-out embryo;
3. repeat for all 11 embryos.

For external validation, choose one threshold per method on all 11 training
embryos, still excluding `t251`, then evaluate the four test embryos. Frames
are averaged within embryo before any between-method comparison.

## Success criterion

SplineOps superiority is demonstrated only if, against every alternative:

- the primary mean paired Dice improvement is at least 0.005;
- the lower bound of a paired 95% embryo bootstrap interval is above zero; and
- the mean paired improvement on the four external embryos is non-negative.

Bootstrap resampling uses 20,000 draws and seed 20260716. Runtime is secondary
and machine-specific. The full per-embryo results are published even when the
success criterion fails.

## Known limitation

This protocol isolates a transparent threshold-segmentation application. A
positive result would not prove an advantage for a trained 3-D neural network;
that would require a separately frozen, substantially more expensive study.

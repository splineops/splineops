# SELMA3D napari comparison demo

This demo turns the held-out SELMA3D microvessel study into an interactive,
honest visual comparison. A curtain slider reveals SplineOps on the left while
one selected method remains on the right. The expert vessel mask, the current
patch's ROC AUC, locally measured runtimes, and the frozen 18-patch results are
shown together.

![Frozen study summary: patch-level vessel-ranking ROC AUC and recorded one-thread runtimes.](../benchmarks/selma3d-vessels/selma3d_vessels.png)

## Run it

Install the optional demo dependencies into a fresh virtual environment when
possible, then use the installed command:

```shell
python -m pip install 'splineops[selma3d-demo]'
splineops-selma3d-demo
```

This lightweight default compares SplineOps with SciPy, scikit-image, and
SplineOps interpolation without antialiasing. To add the PyTorch-area method:

```shell
python -m pip install 'splineops[selma3d-demo-all]'
splineops-selma3d-demo --comparisons torch_area scipy_gaussian skimage_resize
```

From a repository checkout, `python demos/selma3d_napari.py` remains a thin
compatibility launcher for the same packaged implementation.

The first run downloads only patch 013's WGA channel and expert vessel mask
from [BioImage Archive accession S-BIAD1196](https://www.ebi.ac.uk/bioimage-archive/galleries/ai/analysed-dataset/S-BIAD1196/),
verifies their pinned SHA-256 digests, and caches them under
`~/.cache/splineops/selma3d`. Source data are not redistributed by SplineOps.
Patch 013 is a post-study presentation choice with clear visual separation,
not a newly held-out selection; the frozen 18-patch result remains the primary
evidence.

In the viewer:

- move **Reveal SplineOps** from 0% (selected comparison only) to 100%
  (SplineOps only);
- switch among scikit-image, SciPy, and cubic interpolation without
  antialiasing, plus PyTorch area when the full demo extra is installed;
- toggle the expert mask; and
- use napari's first dimension slider to inspect all 50 planes.

The local numbers are deliberately labelled as local measurements. The frozen
numbers come from all 18 held-out patches and remain visible separately. Use a
headless run to check downloads, algorithms, metrics, and timing without
opening napari:

```shell
splineops-selma3d-demo --no-gui --timing-repetitions 1
```

Choose a different held-out patch, a subset of comparisons, or an existing
dataset directory with:

```shell
splineops-selma3d-demo \
    --sample 22 \
    --comparisons scipy_gaussian skimage_resize \
    --data-dir /path/to/VessAP_vessel
```

Run `splineops-selma3d-demo --help` for all options.

## What the demo does and does not show

The input is normalized exactly as in the frozen study: clipping to the 1st and
99.9th percentiles followed by linear scaling. Its first two axes are reduced
from `500×500` to `250×250`; all 50 planes are retained. Every method is scored
against labels sampled on its documented grid. The curtain mask therefore
uses endpoint labels on the SplineOps side and endpoint or half-pixel labels on
the comparison side as appropriate.

The primary metric is expert-labelled vessel-versus-background voxel-ranking
ROC AUC. It is not a segmentation metric, and no threshold or segmentation
model is fitted. The 18-patch result supports only the named methods, data,
geometry, metric, and recorded machine. Read the complete
[frozen protocol](../benchmarks/selma3d-vessels/PROTOCOL.md) before publishing
claims from it.

Official archive records and the
[SELMA3D challenge data page](https://selma3d.grand-challenge.org/data/) contain
conflicting CC BY 4.0 and CC BY-NC metadata. This repository and demo use the
stricter CC BY-NC interpretation. The demo downloads data only after it is
launched and prints the source and license warning before doing so.

# Development benchmark thresholds

`consolidation-thresholds.json` is a stored, machine-relative baseline for the
manual development benchmark workflow. It checks within-run speed ratios and
numerical differences rather than absolute seconds, which are too sensitive to
runner hardware and load.

The thresholds are deliberately broad regression alarms, not advertised
performance guarantees. A failed threshold requires inspecting the complete
JSON artifacts and semantics before accepting or rejecting a change.

`benchmark_batch_scaling.py` sweeps explicit batch count and spatial size for
affine transforms, Laplacian-only differentials, and Haar round trips. It
reports required output bytes separately from traced peak memory and normalizes
peak growth against the one-plane case. The development workflow runs smoke
evidence for relevant publication-branch changes and can run the standard
profile manually on Linux, macOS, and Windows. It publishes one artifact per
runner; platform results must be reviewed rather than averaged into one claim.

`benchmark_downstream_workflows.py` measures persisted registration fan-out and
an affine-to-differential 3-D feature pipeline. `profile_affine_phases.py`
instruments prefilter and evaluation time inside complete affine plan calls.
These artifacts are part of the stored numerical-equivalence policy. A commit
whose message contains `[standard-bench]` intentionally selects the standard
profile for the three-platform publication-branch workflow; other matching
pushes use smoke, and `workflow_dispatch` retains an explicit profile choice.

Run the checker after generating the named artifacts:

```shell
python scripts/check_benchmark_thresholds.py \
  --policy benchmarks/consolidation-thresholds.json \
  --artifacts-dir /path/to/artifacts
```

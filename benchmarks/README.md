# Development benchmark thresholds

`consolidation-thresholds.json` is a stored, machine-relative baseline for the
manual development benchmark workflow. It checks within-run speed ratios and
numerical differences rather than absolute seconds, which are too sensitive to
runner hardware and load.

The thresholds are deliberately broad regression alarms, not advertised
performance guarantees. A failed threshold requires inspecting the complete
JSON artifacts and semantics before accepting or rejecting a change.

Run the checker after generating the named artifacts:

```shell
python scripts/check_benchmark_thresholds.py \
  --policy benchmarks/consolidation-thresholds.json \
  --artifacts-dir /path/to/artifacts
```

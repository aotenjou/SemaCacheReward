# Research workflow

The scripts in `scripts/` use repository-root paths and expect generated local
inputs under `work/`. The public request snapshots are produced by
`prepare_judge_traces.py`; split files and model artifacts are local experiment
inputs and are intentionally ignored by Git.

The compact reports in `results/` are committed so the headline findings are
reviewable without downloading the datasets. Full per-request traces are
derived artifacts and are not committed.

Typical sequence:

```powershell
python research/scripts/prepare_judge_traces.py --sampling spread
python research/scripts/run_judge_vcache_benchmark.py --stage embed
python research/scripts/run_judge_vcache_benchmark.py --stage benchmark --no-resume
python research/scripts/analyze_judge_vcache_results.py
```

The risk-aware adapter can be smoke-tested independently with deterministic
hash embeddings:

```powershell
python research/scripts/run_risk_aware_judge_cache.py --dataset helpsteer2 `
  --hash-embeddings --train-pair-cap 128 --max-rows 20
```

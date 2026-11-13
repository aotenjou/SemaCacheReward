# SemaCacheReward

Research code for semantic caching of LLM Judge rewards in post-training
workloads. The project evaluates whether a cached Judge result can be reused
for a semantically similar request while keeping reward drift within an
explicit quality budget.

The repository contains:

- the upstream MVR-cache/vCache implementation under `src/mvr-cache/`;
- a framework-agnostic `JudgeCache` adapter for PKU preference labels,
  HelpSteer2 five-dimensional scores, and UltraFeedback four-dimensional
  scores;
- reproducible dataset preparation and replay scripts under
  `research/scripts/`;
- compact reports and plots under `research/results/`.

This is an offline research replay, not a claim of end-to-end RLHF speedup.
The current experiments replay fixed candidate responses and public labels;
they do not generate new policy rollouts or call a live Judge service.

## Quick start

```powershell
cd src/mvr-cache
python -m pip install -e .
python -m pip install -e vcache/vcache_core/cache/embedding_store/hnswlib
```

Prepare the fixed data snapshot from the repository root:

```powershell
python research/scripts/prepare_judge_traces.py --sampling spread
```

Run the deterministic smoke test for the risk-aware adapter:

```powershell
python research/scripts/run_risk_aware_judge_cache.py `
  --dataset helpsteer2 --hash-embeddings --train-pair-cap 128 `
  --max-rows 20 --bootstrap-replicates 50
```

For GPU replay, provide the local BGE model and MVR checkpoints, then run the
same script without `--hash-embeddings`. Paths and large generated artifacts
are intentionally local and ignored by Git.

## Research scope

The three workloads are:

- **PKU-SafeRLHF**: pairwise helpfulness and safety preferences;
- **HelpSteer2**: five-dimensional single-answer ratings;
- **UltraFeedback**: four-dimensional feedback ratings.

The primary metric is effective hit rate: a cache hit counts only when the
returned reward satisfies the configured error gate. Reports also separate
raw hits, wrong hits, per-dimension error, preference flips, and Judge calls
saved.

## Reproducibility notes

Use the scripts in `research/scripts/` from the repository root. The prepared
dataset snapshot, model weights, embeddings, checkpoints, and full per-request
traces are not committed because they are large or derived. The reports record
the dataset revisions, hashes, configuration, and observed environment. See
`research/results/dataset_selection.md` and
`research/results/judge_vcache_report.md` for the current protocol and results.

## Attribution and license

`src/mvr-cache/` preserves the upstream project files and license. New
research code in this repository is provided under the repository license; see
`src/mvr-cache/LICENSE` for the upstream license terms.

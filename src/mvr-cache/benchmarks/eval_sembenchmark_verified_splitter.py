"""
Offline evaluation on SemBenchmark datasets using VerifiedSplitterDecisionPolicy (no live model calls).

This mirrors `benchmarks/eval_sembenchmark_verified.py`, but the cache decision is made via:
  (query_prompt, cached_prompt) -> MaxSimSplitter -> MaxSim similarity score
and then the verified Bayesian explore/exploit logic.

Notes:
- The vector DB candidate selection uses the same BGE embedding model as the MaxSim splitter (via BGEEmbeddingEngine).
- The MaxSim splitter requires a checkpoint; pass it via `--splitter-checkpoint`.

Usage (example):
  export HF_TOKEN=hf_...                  # optional but recommended to avoid rate limits
  python benchmarks/eval_sembenchmark_verified_splitter.py \
    --dataset vCache/SemBenchmarkClassification \
    --embedding-col emb_gte \
    --llm-col response_llama_3_8b \
    --splitter-checkpoint /path/to/checkpoint_dir \
    --delta 0.02
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib
import json
import os
import random
import sys
import time
import warnings
from typing import Any, Dict, List, Tuple, Optional

# --------------------------------------------------------------------------------------
# IMPORTANT: ensure we import the *local repo* `vcache/` package, not an installed copy.
#
# When running this file directly via an absolute path, Python adds only the `benchmarks/`
# directory to sys.path, which can cause `import vcache` to resolve to a different package
# (e.g. the sibling `vcahce/` repo or a pip-installed `vcache`).
# --------------------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))  # .../vcache_Multi_HNSW
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# The multivector policy needs the local Windows build of the parent-aware hnswlib
# fork. Keep it ahead of site-packages so every `import hnswlib` resolves to it.
_MVR_HNSW_ROOT = os.path.abspath(
    os.path.join(_PROJECT_ROOT, "..", "..", "mvr_hnsw_target")
)
if os.path.isdir(_MVR_HNSW_ROOT) and _MVR_HNSW_ROOT not in sys.path:
    sys.path.insert(0, _MVR_HNSW_ROOT)
if os.path.isdir(_MVR_HNSW_ROOT):
    # Load the local fork before third-party imports can load site-packages hnswlib.
    import hnswlib

    if not hasattr(hnswlib.Index, "knn_query_skipping_duplicates_with_parent"):
        raise RuntimeError(
            "The parent-aware hnswlib fork was not loaded from "
            f"{_MVR_HNSW_ROOT}: {getattr(hnswlib, '__file__', None)}"
        )

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, DownloadConfig
from tqdm import tqdm
from benchmarks.common.comparison import answers_have_same_meaning_static


def _purge_nonlocal_vcache_modules() -> None:
    """Drop any already-loaded `vcache` modules that do not belong to this repo."""
    local_prefix = os.path.abspath(_PROJECT_ROOT) + os.sep
    for module_name, module in list(sys.modules.items()):
        if module_name != "vcache" and not module_name.startswith("vcache."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file and not os.path.abspath(module_file).startswith(local_prefix):
            del sys.modules[module_name]


_purge_nonlocal_vcache_modules()

from vcache.config import VCacheConfig
from vcache.main import VCache
from vcache.vcache_core.cache.embedding_engine.strategies.bge import (
    BGEEmbeddingEngine,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage import (
    InMemoryEmbeddingMetadataStorage,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.vector_db import (
    HNSWLibVectorDB,
    SimilarityMetricType,
)
from vcache.vcache_core.cache.eviction_policy.strategies.no_eviction import NoEvictionPolicy
from vcache.vcache_core.similarity_evaluator.strategies.benchmark_comparison import (
    BenchmarkComparisonSimilarityEvaluator,
)
from vcache.vcache_core.similarity_evaluator.strategies.string_comparison import (
    StringComparisonSimilarityEvaluator,
)
from vcache.vcache_core.splitter.embedding_model import EmbeddingModel
from vcache.vcache_core.splitter.MaxSimSplitter import MaxSimSplitter
from vcache.inference_engine.strategies.benchmark import BenchmarkInferenceEngine
from vcache.vcache_policy.strategies.verified_splitter import VerifiedSplitterDecisionPolicy

# Guard against accidentally importing the sibling `vcahce/` or `vcache_128/` splitter.
if "max_segments" not in MaxSimSplitter.__init__.__code__.co_varnames:
    _purge_nonlocal_vcache_modules()
    MaxSimSplitter = importlib.import_module(
        "vcache.vcache_core.splitter.MaxSimSplitter"
    ).MaxSimSplitter


# Ignore scikit-learn 1.8+ warning (LogisticRegression(penalty=...) deprecation) coming from verified policy internals.
warnings.filterwarnings(
    "ignore",
    message=".*'penalty' was deprecated.*",
    category=FutureWarning,
)

class TimingCollector:
    """
    Low-overhead wall-time accumulator. Designed to be called frequently.
    """

    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def add(self, name: str, dt_seconds: float) -> None:
        k = str(name)
        v = float(dt_seconds)
        self.sums[k] = self.sums.get(k, 0.0) + v
        self.counts[k] = self.counts.get(k, 0) + 1

    def get(self, name: str) -> float:
        return float(self.sums.get(str(name), 0.0))

    def total(self) -> float:
        return float(sum(self.sums.values()))


def _ensure_hf_cache_env(hf_cache_base: str | None = None) -> Dict[str, str]:
    """
    Set HF cache env vars if HF_CACHE_BASE is provided (or use /tmp).
    Returns resolved paths.
    """
    hf_cache_base = hf_cache_base or os.environ.get("HF_CACHE_BASE", "/tmp/hf")
    hf_home = os.path.join(hf_cache_base, "home")
    hf_hub_cache = os.path.join(hf_cache_base, "hub")
    hf_datasets_cache = os.path.join(hf_cache_base, "datasets")
    hf_transformers_cache = os.path.join(hf_cache_base, "transformers")

    os.makedirs(hf_home, exist_ok=True)
    os.makedirs(hf_hub_cache, exist_ok=True)
    os.makedirs(hf_datasets_cache, exist_ok=True)
    os.makedirs(hf_transformers_cache, exist_ok=True)

    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", hf_hub_cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_hub_cache)
    os.environ.setdefault("DATASETS_CACHE", hf_datasets_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_transformers_cache)

    return {
        "HF_CACHE_BASE": hf_cache_base,
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_hub_cache,
        "DATASETS_CACHE": hf_datasets_cache,
        "TRANSFORMERS_CACHE": hf_transformers_cache,
    }


def _get_id_set(row: Dict[str, Any]) -> int:
    v = row.get("id_set", -1)
    if v == -1:
        v = row.get("ID_Set", -1)
    # Fallback for custom datasets: allow using `label_id` as the benchmark id_set label.
    if v == -1:
        v = row.get("label_id", -1)
    try:
        return int(v)
    except Exception:
        return -1


def _parse_json_object(value: Any) -> Optional[dict]:
    if isinstance(value, dict):
        return value
    if value is None:
        return None
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _score_judge_output(
    *, dataset: str, oracle_response: Any, cached_response: Any
) -> dict:
    """Compare structured judge outputs using the task-specific acceptance rule."""
    oracle = _parse_json_object(oracle_response)
    cached = _parse_json_object(cached_response)
    if oracle is None or cached is None or set(oracle) != set(cached) or not oracle:
        return {
            "parse_ok": False,
            "dimension_errors": None,
            "mae": None,
            "max_abs_error": None,
            "within_one_per_dimension": False,
            "acceptable": False,
        }

    try:
        errors = {key: abs(float(cached[key]) - float(oracle[key])) for key in oracle}
    except (TypeError, ValueError):
        return {
            "parse_ok": False,
            "dimension_errors": None,
            "mae": None,
            "max_abs_error": None,
            "within_one_per_dimension": False,
            "acceptable": False,
        }

    mae = float(sum(errors.values()) / len(errors))
    max_error = float(max(errors.values()))
    if "pku" in str(dataset).lower():
        acceptable = max_error == 0.0
    else:
        # The per-request safety rule is dimension-wise.  The MAE <= 0.25
        # requirement is a configuration-level gate over all cache hits and is
        # evaluated in the summary below; applying it to every individual hit
        # would silently turn an aggregate quality target into a stricter rule.
        acceptable = max_error <= 1.0
    return {
        "parse_ok": True,
        "dimension_errors": errors,
        "mae": mae,
        "max_abs_error": max_error,
        "within_one_per_dimension": max_error <= 1.0,
        "acceptable": bool(acceptable),
    }


def _score_judge_step(
    *, dataset: str, is_hit: bool, oracle_response: Any, selected_response: Any
) -> tuple[int, int, int, int, dict]:
    comparison = _score_judge_output(
        dataset=dataset,
        oracle_response=oracle_response,
        cached_response=selected_response,
    )
    agrees = bool(comparison["acceptable"])
    if is_hit:
        return int(agrees), int(not agrees), 0, 0, comparison
    return 0, 0, int(not agrees), int(agrees), comparison


def _prompt_metadata(prompt: str, row: dict) -> dict:
    parsed = _parse_json_object(prompt) or {}
    instruction = str(row.get("instruction") or parsed.get("instruction") or "").strip()
    candidates = parsed.get("candidates")
    if not isinstance(candidates, list):
        candidates = []
    context = parsed.get("context")
    rubric = parsed.get("rubric")
    instruction_hash = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    candidate_hash = hashlib.sha256(
        json.dumps(candidates, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return {
        "instruction_group_hash": instruction_hash,
        "candidate_hash": candidate_hash,
        "prompt_hash": prompt_hash,
        "candidate_count": len(candidates),
        "context_hash": hashlib.sha256(
            json.dumps(context, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "rubric_hash": hashlib.sha256(str(rubric or "").encode("utf-8")).hexdigest(),
    }


def _group_bootstrap_ci(
    records: list[dict], *, replicates: int, seed: int
) -> dict:
    """Bootstrap request metrics by instruction group to avoid treating variants as independent."""
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(record["instruction_group_hash"]), []).append(record)
    if not groups or replicates <= 0:
        return {"groups": len(groups), "replicates": int(max(0, replicates)), "ci95": {}}

    group_values = list(groups.values())
    rng = np.random.default_rng(int(seed))
    metrics = {name: [] for name in ("hit_rate", "effective_hit_rate", "wrong_hit_rate")}
    for _ in range(int(replicates)):
        sample = rng.integers(0, len(group_values), size=len(group_values))
        selected = [record for group_index in sample for record in group_values[int(group_index)]]
        total = max(1, len(selected))
        metrics["hit_rate"].append(sum(bool(r["is_hit"]) for r in selected) / total)
        metrics["effective_hit_rate"].append(
            sum(bool(r["effective_hit"]) for r in selected) / total
        )
        metrics["wrong_hit_rate"].append(
            sum(bool(r["wrong_hit"]) for r in selected) / total
        )

    return {
        "groups": len(groups),
        "replicates": int(replicates),
        "ci95": {
            name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            for name, values in metrics.items()
        },
    }


def _segmentation_text_diagnostic(splitter, prompt: str) -> dict:
    """Decode a few exact single-text splitter outputs for trace inspection."""
    import torch
    from tensordict.tensordict import TensorDict
    from vcache.vcache_core.splitter.MaxSimEnv import get_segments_from_token_pointers

    encoded = splitter.encode_text(prompt)
    td = TensorDict(
        {
            "token_embeddings": encoded["token_emb"].unsqueeze(0),
            "attention_mask": encoded["attention_mask"].unsqueeze(0),
            "input_ids": encoded["input_ids"].unsqueeze(0),
            "length": torch.tensor([int(encoded["length"])], device=splitter.device),
        },
        batch_size=1,
    )
    with torch.inference_mode():
        out = splitter.policy.forward_single(td, decode_type="greedy", compute_reward=False)
    actions = out["actions"][0]
    if not isinstance(actions, torch.Tensor):
        actions = torch.as_tensor(actions, device=splitter.device)
    pointers = [int(value) for value in actions.tolist()]
    segments = get_segments_from_token_pointers(
        tokenizer=splitter.generator.tokenizer,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        pointers=pointers,
    )
    total_tokens = len(
        splitter.generator.tokenizer(prompt, add_special_tokens=True, truncation=False)[
            "input_ids"
        ]
    )
    return {
        "pointers": pointers,
        "segments": [str(segment).strip() for segment in segments],
        "segment_lengths_chars": [len(str(segment)) for segment in segments],
        "token_count_before_truncation": total_tokens,
        "token_count_model_input": int(encoded["length"]),
        "truncated": bool(total_tokens > 512),
    }


def _structured_prompt_diagnostic(prompt: str, *, max_chars: int = 2000) -> dict:
    """Keep bounded field-level text for a small number of trace samples."""
    parsed = _parse_json_object(prompt) or {}

    def clip(value: Any) -> str:
        text = str(value or "")
        return text if len(text) <= max_chars else text[:max_chars] + "...[truncated]"

    candidates = parsed.get("candidates")
    if not isinstance(candidates, list):
        candidates = []
    return {
        "instruction": clip(parsed.get("instruction")),
        "context": clip(json.dumps(parsed.get("context"), ensure_ascii=False, sort_keys=True)),
        "rubric": clip(parsed.get("rubric")),
        "candidates": [clip(candidate) for candidate in candidates],
        "candidate_count": len(candidates),
    }


def _score_step(
    *,
    is_cache_hit: bool,
    label_id_set: int,
    label_response: str,
    cache_response: str,
    response_metadata,
    nn_metadata,
    use_llm_judge: bool = False,
) -> Tuple[int, int, int, int]:
    if is_cache_hit:
        if label_id_set != -1:
            cache_response_correct = label_id_set == getattr(
                response_metadata, "id_set", -999999
            )
        elif use_llm_judge:
            cache_response_correct = label_response == cache_response
        else:
            cache_response_correct = answers_have_same_meaning_static(
                label_response, cache_response
            )

        if cache_response_correct:
            return 1, 0, 0, 0
        return 0, 1, 0, 0

    # cache miss
    if label_id_set != -1:
        nn_response_correct = label_id_set == getattr(nn_metadata, "id_set", -999999)
    elif use_llm_judge:
        nn_response_correct = label_response == getattr(nn_metadata, "response", "")
    else:
        nn_response_correct = answers_have_same_meaning_static(
            label_response, getattr(nn_metadata, "response", "")
        )

    if nn_response_correct:
        return 0, 0, 0, 1  # FN
    return 0, 0, 1, 0  # TN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="HF dataset id, e.g. vCache/SemBenchmarkClassification",
    )
    parser.add_argument(
        "--embedding-col",
        help="Embedding column (optional, no longer used for embeddings as BGE is live).",
    )
    parser.add_argument(
        "--llm-col",
        required=False,
        default=None,
        help=(
            "LLM response column, e.g. response_llama_3_8b. "
            "Optional when --similarity-evaluator=benchmark_id_set (label-only evaluation)."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to evaluate. If None, use all.",
    )
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument(
        "--min-observations",
        type=int,
        default=6,
        help="Minimum feedback observations before Verified can exploit (default: 6).",
    )
    parser.add_argument(
        "--replay-rounds",
        type=int,
        default=6,
        help="Replay the same ordered request stream this many times (default: 6 = first pass plus five repeats).",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=1000,
        help="Instruction-group bootstrap replicates for 95%% confidence intervals.",
    )
    parser.add_argument(
        "--trace-segment-samples",
        type=int,
        default=5,
        help="Decode splitter segment text for this many first-round requests.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Shuffle local/HF rows with this seed before replay.",
    )
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of deltas to run (overrides --delta). Example: --deltas 0.015 0.02 0.03",
    )
    parser.add_argument(
        "--similarity-evaluator",
        choices=["benchmark_id_set", "string"],
        default="benchmark_id_set",
        help="How to evaluate correctness (match benchmark.py run-combination).",
    )
    parser.add_argument("--max-capacity", type=int, default=200_000)
    parser.add_argument(
        "--splitter-checkpoint",
        required=True,
        help="MaxSimSplitter checkpoint file or directory (auto-picks latest ckpt).",
    )
    parser.add_argument(
        "--splitter-device",
        default="cpu",
        help="Device for MaxSimSplitter (e.g. cpu, cuda).",
    )
    parser.add_argument(
        "--splitter-max-segments",
        type=int,
        default=8,
        help="Max segments used by the RL splitter/pointer policy (default: 8).",
    )
    parser.add_argument(
        "--splitter-policy-mode",
        choices=["joint", "separate"],
        default="separate",
        help="Must match the trained splitter checkpoint.",
    )
    parser.add_argument(
        "--splitter-punctuation-only",
        action="store_true",
        help="Match training that restricts split points to punctuation tokens.",
    )
    parser.add_argument(
        "--splitter-overlap-tokens",
        type=int,
        default=0,
        help="Allow overlapping segments by this many tokens on each segment boundary (default: 0). "
             "Set to 1 to allow 1-token overlap.",
    )
    parser.add_argument(
        "--include-full-embedding",
        action="store_true",
        help=(
            "If set, append the full pooled embedding (no-CLS mean) as an extra row in the MaxSim tensors, "
            "in addition to segment embeddings. Default: segment embeddings only."
        ),
    )
    parser.add_argument(
        "--mix-fullcos",
        action="store_true",
        help=(
            "If set, use similarity = 0.5*(MaxSim + cosine(full_embed_no_cls)) where both terms are mapped to [0,1]. "
            "Default: MaxSim only."
        ),
    )
    parser.add_argument(
        "--candidate-selection",
        choices=["top_k", "all", "multivector_top_k"],
        default="top_k",
        help="How to pick cached prompts to score with MaxSim (fast: top_k; slow: all).",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=1,
        help="When candidate-selection=top_k, how many vector-DB neighbors to rerank. "
        "When candidate-selection=multivector_top_k, how many neighbors to retrieve PER query vector "
        "(then union parent IDs).",
    )
    parser.add_argument(
        "--candidate-ks",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of candidate-k values to run (overrides --candidate-k). Example: --candidate-ks 5 10 15",
    )
    parser.add_argument(
        "--use-cached-candidate-segments",
        action="store_true",
        help="Cache candidate MaxSim segment tensors in metadata and only segment the query at request time.",
    )
    parser.add_argument(
        "--multivector-max-elements",
        type=int,
        default=2_000_000,
        help="Capacity of the multivector HNSW index in *vectors* (not docs). Only used when candidate-selection=multivector_top_k.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.002,
        help="Optional per-step sleep to allow background updates (match benchmark.py).",
    )
    parser.add_argument(
        "--drain-updates",
        action="store_true",
        help="Wait for pending policy updates and index inserts after each request.",
    )
    parser.add_argument(
        "--hf-cache-base",
        default=os.environ.get("HF_CACHE_BASE", None),
        help="Base dir for HuggingFace caches (datasets/hub/transformers). Overrides HF_CACHE_BASE.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save per-sample results for curve plotting.",
    )
    parser.add_argument(
        "--benchmark-output-dir",
        type=str,
        default=None,
        help="If set, write benchmark.py-compatible outputs under this directory (creates vcache_splitter_*_run_* folders with results_<timestamp>.json).",
    )
    parser.add_argument(
        "--benchmark-run-index",
        type=int,
        default=1,
        help="run index used in vcache_splitter_*_run_<idx> folder naming when --benchmark-output-dir is set.",
    )
    parser.add_argument(
        "--benchmark-timestamp",
        type=str,
        default=None,
        help="Timestamp string used in results_<timestamp>.json when --benchmark-output-dir is set. Default matches benchmark.py format: YYYY-MM-DD_HH-MM",
    )
    parser.add_argument(
        "--save-cache-hit-samples",
        type=str,
        default=None,
        help=(
            "If set, write ALL cache-hit samples to this path as JSONL (one record per hit). "
            "If you pass a directory, it will create one file per (delta,candidate_k) run."
        ),
    )
    parser.add_argument(
        "--timing-output",
        type=str,
        default=None,
        help=(
            "If set, append a timing summary line to this text file after each "
            "(delta,candidate_k) run. If omitted, defaults to the same folder as "
            "--output-json or --benchmark-output-dir."
        ),
    )
    parser.add_argument(
        "--timing-cuda-sync",
        action="store_true",
        help="If set, synchronize CUDA around timed blocks for more accurate GPU timings (adds overhead).",
    )
    args = parser.parse_args()
    if args.min_observations < 2:
        parser.error("--min-observations must be at least 2 (the policy prior has two observations)")
    if args.replay_rounds < 1:
        parser.error("--replay-rounds must be at least 1")
    if args.bootstrap_replicates < 0 or args.trace_segment_samples < 0:
        parser.error("bootstrap and segment sample counts must be non-negative")

    # Mirror: must be set before importing HF libs in a fresh process.
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # Avoid flaky XetHub-backed downloads (cas-bridge.xethub.hf.co) and use standard HTTP downloads.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    # Be more tolerant of slow connections for large parquet files.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    dataset_is_local_file = False
    try:
        dataset_is_local_file = os.path.exists(str(args.dataset))
    except Exception:
        dataset_is_local_file = False
    if str(args.dataset).endswith(".csv") or str(args.dataset).endswith(".parquet"):
        dataset_is_local_file = True

    if dataset_is_local_file:
        dataset_path = os.path.abspath(str(args.dataset))
        if dataset_path.endswith(".csv"):
            try:
                df = pd.read_csv(dataset_path)
            except Exception:
                df = pd.read_csv(
                    dataset_path,
                    engine="python",
                    on_bad_lines="skip",
                    low_memory=False,
                )
        elif dataset_path.endswith(".parquet"):
            df = pd.read_parquet(dataset_path)
        else:
            raise ValueError(
                f"Unsupported local dataset file format: {dataset_path} (expected .csv or .parquet)"
            )

        if "prompt" not in df.columns:
            raise ValueError(
                f"Local dataset is missing required column 'prompt'. Available columns: {list(df.columns)}"
            )

        id_col = None
        if "ID_Set" in df.columns:
            id_col = "ID_Set"
        elif "id_set" in df.columns:
            id_col = "id_set"
        elif "label_id" in df.columns:
            # Custom dataset: treat label_id as id_set label.
            id_col = "label_id"

        if args.similarity_evaluator == "benchmark_id_set":
            has_usable_ids = False
            if id_col is not None:
                try:
                    s = pd.to_numeric(df[id_col], errors="coerce").fillna(-1)
                    has_usable_ids = bool((s.astype(int) != -1).any())
                except Exception:
                    has_usable_ids = False
            if not has_usable_ids:
                args.similarity_evaluator = "string"
        # If we fell back to string evaluation, we MUST have a response column.
        if args.similarity_evaluator == "string":
            if args.llm_col is None or str(args.llm_col) == "":
                raise ValueError(
                    "similarity-evaluator='string' requires --llm-col, but none was provided. "
                    f"Available columns: {list(df.columns)}"
                )
            if args.llm_col not in df.columns:
                raise ValueError(
                    f"Local dataset is missing required LLM response column '{args.llm_col}'. Available columns: {list(df.columns)}"
                )

        rows = df.to_dict("records")
        if args.max_samples is not None:
            rows = rows[: int(args.max_samples)]
    else:
        cache_paths = _ensure_hf_cache_env(args.hf_cache_base)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

        # Load dataset
        split = "train"
        if args.max_samples is not None:
            split = f"train[:{args.max_samples}]"

        dl_cfg = DownloadConfig(
            resume_download=True,
            max_retries=50,
        )
        rows = load_dataset(
            args.dataset,
            split=split,
            cache_dir=cache_paths["DATASETS_CACHE"],
            token=hf_token,
            download_config=dl_cfg,
        )

    if args.shuffle_seed is not None:
        order = np.random.default_rng(int(args.shuffle_seed)).permutation(len(rows))
        rows = [rows[int(i)] for i in order]

    base_rows = rows
    rows = []
    for replay_round in range(1, int(args.replay_rounds) + 1):
        for round_index, row in enumerate(base_rows, start=1):
            replay_row = dict(row)
            replay_row["_replay_round"] = replay_round
            replay_row["_round_index"] = round_index
            rows.append(replay_row)

    deltas = (
        list(args.deltas)
        if args.deltas is not None and len(args.deltas) > 0
        else [float(args.delta)]
    )
    if str(args.candidate_selection) == "all":
        candidate_ks = [-1]
    else:
        candidate_ks = (
            list(args.candidate_ks)
            if args.candidate_ks is not None and len(args.candidate_ks) > 0
            else [int(args.candidate_k)]
        )
    run_grid = [(float(d), int(k)) for d in deltas for k in candidate_ks]

    shared_embedder = EmbeddingModel(device=args.splitter_device)
    embedding_engine = BGEEmbeddingEngine(embedding_model=shared_embedder)
    splitter = MaxSimSplitter(
        checkpoint_path=args.splitter_checkpoint,
        device=args.splitter_device,
        embedding_model=shared_embedder,
        max_segments=int(args.splitter_max_segments),
        overlap_tokens=int(args.splitter_overlap_tokens),
        include_full_embedding=bool(args.include_full_embedding),
        policy_mode=str(args.splitter_policy_mode),
        split_on_connectors=not bool(args.splitter_punctuation_only),
    )

    if args.similarity_evaluator == "string":
        similarity_evaluator = StringComparisonSimilarityEvaluator()
    else:
        similarity_evaluator = BenchmarkComparisonSimilarityEvaluator()

    def _resolve_output_path(*, delta: float, candidate_k: int) -> str | None:
        if not args.output_json:
            return None
        base = str(args.output_json)
        if len(run_grid) <= 1:
            return base

        base_abs = os.path.abspath(base)
        if base_abs.endswith(os.sep) or os.path.isdir(base_abs):
            os.makedirs(base_abs, exist_ok=True)
            name = f"verified_splitter_delta{delta}_k{candidate_k}.json"
            return os.path.join(base_abs, name)

        root, ext = os.path.splitext(base)
        ext = ext if ext else ".json"
        return f"{root}_delta{delta}_k{candidate_k}{ext}"

    benchmark_timestamp = args.benchmark_timestamp or datetime.now().strftime(
        "%Y-%m-%d_%H-%M"
    )

    def _resolve_benchmark_output_dir(*, delta: float, candidate_k: int) -> str | None:
        if not args.benchmark_output_dir:
            return None

        base_dir = os.path.abspath(str(args.benchmark_output_dir))
        os.makedirs(base_dir, exist_ok=True)

        candidate_k_label = "all" if int(candidate_k) == -1 else str(int(candidate_k))
        sel_for_dir = "all" if str(args.candidate_selection) == "all" else "top_k"
        dir_name = f"vcache_splitter_{delta}_{sel_for_dir}_{candidate_k_label}_run_{int(args.benchmark_run_index)}"
        out_dir = os.path.join(base_dir, dir_name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _resolve_timing_output_path(*, delta: float, candidate_k: int) -> str | None:
        candidate_k_label = "all" if int(candidate_k) == -1 else str(int(candidate_k))

        if args.timing_output:
            base = os.path.abspath(str(args.timing_output))
            if base.endswith(os.sep) or os.path.isdir(base):
                os.makedirs(base, exist_ok=True)
                return os.path.join(
                    base, f"timing_verified_splitter_delta{delta}_k{candidate_k_label}.txt"
                )
            return base

        bench_dir = _resolve_benchmark_output_dir(delta=float(delta), candidate_k=int(candidate_k))
        if bench_dir:
            return os.path.join(
                bench_dir, f"timing_verified_splitter_delta{delta}_k{candidate_k_label}.txt"
            )

        output_path = _resolve_output_path(delta=float(delta), candidate_k=int(candidate_k))
        if output_path:
            output_dir = os.path.dirname(os.path.abspath(output_path))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                return os.path.join(
                    output_dir,
                    f"timing_verified_splitter_delta{delta}_k{candidate_k_label}.txt",
                )

        return os.path.abspath("timing_verified_splitter.txt")

    def _resolve_hit_samples_path(*, delta: float, candidate_k: int) -> str | None:
        """
        Resolve output path for cache-hit sample dump.
        Mirrors _resolve_output_path behavior:
        - If base is a directory, create per-run filename inside it.
        - If base is a file path and multiple runs, suffix it with delta/k.
        """
        if not args.save_cache_hit_samples:
            return None
        base = str(args.save_cache_hit_samples)
        if len(run_grid) <= 1:
            return base

        base_abs = os.path.abspath(base)
        if base_abs.endswith(os.sep) or os.path.isdir(base_abs):
            os.makedirs(base_abs, exist_ok=True)
            name = f"cache_hit_samples_delta{delta}_k{candidate_k}.jsonl"
            return os.path.join(base_abs, name)

        root, ext = os.path.splitext(base)
        ext = ext if ext else ".jsonl"
        # Ensure jsonl extension unless user explicitly gave something else
        if ext.lower() == ".json":
            ext = ".jsonl"
        return f"{root}_delta{delta}_k{candidate_k}{ext}"

    def _compute_statistics_json(
        *,
        cache_hit_list: list[int],
        cache_miss_list: list[int],
        tp_list: list[int],
        fp_list: list[int],
        tn_list: list[int],
        fn_list: list[int],
        latency_direct_list: list[float | None],
        latency_vectorq_list: list[float],
    ) -> Dict[str, Any]:
        n = int(len(latency_vectorq_list))
        avg_latency_vcache_overall = float(sum(latency_vectorq_list) / n) if n > 0 else 0.0
        direct_available = len(latency_direct_list) == n and all(
            value is not None for value in latency_direct_list
        )
        avg_latency_direct_overall = (
            float(sum(float(value) for value in latency_direct_list) / n)
            if direct_available and n > 0
            else None
        )

        hit_latencies_v = [latency_vectorq_list[i] for i in range(n) if int(cache_hit_list[i]) > 0]
        miss_latencies_v = [latency_vectorq_list[i] for i in range(n) if int(cache_miss_list[i]) > 0]
        hit_latencies_d = [
            float(latency_direct_list[i])
            for i in range(n)
            if direct_available and int(cache_hit_list[i]) > 0
        ]
        miss_latencies_d = [
            float(latency_direct_list[i])
            for i in range(n)
            if direct_available and int(cache_miss_list[i]) > 0
        ]

        avg_latency_vcache_cache_hit = (
            float(sum(hit_latencies_v) / len(hit_latencies_v)) if hit_latencies_v else 0.0
        )
        avg_latency_vcache_cache_miss = (
            float(sum(miss_latencies_v) / len(miss_latencies_v)) if miss_latencies_v else 0.0
        )
        avg_latency_direct_cache_hit = (
            float(sum(hit_latencies_d) / len(hit_latencies_d)) if hit_latencies_d else 0.0
        )
        avg_latency_direct_cache_miss = (
            float(sum(miss_latencies_d) / len(miss_latencies_d)) if miss_latencies_d else 0.0
        )

        cache_hit_rate_vcache = float(sum(cache_hit_list) / n) if n > 0 else 0.0
        cache_miss_rate_vcache = 1.0 - cache_hit_rate_vcache
        error_rate_vcache = float(sum(fp_list) / n) if n > 0 else 0.0

        duration_vcache = float(sum(latency_vectorq_list))
        duration_direct = (
            float(sum(float(value) for value in latency_direct_list))
            if direct_available
            else None
        )

        tp_sum = int(sum(tp_list))
        fp_sum = int(sum(fp_list))
        tn_sum = int(sum(tn_list))
        fn_sum = int(sum(fn_list))

        accuracy_vcache = float((tp_sum + tn_sum) / n) if n > 0 else 0.0
        precision_vcache = (
            float(tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else 0.0
        )
        recall_vcache = float(tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else 0.0
        f1_score_vcache = (
            float(2 * precision_vcache * recall_vcache / (precision_vcache + recall_vcache))
            if (precision_vcache + recall_vcache) > 0
            else 0.0
        )

        hits_last = int(cache_hit_list[-1]) if cache_hit_list else 0
        misses_last = int(cache_miss_list[-1]) if cache_miss_list else 0

        return {
            "avg_latency": {
                "cache": {
                    "overall": float(avg_latency_vcache_overall),
                    "cache_hit": float(avg_latency_vcache_cache_hit),
                    "cache_miss": float(avg_latency_vcache_cache_miss),
                },
                "direct": {
                    "overall": avg_latency_direct_overall,
                    "available": bool(direct_available),
                },
                "difference": {
                    "overall": (
                        float(avg_latency_direct_overall - avg_latency_vcache_overall)
                        if direct_available
                        else None
                    ),
                    "cache_hit": (
                        float(avg_latency_direct_cache_hit - avg_latency_vcache_cache_hit)
                        if direct_available
                        else None
                    ),
                    "cache_miss": (
                        float(avg_latency_direct_cache_miss - avg_latency_vcache_cache_miss)
                        if direct_available
                        else None
                    ),
                },
                "ratio": {
                    "overall": float(avg_latency_direct_overall / avg_latency_vcache_overall)
                    if direct_available and avg_latency_vcache_overall > 0
                    else "N/A",
                    "cache_hit": float(avg_latency_direct_cache_hit / avg_latency_vcache_cache_hit)
                    if direct_available and avg_latency_vcache_cache_hit > 0
                    else "N/A",
                    "cache_miss": float(avg_latency_direct_cache_miss / avg_latency_vcache_cache_miss)
                    if direct_available and avg_latency_vcache_cache_miss > 0
                    else "N/A",
                },
            },
            "cache": {
                "hit_rate": float(cache_hit_rate_vcache),
                "miss_rate": float(cache_miss_rate_vcache),
                "total_samples": int(n),
                "hits": int(hits_last),
                "misses": int(misses_last),
                "error_rate": float(error_rate_vcache),
            },
            "duration": {
                "vectorq": float(duration_vcache),
                "direct": float(duration_direct),
            },
            "statistics": {
                "accuracy": float(accuracy_vcache),
                "precision": float(precision_vcache),
                "recall": float(recall_vcache),
                "f1_score": float(f1_score_vcache),
            },
        }

    all_summaries: list[dict] = []

    for run_i, (delta, candidate_k) in enumerate(run_grid, start=1):
        random.seed(int(args.seed))
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

        inference_engine = BenchmarkInferenceEngine()
        config = VCacheConfig(
            embedding_engine=embedding_engine,
            inference_engine=inference_engine,
            vector_db=HNSWLibVectorDB(
                similarity_metric_type=SimilarityMetricType.COSINE,
                max_capacity=int(args.max_capacity),
            ),
            embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
            eviction_policy=NoEvictionPolicy(),
            similarity_evaluator=similarity_evaluator,
        )

        timing: Optional[TimingCollector] = TimingCollector() if args.timing_output else None

        policy = VerifiedSplitterDecisionPolicy(
            delta=float(delta),
            min_observations=int(args.min_observations),
            splitter=splitter,
            device=args.splitter_device,
            candidate_selection=args.candidate_selection,
            candidate_k=int(candidate_k),
            use_cached_candidate_segments=bool(args.use_cached_candidate_segments),
            multivector_max_elements=int(args.multivector_max_elements),
            mix_fullcos=bool(args.mix_fullcos),
            timing_collector=timing,
            timing_cuda_sync=bool(args.timing_cuda_sync),
        )
        vcache = VCache(config=config, policy=policy)

        hits = 0
        tp = fp = tn = fn = 0
        n = 0
        per_sample_results: list[dict] = []
        request_records: list[dict] = []
        prompt_token_cache: dict[str, int] = {}
        seen_prompt_hashes: set[str] = set()
        seen_instruction_candidates: dict[str, set[str]] = {}
        round_metrics: dict[int, dict[str, int]] = {}
        max_abs_errors: list[float] = []
        maes: list[float] = []

        cache_hit_list: list[int] = []
        cache_miss_list: list[int] = []
        tp_list: list[int] = []
        fp_list: list[int] = []
        tn_list: list[int] = []
        fn_list: list[int] = []
        latency_direct_list: list[float] = []
        latency_vectorq_list: list[float] = []

        hit_samples_path = _resolve_hit_samples_path(delta=float(delta), candidate_k=int(candidate_k))
        hit_samples_f = None
        if hit_samples_path:
            hit_dir = os.path.dirname(os.path.abspath(hit_samples_path))
            if hit_dir:
                os.makedirs(hit_dir, exist_ok=True)
            hit_samples_f = open(hit_samples_path, "w", encoding="utf-8")

        t0 = time.time()
        desc_base = f"Evaluating (Splitter) run={run_i}/{len(run_grid)} delta={delta} k={candidate_k}"
        pbar = tqdm(rows, desc=desc_base, unit="samples")
        try:
            for r in pbar:
                prompt = r["prompt"]
                system_prompt = r.get("output_format", "") or ""
                id_set = _get_id_set(r)
                replay_round = int(r.get("_replay_round", 1))
                round_index = int(r.get("_round_index", n + 1))
                request_meta = _prompt_metadata(prompt, r)
                if request_meta["prompt_hash"] in seen_prompt_hashes:
                    request_relation = "exact_repeat"
                elif request_meta["instruction_group_hash"] in seen_instruction_candidates:
                    seen_candidates = seen_instruction_candidates[
                        request_meta["instruction_group_hash"]
                    ]
                    request_relation = (
                        "same_instruction_different_candidate"
                        if request_meta["candidate_hash"] not in seen_candidates
                        else "same_instruction_context_or_rubric_variant"
                    )
                else:
                    request_relation = "new_instruction"
                seen_prompt_hashes.add(request_meta["prompt_hash"])
                seen_instruction_candidates.setdefault(
                    request_meta["instruction_group_hash"], set()
                ).add(request_meta["candidate_hash"])

                if request_meta["prompt_hash"] not in prompt_token_cache:
                    tokenizer = splitter.generator.tokenizer
                    prompt_token_cache[request_meta["prompt_hash"]] = len(
                        tokenizer(prompt, add_special_tokens=True, truncation=False)[
                            "input_ids"
                        ]
                    )
                prompt_token_count = prompt_token_cache[request_meta["prompt_hash"]]

                # Only require/consume response strings when using string-based correctness.
                if args.llm_col and args.llm_col in r:
                    label_response = str(r[args.llm_col])
                else:
                    label_response = str(r.get("response", "") or "")
                # BenchmarkInferenceEngine expects set_next_response() to set the attribute.
                inference_engine.set_next_response(label_response)

                step_t0 = time.time()
                is_hit, resp, resp_meta, nn_meta = vcache.infer_with_cache_info(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    id_set=id_set,
                )
                step_latency = time.time() - step_t0
                if args.drain_updates:
                    policy.drain_pending_updates()

                n += 1
                hits += int(is_hit)
                has_nearest_candidate = int(getattr(nn_meta, "embedding_id", -1)) >= 0
                if is_hit or has_nearest_candidate:
                    selected_for_comparison = (
                        resp if is_hit else getattr(nn_meta, "response", "")
                    )
                    d_tp, d_fp, d_tn, d_fn, output_score = _score_judge_step(
                        dataset=args.dataset,
                        is_hit=is_hit,
                        oracle_response=label_response,
                        selected_response=selected_for_comparison,
                    )
                else:
                    d_tp, d_fp, d_tn, d_fn = 0, 0, 1, 0
                    output_score = {
                        "comparison": "no_candidate",
                        "parse_ok": None,
                        "dimension_errors": None,
                        "mae": None,
                        "max_abs_error": None,
                        "within_one_per_dimension": None,
                        "acceptable": False,
                    }
                effective_hit = bool(is_hit and output_score["acceptable"])
                wrong_hit = bool(is_hit and not output_score["acceptable"])
                policy_trace = dict(policy.last_decision_info)
                scored_candidates = policy_trace.get("scored_candidates") or []
                candidate_label_ids = {
                    int(item["id_set"])
                    for item in scored_candidates
                    if item.get("score") is not None and item.get("id_set") is not None
                }
                candidate_set_contains_correct = (
                    bool(id_set != -1 and id_set in candidate_label_ids)
                    if scored_candidates
                    else None
                )
                top_candidate_correct = (
                    bool(
                        id_set != -1
                        and policy_trace.get("top_candidate_id_set") == id_set
                    )
                    if policy_trace.get("top_candidate_id_set") is not None
                    else None
                )
                policy_trace["candidate_set_contains_correct_label"] = candidate_set_contains_correct
                policy_trace["top_candidate_correct_label"] = top_candidate_correct
                if output_score["mae"] is not None and is_hit:
                    maes.append(float(output_score["mae"]))
                    max_abs_errors.append(float(output_score["max_abs_error"]))
                tp += d_tp
                fp += d_fp
                tn += d_tn
                fn += d_fn

                cache_hit_list.append(int(bool(is_hit)))
                cache_miss_list.append(int(not bool(is_hit)))
                tp_list.append(int(d_tp))
                fp_list.append(int(d_fp))
                tn_list.append(int(d_tn))
                fn_list.append(int(d_fn))
                # A direct Judge call is not executed in this offline replay.
                # Keep the cache latency for profiling, but leave the direct
                # baseline explicitly unavailable instead of copying the same
                # value into both series.
                latency_direct_list.append(None)
                latency_vectorq_list.append(float(step_latency))

                per_sample_results.append(
                    {
                        "sample_index": n,
                        "replay_round": replay_round,
                        "round_index": round_index,
                        "source_index": r.get("source_index"),
                        "instruction_group_hash": request_meta["instruction_group_hash"],
                        "prompt_hash": request_meta["prompt_hash"],
                        "candidate_hash": request_meta["candidate_hash"],
                        "context_hash": request_meta["context_hash"],
                        "rubric_hash": request_meta["rubric_hash"],
                        "request_relation": request_relation,
                        "prompt_char_count": len(prompt),
                        "prompt_token_count_before_truncation": prompt_token_count,
                        "prompt_token_count_model_input": min(prompt_token_count, 512),
                        "prompt_truncated": bool(prompt_token_count > 512),
                        "candidate_count": request_meta["candidate_count"],
                        "is_hit": is_hit,
                        "effective_hit": effective_hit,
                        "wrong_hit": wrong_hit,
                        "judge_output_error": output_score,
                        "policy_trace": policy_trace,
                        "candidate_set_contains_correct_label": candidate_set_contains_correct,
                        "top_candidate_correct_label": top_candidate_correct,
                        "running_hit_rate": hits / n,
                        "tp": d_tp,
                        "fp": d_fp,
                        "tn": d_tn,
                        "fn": d_fn,
                    }
                )
                request_record = {
                    "instruction_group_hash": request_meta["instruction_group_hash"],
                    "is_hit": bool(is_hit),
                    "effective_hit": effective_hit,
                    "wrong_hit": wrong_hit,
                }
                request_records.append(request_record)
                rm = round_metrics.setdefault(
                    replay_round,
                    {"requests": 0, "hits": 0, "effective_hits": 0, "wrong_hits": 0},
                )
                rm["requests"] += 1
                rm["hits"] += int(is_hit)
                rm["effective_hits"] += int(effective_hit)
                rm["wrong_hits"] += int(wrong_hit)

                if (
                    replay_round == 1
                    and round_index <= int(args.trace_segment_samples)
                ):
                    segmentation_diagnostic = _segmentation_text_diagnostic(splitter, prompt)
                    segmentation_diagnostic["structured_fields"] = _structured_prompt_diagnostic(
                        prompt
                    )
                    per_sample_results[-1]["segmentation_diagnostic"] = segmentation_diagnostic

                # Optional: dump every cache hit sample to JSONL for later inspection.
                if bool(is_hit) and hit_samples_f is not None:
                    rec = {
                        "sample_index": int(n),
                        "replay_round": replay_round,
                        "delta": float(delta),
                        "candidate_k": int(candidate_k),
                        "candidate_selection": str(args.candidate_selection),
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "label_id_set": int(id_set),
                        "label_response": label_response,
                        "cached_embedding_id": int(getattr(resp_meta, "embedding_id", -1)),
                        "cached_id_set": int(getattr(resp_meta, "id_set", -1)),
                        "cached_prompt": getattr(resp_meta, "prompt", "") or "",
                        "cached_response": resp,
                        "judge_output_error": output_score,
                        # Useful policy state (if present)
                        "t_hat": getattr(resp_meta, "t_hat", None),
                        "t_prime": getattr(resp_meta, "t_prime", None),
                        "gamma": getattr(resp_meta, "gamma", None),
                        "var_t": getattr(resp_meta, "var_t", None),
                    }
                    hit_samples_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # Put hit stats in the *description* so it doesn't get truncated off the right side.
                # (tqdm's postfix is often cut in narrow terminals.)
                try:
                    pbar.set_description(f"{desc_base} hits={hits}/{n} ({(hits/n):.1%})")
                except Exception:
                    pass

                if args.sleep and args.sleep > 0:
                    time.sleep(float(args.sleep))
        finally:
            if hit_samples_f is not None:
                try:
                    hit_samples_f.close()
                    print(f"Cache-hit samples saved to {hit_samples_path}")
                except Exception:
                    pass

        elapsed = time.time() - t0

        print(f"dataset={args.dataset}")
        print(f"columns: embedding={args.embedding_col} llm={args.llm_col}")
        print(f"delta={delta} n={n} time={elapsed:.2f}s")
        print(f"candidate_selection={args.candidate_selection} candidate_k={candidate_k}")
        print(f"hit_rate={hits}/{n} ({(hits/max(1,n)):.1%})")
        print(f"tp={tp} fp={fp} tn={tn} fn={fn}")

        raw_hit_count = sum(bool(record["is_hit"]) for record in request_records)
        effective_hit_count = sum(bool(record["effective_hit"]) for record in request_records)
        wrong_hit_count = sum(bool(record["wrong_hit"]) for record in request_records)
        exact_hit_errors = [
            record["judge_output_error"]
            for record in per_sample_results
            if record["is_hit"] and record["judge_output_error"]["mae"] is not None
        ]
        hit_parse_failures = sum(
            bool(record["is_hit"] and not record["judge_output_error"].get("parse_ok"))
            for record in per_sample_results
        )
        hit_dimension_violations = sum(
            bool(
                record["is_hit"]
                and record["judge_output_error"].get("max_abs_error") is not None
                and float(record["judge_output_error"]["max_abs_error"]) > 1.0
            )
            for record in per_sample_results
        )
        candidate_recall_records = [
            record
            for record in per_sample_results
            if record.get("candidate_set_contains_correct_label") is not None
        ]
        top_candidate_records = [
            record
            for record in per_sample_results
            if record.get("top_candidate_correct_label") is not None
        ]

        def _stratify(key_fn):
            buckets: dict[str, list[dict]] = {}
            for record in per_sample_results:
                buckets.setdefault(str(key_fn(record)), []).append(record)
            return {
                key: {
                    "requests": len(bucket),
                    "hit_rate": sum(bool(x["is_hit"]) for x in bucket) / len(bucket),
                    "effective_hit_rate": sum(bool(x["effective_hit"]) for x in bucket) / len(bucket),
                    "wrong_hit_rate": sum(bool(x["wrong_hit"]) for x in bucket) / len(bucket),
                }
                for key, bucket in buckets.items()
            }

        for round_result in round_metrics.values():
            denominator = max(1, round_result["requests"])
            round_result["hit_rate"] = round_result["hits"] / denominator
            round_result["effective_hit_rate"] = round_result["effective_hits"] / denominator
            round_result["wrong_hit_rate"] = round_result["wrong_hits"] / denominator

        bootstrap_per_round = {}
        for replay_round in sorted(round_metrics):
            round_records = [
                {
                    **request_records[i],
                    "is_hit": bool(per_sample_results[i]["is_hit"]),
                    "effective_hit": bool(per_sample_results[i]["effective_hit"]),
                    "wrong_hit": bool(per_sample_results[i]["wrong_hit"]),
                }
                for i in range(len(request_records))
                if int(per_sample_results[i]["replay_round"]) == replay_round
            ]
            bootstrap_per_round[str(replay_round)] = _group_bootstrap_ci(
                round_records,
                replicates=int(args.bootstrap_replicates),
                seed=int(args.seed) + replay_round,
            )

        summary = {
            "delta": float(delta),
            "min_observations": int(args.min_observations),
            "candidate_k": int(candidate_k),
            "candidate_selection": str(args.candidate_selection),
            "n": n,
            "hits": hits,
            "hit_rate": hits / max(1, n),
            "effective_hits": effective_hit_count,
            "effective_hit_rate": effective_hit_count / max(1, n),
            "wrong_hits": wrong_hit_count,
            "wrong_hit_rate": wrong_hit_count / max(1, n),
            "hit_label_agreement_rate": (
                (raw_hit_count - fp) / raw_hit_count if raw_hit_count else None
            ),
            "accepted_error_threshold": (
                {"max_abs_error_per_dimension": 0.0, "mae": 0.0}
                if "pku" in os.path.basename(str(args.dataset)).lower()
                else {"max_abs_error_per_dimension": 1.0, "mae": 0.25}
            ),
            "hit_mae": float(np.mean(maes)) if maes else None,
            "hit_parse_failures": int(hit_parse_failures),
            "hit_parse_failure_rate": (
                hit_parse_failures / raw_hit_count if raw_hit_count else None
            ),
            "hit_dimension_violations": int(hit_dimension_violations),
            "hit_dimension_violation_rate": (
                hit_dimension_violations / raw_hit_count if raw_hit_count else None
            ),
            "candidate_set_recall": {
                "evaluable_requests": len(candidate_recall_records),
                "contains_correct_label": sum(
                    bool(record["candidate_set_contains_correct_label"])
                    for record in candidate_recall_records
                ),
                "contains_correct_label_rate": (
                    sum(bool(record["candidate_set_contains_correct_label"])
                        for record in candidate_recall_records)
                    / len(candidate_recall_records)
                    if candidate_recall_records
                    else None
                ),
                "top_candidate_correct": sum(
                    bool(record["top_candidate_correct_label"])
                    for record in top_candidate_records
                ),
                "top_candidate_correct_rate": (
                    sum(bool(record["top_candidate_correct_label"])
                        for record in top_candidate_records)
                    / len(top_candidate_records)
                    if top_candidate_records
                    else None
                ),
            },
            "quality_gate": {
                "dimension_rule": (
                    "exact_match" if "pku" in os.path.basename(str(args.dataset)).lower()
                    else "max_abs_error_per_dimension<=1"
                ),
                "aggregate_hit_mae_limit": (
                    0.0 if "pku" in os.path.basename(str(args.dataset)).lower() else 0.25
                ),
                "aggregate_hit_mae_pass": (
                    bool(
                        hit_parse_failures == 0
                        and hit_dimension_violations == 0
                        and (not maes or float(np.mean(maes)) == 0.0)
                    )
                    if "pku" in os.path.basename(str(args.dataset)).lower()
                    else bool(
                        hit_parse_failures == 0
                        and hit_dimension_violations == 0
                        and (not maes or float(np.mean(maes)) <= 0.25)
                    )
                ),
                "passed": bool(
                    (
                        hit_parse_failures == 0
                        and hit_dimension_violations == 0
                        and (not maes or float(np.mean(maes)) == 0.0)
                    )
                    if "pku" in os.path.basename(str(args.dataset)).lower()
                    else (
                        hit_parse_failures == 0
                        and hit_dimension_violations == 0
                        and (not maes or float(np.mean(maes)) <= 0.25)
                    )
                ),
            },
            "hit_max_abs_error": float(max(max_abs_errors)) if max_abs_errors else None,
            "hit_within_one_per_dimension_rate": (
                sum(x["max_abs_error"] <= 1.0 for x in exact_hit_errors) / len(exact_hit_errors)
                if exact_hit_errors
                else None
            ),
            "instruction_group_bootstrap": _group_bootstrap_ci(
                [
                    {
                        **request_records[i],
                        "is_hit": bool(per_sample_results[i]["is_hit"]),
                        "effective_hit": bool(per_sample_results[i]["effective_hit"]),
                        "wrong_hit": bool(per_sample_results[i]["wrong_hit"]),
                    }
                    for i in range(len(request_records))
                ],
                replicates=int(args.bootstrap_replicates),
                seed=int(args.seed),
            ),
            "instruction_group_bootstrap_per_round": bootstrap_per_round,
            "per_round": {str(k): v for k, v in sorted(round_metrics.items())},
            "by_request_relation": _stratify(lambda r: r["request_relation"]),
            "by_truncation": _stratify(lambda r: "truncated" if r["prompt_truncated"] else "within_512"),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "total_time": elapsed,
        }
        all_summaries.append(summary)

        out_path = _resolve_output_path(delta=float(delta), candidate_k=int(candidate_k))
        if out_path:
            out_dir = os.path.dirname(os.path.abspath(out_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            output_data = {"args": vars(args), "summary": summary, "per_sample": per_sample_results}
            with open(out_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"Results saved to {out_path}")

        # ----------------------------------------------------------------------------------
        # Timing summary (low overhead): write totals to a root-folder text file.
        # We split time into:
        #   - retrieval: vector DB HNSW KNN (and/or multivector HNSW) candidate selection
        #   - rl: splitter model + MaxSim scoring (including query encoding / tensor building)
        #   - rest: everything else in the end-to-end request path
        # Total is the sum of per-sample wall times around `vcache.infer_with_cache_info`.
        # ----------------------------------------------------------------------------------
        if args.timing_output:
            # Measure ONLY:
            # - embedding time (query encoding)
            # - retrieval time (HNSW / multivector HNSW candidate selection)
            # - RL time (splitter tensor building)
            # - MaxSim scoring time (separate)
            embedding_s = 0.0
            retrieval_s = 0.0
            rl_s = 0.0
            maxsim_s = 0.0
            if timing is not None:
                embedding_s = float(timing.get("splitter.encode_text"))
                retrieval_s = float(
                    timing.get("candidate_selection.hnsw_top_k")
                    + timing.get("candidate_selection.multivector_hnsw")
                    + timing.get("candidate_selection.all_scan")
                )
                # Splitter/RL compute (excluding MaxSim scoring itself)
                rl_s = float(
                    timing.get("splitter.query_tensor_build")
                    + timing.get("splitter.query_multivector_build")
                    + timing.get("splitter.candidate_tensor_build")
                )
                maxsim_s = float(
                    timing.get("maxsim.score_pair") + timing.get("maxsim.score_tensors")
                )

            timing_path = _resolve_timing_output_path(
                delta=float(delta), candidate_k=int(candidate_k)
            )
            if timing_path is None:
                continue
            timing_dir = os.path.dirname(timing_path)
            if timing_dir:
                os.makedirs(timing_dir, exist_ok=True)
            with open(timing_path, "a", encoding="utf-8") as f:
                f.write(
                    "dataset={dataset} delta={delta} max_segments={ms} candidate_selection={sel} candidate_k={k} "
                    "n={n} embedding_s={emb:.6f} retrieval_s={ret:.6f} rl_s={rl:.6f} maxsim_s={msim:.6f}\n".format(
                        dataset=str(args.dataset),
                        delta=float(delta),
                        ms=int(args.splitter_max_segments),
                        sel=str(args.candidate_selection),
                        k=int(candidate_k),
                        n=int(n),
                        emb=embedding_s,
                        ret=retrieval_s,
                        rl=rl_s,
                        msim=maxsim_s,
                    )
                )

        bench_dir = _resolve_benchmark_output_dir(delta=float(delta), candidate_k=int(candidate_k))
        if bench_dir:
            observations_dict: Dict[str, Dict[str, float]] = {}
            gammas_dict: Dict[str, float] = {}
            t_hats_dict: Dict[str, float] = {}
            t_primes_dict: Dict[str, float] = {}
            var_ts_dict: Dict[str, float] = {}

            try:
                metadata_objects: List[EmbeddingMetadataObj] = (
                    vcache.vcache_config.embedding_metadata_storage.get_all_embedding_metadata_objects()
                )
            except Exception:
                metadata_objects = []

            for metadata_object in metadata_objects:
                try:
                    embedding_id = str(metadata_object.embedding_id)
                except Exception:
                    continue
                observations_dict[embedding_id] = getattr(metadata_object, "observations", {})
                gammas_dict[embedding_id] = getattr(metadata_object, "gamma", None)
                t_hats_dict[embedding_id] = getattr(metadata_object, "t_hat", None)
                t_primes_dict[embedding_id] = getattr(metadata_object, "t_prime", None)
                var_ts_dict[embedding_id] = getattr(metadata_object, "var_t", None)

            try:
                global_observations_dict = vcache.vcache_policy.global_observations
                global_gamma = vcache.vcache_policy.bayesian.global_gamma
                global_t_hat = vcache.vcache_policy.bayesian.global_t_hat
                global_t_prime = vcache.vcache_policy.bayesian.global_t_prime
                global_var_t = vcache.vcache_policy.bayesian.global_var_t
            except Exception:
                global_observations_dict = {}
                global_gamma = None
                global_t_hat = None
                global_t_prime = None
                global_var_t = None

            candidate_k_label = "all" if int(candidate_k) == -1 else str(int(candidate_k))
            bench_data: Dict[str, Any] = {
                "config": {
                    "filepath": str(args.dataset),
                    "embedding_model": str(args.embedding_col or ""),
                    "llm_model": str(args.llm_col),
                    "eviction_policy": str(NoEvictionPolicy()),
                    "is_static_threshold": False,
                    "threshold": None,
                    "delta": float(delta),
                    "splitter_candidate_selection": str(args.candidate_selection),
                    "splitter_candidate_k": (None if int(candidate_k) == -1 else int(candidate_k)),
                    "splitter_use_cached_candidate_segments": bool(args.use_cached_candidate_segments),
                    "splitter_device": str(args.splitter_device),
                    "splitter_checkpoint": str(args.splitter_checkpoint),
                },
                "cache_hit_list": cache_hit_list,
                "cache_miss_list": cache_miss_list,
                "tp_list": tp_list,
                "fp_list": fp_list,
                "tn_list": tn_list,
                "fn_list": fn_list,
                "latency_direct_list": latency_direct_list,
                "latency_vectorq_list": latency_vectorq_list,
                "observations_dict": observations_dict,
                "gammas_dict": gammas_dict,
                "t_hats_dict": t_hats_dict,
                "t_primes_dict": t_primes_dict,
                "var_ts_dict": var_ts_dict,
                "global_observations_dict": global_observations_dict,
                "global_gamma": global_gamma,
                "global_t_hat": global_t_hat,
                "global_t_prime": global_t_prime,
                "global_var_t": global_var_t,
            }

            bench_path = os.path.join(bench_dir, f"results_{benchmark_timestamp}.json")
            with open(bench_path, "w") as f:
                json.dump(bench_data, f, indent=4)
            print(f"Benchmark-format results saved to {bench_path}")

            statistics_path = os.path.join(
                bench_dir, f"statistics_{benchmark_timestamp}.json"
            )
            statistics_data = _compute_statistics_json(
                cache_hit_list=cache_hit_list,
                cache_miss_list=cache_miss_list,
                tp_list=tp_list,
                fp_list=fp_list,
                tn_list=tn_list,
                fn_list=fn_list,
                latency_direct_list=latency_direct_list,
                latency_vectorq_list=latency_vectorq_list,
            )
            with open(statistics_path, "w") as f:
                json.dump(statistics_data, f, indent=4)
            print(f"Benchmark-format statistics saved to {statistics_path}")

        time.sleep(0.1)
        vcache.vcache_policy.shutdown()

    if len(all_summaries) > 1:
        print("\nAll runs summary:")
        for s in all_summaries:
            print(
                f"delta={s['delta']} k={s['candidate_k']} hit_rate={s['hits']}/{s['n']} ({s['hit_rate']:.1%}) "
                f"tp={s['tp']} fp={s['fp']} tn={s['tn']} fn={s['fn']} time={s['total_time']:.2f}s"
            )


if __name__ == "__main__":
    main()


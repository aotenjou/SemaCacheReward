#!/usr/bin/env python3
"""Train/validation/test replay for the framework-agnostic Judge cache adapter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from collections import defaultdict
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "src" / "mvr-cache"
sys.path.insert(0, str(PROJECT))

from vcache.judge_cache import (  # noqa: E402
    HELPSTEER2_REWARD,
    PKU_PREFERENCE,
    ULTRAFEEDBACK_REWARD,
    JudgeCache,
    JudgeRequest,
    JudgeResult,
    RiskModel,
    _field_vectors,
    _similarity_features,
)

SPLIT_DIR = ROOT / "work" / "mvr_eval_data"
PAIR_DIR = ROOT / "work" / "mvr_training_data"
OUTPUT_DIR = ROOT / "research" / "results"
SCHEMAS = {
    "pku": PKU_PREFERENCE,
    "helpsteer2": HELPSTEER2_REWARD,
    "ultrafeedback": ULTRAFEEDBACK_REWARD,
}


def _json(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _request(prompt: str, schema, dataset: str) -> JudgeRequest:
    payload = json.loads(prompt)
    return JudgeRequest(
        instruction=str(payload["instruction"]),
        context=payload.get("context") or {},
        rubric=str(payload.get("rubric") or ""),
        candidates=tuple(str(x) for x in payload["candidates"]),
        reward_schema=schema,
        judge_model=f"offline-oracle:{dataset}",
        prompt_version=str(payload.get("rubric") or "normalized-rubric-v1"),
        generation_config={"temperature": 0, "response_format": "structured-json"},
    )


def _read_split(dataset: str, split: str, limit: int | None = None) -> list[dict[str, Any]]:
    path = SPLIT_DIR / f"{dataset}_{split}.csv"
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result = _json(row.get("response"))
            if not isinstance(result, dict) or any(result.get(key) is None for key in SCHEMAS[dataset].fields):
                continue
            try:
                result = SCHEMAS[dataset].validate(result)
                request = _request(row["prompt"], SCHEMAS[dataset], dataset)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
            rows.append({"request": request, "oracle": result, "id": row.get("source_index", "")})
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _hash_embedding(text: str, dimension: int = 384) -> list[float]:
    values = [0.0] * dimension
    tokens = re.findall(r"[\w]+", text.lower())
    if not tokens:
        tokens = [text or "<empty>"]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dimension
        values[index] += 1.0
    return values


class LocalEmbedder:
    def __init__(self, model_path: Path, device: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
        self.model.eval()
        self.device = device
        self.cache: dict[str, list[float]] = {}

    def chunk_text(self, text: str, max_tokens: int = 480, overlap_tokens: int = 32) -> list[str]:
        tokenizer = self.model.tokenizer
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return [text or " "]
        chunks = []
        start = 0
        while start < len(token_ids):
            end = min(len(token_ids), start + max_tokens)
            chunks.append(tokenizer.decode(token_ids[start:end], skip_special_tokens=True))
            if end == len(token_ids):
                break
            start = end - overlap_tokens
        return chunks

    def encode_many(self, texts: list[str], batch_size: int = 64) -> None:
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            vectors = self.model.encode(
                batch, batch_size=batch_size, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            )
            self.cache.update({
                text: vector.astype("float32", copy=False)
                for text, vector in zip(batch, vectors)
            })

    def __call__(self, text: str) -> list[float]:
        if text not in self.cache:
            self.encode_many([text])
        return self.cache[text]


def _prepare_embedding_cache(requests, embed) -> None:
    if not hasattr(embed, "encode_many"):
        return
    texts = []
    for request in requests:
        for text in request.field_texts().values():
            texts.extend(embed.chunk_text(text))
    embed.encode_many(texts)


def _train_pairs(dataset: str, embed, cap: int, seed: int,
                 chunker=None) -> tuple[list[dict], list[bool]]:
    path = PAIR_DIR / f"{dataset}_train.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    if len(records) > cap:
        by_label = {False: [], True: []}
        for record in records:
            by_label[bool(record.get("correct", 0))].append(record)
        per_label = cap // 2
        records = rng.sample(by_label[False], min(per_label, len(by_label[False])))
        records += rng.sample(by_label[True], min(cap - len(records), len(by_label[True])))
        rng.shuffle(records)
    features, unsafe = [], []
    skipped: dict[str, int] = defaultdict(int)
    parsed = []
    for record in records:
        left, right = _json(record.get("sentence_1")), _json(record.get("sentence_2"))
        if not isinstance(left, dict) or not isinstance(right, dict):
            skipped["invalid_pair"] += 1
            continue
        try:
            left_req = _request(json.dumps(left, ensure_ascii=False), SCHEMAS[dataset], dataset)
            right_req = _request(json.dumps(right, ensure_ascii=False), SCHEMAS[dataset], dataset)
            parsed.append((record, left_req, right_req))
        except (ValueError, KeyError, TypeError) as exc:
            skipped[type(exc).__name__ + ":" + str(exc)[:80]] += 1
    _prepare_embedding_cache(
        [request for _, left, right in parsed for request in (left, right)], embed
    )
    for record, left_req, right_req in parsed:
        try:
            left_vectors = _field_vectors(left_req, embed, max_chunk_chars=1600,
                                          chunk_overlap=120, chunker=chunker)
            right_vectors = _field_vectors(right_req, embed, max_chunk_chars=1600,
                                           chunk_overlap=120, chunker=chunker)
            base = _similarity_features(left_vectors, right_vectors)
            base["similarity_margin"] = base["overall_similarity"]
            base["log_observations"] = math.log(2.0)
        except (ValueError, KeyError, TypeError) as exc:
            skipped[type(exc).__name__ + ":" + str(exc)[:80]] += 1
            continue
        features.append(base)
        unsafe.append(not bool(record.get("correct", 0)))
    if len(set(unsafe)) != 2:
        raise ValueError(
            f"{dataset}: training pairs must include both safe and unsafe examples; "
            f"usable={len(features)}, unsafe_labels={set(unsafe)}, skipped={dict(skipped)}"
        )
    return features, unsafe


def _error(oracle: dict, predicted: dict, dataset: str) -> dict:
    if dataset == "pku":
        flips = {key: oracle[key] != predicted[key] for key in oracle}
        return {"mae": 0.0 if not any(flips.values()) else 1.0,
                "max_abs_error": int(any(flips.values())), "dimension_errors": flips,
                "acceptable": not any(flips.values())}
    errors = {key: abs(float(predicted[key]) - float(oracle[key])) for key in oracle}
    return {"mae": sum(errors.values()) / len(errors), "max_abs_error": max(errors.values()),
            "dimension_errors": errors, "acceptable": max(errors.values()) <= 1.0}


def _group_bootstrap(records: list[dict], replicates: int, seed: int) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        groups[row["group"]].append(row)
    if not groups:
        return {"low": None, "high": None, "groups": 0}
    keys = sorted(groups)
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        selected = [rng.choice(keys) for _ in keys]
        rows = [row for key in selected for row in groups[key]]
        samples.append(sum(row["effective"] for row in rows) / len(rows))
    samples.sort()
    return {"low": samples[int(0.025 * (len(samples) - 1))],
            "high": samples[int(0.975 * (len(samples) - 1))], "groups": len(keys)}


def _replay(dataset: str, rows: list[dict], embed, model: RiskModel | None,
            threshold: float | None, rounds: int, seed: int, bootstrap: int,
            order_override: list[int] | None = None) -> dict:
    if model is None:
        model = RiskModel()
    model.threshold = threshold
    chunker = (lambda text, _chars, _overlap: embed.chunk_text(text)) if hasattr(embed, "chunk_text") else None
    cache = JudgeCache(embed, model, chunker=chunker)
    order = list(range(len(rows))) if order_override is None else list(order_override)
    if sorted(order) != list(range(len(rows))):
        raise ValueError("request order override must be a permutation of the replay rows")
    if order_override is None:
        random.Random(seed).shuffle(order)
    result_records = []
    for round_index in range(rounds):
        for index in order:
            item = rows[index]
            request = item["request"]
            answer = cache.evaluate(request, lambda _req, expected=item["oracle"]: JudgeResult(expected))
            diag = _error(item["oracle"], answer.values, dataset)
            trace = cache.trace[-1]
            hit = trace["decision"] == "hit"
            result_records.append({
                "round": round_index + 1,
                "request_id": item["id"],
                "group": request.instruction.strip(),
                "hit": hit,
                "effective": bool(hit and diag["acceptable"]),
                "wrong": bool(hit and not diag["acceptable"]),
                "judge_called": trace["judge_called"],
                "reason": trace["reason"],
                "risk": trace.get("risk"),
                "oracle": item["oracle"],
                "cached_or_judge_result": dict(answer.values),
                "error": diag,
                "trace": {key: value for key, value in trace.items() if not key.startswith("_")},
            })
    hits = [row for row in result_records if row["hit"]]
    errors = [row["error"] for row in hits]
    dimensions: dict[str, list[float]] = defaultdict(list)
    for error in errors:
        for name, value in error["dimension_errors"].items():
            dimensions[name].append(float(value))
    return {
        "requests": len(result_records),
        "raw_hits": len(hits),
        "raw_hit_rate": len(hits) / max(1, len(result_records)),
        "effective_hits": sum(row["effective"] for row in result_records),
        "effective_hit_rate": sum(row["effective"] for row in result_records) / max(1, len(result_records)),
        "wrong_hits": sum(row["wrong"] for row in result_records),
        "wrong_hit_rate": sum(row["wrong"] for row in result_records) / max(1, len(result_records)),
        "judge_calls": sum(row["judge_called"] for row in result_records),
        "judge_call_savings": 1 - sum(row["judge_called"] for row in result_records) / max(1, len(result_records)),
        "hit_mae": sum(error["mae"] for error in errors) / max(1, len(errors)),
        "max_dimension_error": max((error["max_abs_error"] for error in errors), default=0),
        "dimension_mae": {key: sum(values) / len(values) for key, values in dimensions.items()},
        "bootstrap_effective_hit_rate_95ci": _group_bootstrap(result_records, bootstrap, seed),
        "by_round": {
            str(round_index): {
                "requests": sum(row["round"] == round_index for row in result_records),
                "hits": sum(row["round"] == round_index and row["hit"] for row in result_records),
                "effective_hits": sum(row["round"] == round_index and row["effective"] for row in result_records),
                "judge_calls": sum(row["round"] == round_index and row["judge_called"] for row in result_records),
            }
            for round_index in range(1, rounds + 1)
        },
        "records": result_records,
    }


def _eligible(summary: dict, dataset: str) -> bool:
    if dataset == "pku":
        return summary["wrong_hits"] == 0
    return summary["wrong_hits"] == 0 and summary["hit_mae"] <= 0.25


def _normalize_mvr_summary(payload: dict, path: Path, dataset: str, seed: int,
                           split: str = "test") -> dict:
    args = payload.get("args", {})
    source = Path(str(args.get("dataset", "")))
    if dataset not in source.name.lower() or split not in source.stem.lower():
        raise ValueError(f"MVR artifact must identify {dataset}_{split}.csv: {path}")
    if int(args.get("replay_rounds", -1)) != 6:
        raise ValueError(f"MVR artifact must use six replay rounds: {path}")
    if int(args.get("seed", -1)) != seed or int(args.get("shuffle_seed", -1)) != seed:
        raise ValueError(f"MVR artifact seed/order does not match this replay: {path}")
    summary = payload.get("summary", {})
    ci = summary.get("instruction_group_bootstrap", {}).get("ci95", {}).get("effective_hit_rate")
    return {
        "path": str(path),
        "dataset": dataset,
        "requests": summary.get("n", summary.get("requests", 0)),
        "raw_hits": summary.get("hits", 0),
        "raw_hit_rate": summary.get("hit_rate", 0.0),
        "effective_hits": summary.get("effective_hits", 0),
        "effective_hit_rate": summary.get("effective_hit_rate", 0.0),
        "wrong_hits": summary.get("wrong_hits", 0),
        "wrong_hit_rate": summary.get("wrong_hit_rate", 0.0),
        "judge_calls": max(0, summary.get("n", 0) - summary.get("hits", 0)),
        "judge_call_savings": summary.get("hit_rate", 0.0),
        "hit_mae": summary.get("hit_mae", 0.0),
        "max_dimension_error": summary.get("hit_max_abs_error", 0.0),
        "dimension_mae": {},
        "bootstrap_effective_hit_rate_95ci": {
            "low": ci[0] if ci else None, "high": ci[1] if ci else None,
            "groups": summary.get("instruction_group_bootstrap", {}).get("groups", 0),
        },
    }


def _mvr_order(payload: dict, rows: list[dict]) -> list[int]:
    by_hash: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(rows):
        prompt = json.dumps(item["request"].content_payload(), ensure_ascii=False,
                            sort_keys=True, separators=(",", ":"))
        # Match the canonical JSON prompt stored in the prepared evaluation CSV.
        by_hash[hashlib.sha256(prompt.encode("utf-8")).hexdigest()].append(index)
    first_round = sorted(
        (row for row in payload.get("per_sample", []) if row.get("replay_round") == 1),
        key=lambda row: row.get("sample_index", 0),
    )
    order = []
    for record in first_round:
        matches = by_hash.get(record.get("prompt_hash"), [])
        if not matches:
            raise ValueError("MVR first-round prompt hashes do not match the current test CSV")
        order.append(matches.pop(0))
    if len(order) != len(rows) or sorted(order) != list(range(len(rows))):
        raise ValueError("MVR artifact and current test CSV do not contain the same request set")
    return order


def _threshold_sweep(dataset: str, validation, embed, model, rounds, seed, bootstrap,
                     order_override: list[int] | None = None) -> tuple[float, dict, list[dict]]:
    candidates = [-1.0, 0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    summaries = []
    for threshold in candidates:
        result = _replay(dataset, validation, embed, model, threshold, rounds, seed,
                         bootstrap, order_override)
        summaries.append({"threshold": threshold, **{k: v for k, v in result.items() if k != "records"},
                          "quality_gate_pass": _eligible(result, dataset)})
    eligible = [row for row in summaries if row["quality_gate_pass"]]
    selected = max(eligible, key=lambda row: (row["effective_hit_rate"], row["threshold"])) if eligible else None
    if selected is None:
        return -1.0, summaries[0], summaries
    return float(selected["threshold"]), selected, summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*SCHEMAS, "all"], default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hash-embeddings", action="store_true",
                        help="Use deterministic feature hashing for smoke tests, not research results.")
    parser.add_argument("--train-pair-cap", type=int, default=10000,
                        help="Maximum number of balanced training pairs per dataset; 0 uses all available pairs.")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit validation/test rows for smoke runs; never use this for reported results.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "risk_aware_judge_cache_results.json")
    parser.add_argument("--mvr-test-json", type=Path, action="append", default=[])
    parser.add_argument("--mvr-validation-json", type=Path, action="append", default=[],
                        help="MVR validation artifacts used for model selection and test gating.")
    args = parser.parse_args()
    if args.rounds != 6:
        parser.error("the frozen replay protocol requires six rounds")
    if args.train_pair_cap == 0:
        args.train_pair_cap = 2**31 - 1

    if args.hash_embeddings:
        embed = _hash_embedding
        chunker = None
        embedding_model = "deterministic-feature-hash (smoke only)"
    else:
        model_path = ROOT / "work" / "models" / "bge-small-en-v1.5"
        if not model_path.exists():
            parser.error(f"local embedding model not found: {model_path}")
        embed = LocalEmbedder(model_path, args.device)
        chunker = lambda text, _chars, _overlap: embed.chunk_text(text)
        embedding_model = str(model_path)

    datasets = list(SCHEMAS) if args.dataset == "all" else [args.dataset]
    output = {"protocol": {"seed": args.seed, "rounds": args.rounds,
                           "embedding_model": embedding_model,
                           "train_pairs_cap": args.train_pair_cap,
                           "threshold_selection": "validation only",
                           "test_used_for_tuning": False}, "datasets": {}}
    mvr_payloads = {}
    mvr_validation_payloads = {}
    for path, target, split in (
        *((path, mvr_payloads, "test") for path in args.mvr_test_json),
        *((path, mvr_validation_payloads, "val") for path in args.mvr_validation_json),
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("args", {})
        source = Path(str(metadata.get("dataset", "")))
        dataset = next((name for name in SCHEMAS if name in source.name.lower()), None)
        if dataset not in datasets:
            parser.error(f"MVR artifact dataset is not selected for this run: {path}")
        try:
            _normalize_mvr_summary(payload, path, dataset, args.seed, split)
        except ValueError as exc:
            parser.error(str(exc))
        target[dataset] = (path, payload)
    for dataset in datasets:
        print(f"[{dataset}] loading splits and training risk model", flush=True)
        train_features, train_unsafe = _train_pairs(
            dataset, embed, args.train_pair_cap, args.seed, chunker
        )
        model = RiskModel()
        model.fit(train_features, train_unsafe)
        validation_rows = _read_split(dataset, "val", args.max_rows)
        _prepare_embedding_cache([row["request"] for row in validation_rows], embed)
        validation_order = None
        if dataset in mvr_validation_payloads:
            try:
                validation_order = _mvr_order(mvr_validation_payloads[dataset][1], validation_rows)
            except ValueError as exc:
                parser.error(str(exc))
        threshold, validation_summary, sweep = _threshold_sweep(
            dataset, validation_rows, embed, model, args.rounds, args.seed,
            args.bootstrap_replicates, validation_order,
        )
        exact_validation = _replay(
            dataset, validation_rows, embed, None, -1.0, args.rounds, args.seed,
            args.bootstrap_replicates, validation_order,
        )
        mvr_validation = None
        if dataset in mvr_validation_payloads:
            path, payload = mvr_validation_payloads[dataset]
            mvr_validation = _normalize_mvr_summary(payload, path, dataset, args.seed, "val")
            mvr_validation["same_request_order_verified"] = True
        test_eligible = _eligible(validation_summary, dataset) and (
            mvr_validation is not None
            and validation_summary["effective_hit_rate"] > mvr_validation["effective_hit_rate"]
        )
        test_result = None
        exact_baseline = None
        if test_eligible:
            test_rows = _read_split(dataset, "test", args.max_rows)
            _prepare_embedding_cache([row["request"] for row in test_rows], embed)
            test_order = None
            if dataset in mvr_payloads:
                try:
                    test_order = _mvr_order(mvr_payloads[dataset][1], test_rows)
                except ValueError as exc:
                    parser.error(str(exc))
            exact_baseline = _replay(dataset, test_rows, embed, None, -1.0, args.rounds,
                                     args.seed, args.bootstrap_replicates, test_order)
            test_result = _replay(dataset, test_rows, embed, model, threshold, args.rounds,
                                  args.seed, args.bootstrap_replicates, test_order)
        output["datasets"][dataset] = {
            "risk_training_examples": len(train_features),
            "risk_training_unsafe_rate": sum(train_unsafe) / len(train_unsafe),
            "selected_validation_threshold": threshold,
            "calibrated_risk_model": model.state_dict(),
            "validation_selected_metrics": validation_summary,
            "validation_threshold_sweep": sweep,
            "exact_cache_validation": exact_validation,
            "mvr_validation_baseline": mvr_validation,
            "test_gate_passed": test_eligible,
            "exact_cache_test": exact_baseline,
            "test": test_result,
        }
        print(f"[{dataset}] threshold={threshold} test-effective={test_result['effective_hit_rate']:.3%} "
              f"test-wrong={test_result['wrong_hits']}", flush=True)
    if args.mvr_test_json:
        output["mvr_baseline_artifacts"] = []
        for dataset, (path, payload) in mvr_payloads.items():
            baseline = _normalize_mvr_summary(payload, path, dataset, args.seed)
            baseline["same_request_order_verified"] = True
            output["mvr_baseline_artifacts"].append(baseline)
    if args.mvr_validation_json:
        output["mvr_validation_baseline_artifacts"] = [
            _normalize_mvr_summary(payload, path, dataset, args.seed, "val")
            for dataset, (path, payload) in mvr_validation_payloads.items()
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = ["# 风险感知 LLM Judge 缓存回放", "",
              f"- 配置：seed={args.seed}，回放轮数={args.rounds}，嵌入={embedding_model}",
              "- 阈值仅由 validation 选择；test 未用于调参。", "",
              "| 数据集 | 策略 | 请求数 | 原始命中率 | 有效命中率 | 有效命中率 95% CI | 错误命中 | Judge 调用节省 | 命中 MAE | 门槛 |",
              "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |"]
    for dataset, value in output["datasets"].items():
        baselines = [x for x in output.get("mvr_baseline_artifacts", []) if x["dataset"] == dataset]
        compared = [
            ("精确缓存验证", value["exact_cache_validation"]),
            ("风险门控验证", value["validation_selected_metrics"]),
        ]
        if value["exact_cache_test"] is not None:
            compared.append(("精确缓存测试", value["exact_cache_test"]))
        if value["test"] is not None:
            compared.append(("风险门控测试", value["test"]))
        validation_baselines = [
            x for x in output.get("mvr_validation_baseline_artifacts", [])
            if x["dataset"] == dataset
        ]
        compared.extend(("MVR 验证", baseline) for baseline in validation_baselines)
        for baseline in baselines:
            compared.append(("MVR 测试", baseline))
        for label, summary in compared:
            gate = "通过" if _eligible(summary, dataset) else "未通过"
            ci = summary.get("bootstrap_effective_hit_rate_95ci", {})
            ci_text = f"[{ci['low']:.2%}, {ci['high']:.2%}]" if ci.get("low") is not None else "不可用"
            report.append(
                f"| {dataset} | {label} | {summary['requests']} | {summary['raw_hit_rate']:.2%} | "
                f"{summary['effective_hit_rate']:.2%} | {ci_text} | {summary['wrong_hits']} | "
                f"{summary['judge_call_savings']:.2%} | {summary['hit_mae']:.4f} | {gate} |"
            )
        if not value["test_gate_passed"]:
            report.append(
                f"{dataset}：未运行测试集；验证集未同时超过 MVR 有效命中率并通过质量门槛。"
            )
    report += ["", "逐请求风险、决策原因、候选与误差记录保存在 JSON 的 `records` 中。",
               "当前 MVR 测试基线仅在提供 `--mvr-test-json` 且该 JSON 来自同一 test workload 时列出。",
               "离线 oracle 回放不代表真实 Judge 的端到端延迟或训练吞吐。", ""]
    args.output.with_suffix(".md").write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()

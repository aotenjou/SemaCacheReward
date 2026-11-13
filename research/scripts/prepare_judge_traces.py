#!/usr/bin/env python3
"""Build reproducible, label-free judge request traces from public HF rows.

Requires Python 3.10+ and requests.  Run:
    python research/scripts/prepare_judge_traces.py

The datasets-server /rows service reads the current processed dataset view; it
does not guarantee a requested Git revision.  We record Hub revisions before
and after collection and refuse a run if one changed.  Stored raw pages permit
an exact offline rebuild with --offline.  Default spread sampling selects
systematic pages across each split; prefix sampling selects leading rows.
Both are pilot workloads, not an estimate of production traffic.  Existing
labels serve as an offline oracle; no judge API is called.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import random
import threading
import time
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "pku": {"id": "PKU-Alignment/PKU-SafeRLHF", "rows": 2000,
            "license": "cc-by-nc-4.0", "label_origin": "human preferences"},
    "helpsteer2": {"id": "nvidia/HelpSteer2", "rows": 2000,
                   "license": "cc-by-4.0", "label_origin": "human ratings"},
    "ultrafeedback": {"id": "openbmb/UltraFeedback", "rows": 500,
                      "license": "mit", "label_origin": "GPT-4 feedback"},
}
SCORE_DIMS = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
UF_DIMS = ("instruction_following", "truthfulness", "honesty", "helpfulness")
RUBRICS = {
    "pku": "pku-helpful-safe-v1: Compare candidate 0 and candidate 1 for this instruction. Return better_response_id for helpfulness and safer_response_id for harmlessness; each is 0 or 1. Judge the two dimensions independently.",
    "helpsteer2": "helpsteer2-five-attribute-v1: Rate candidate 0 on helpfulness, correctness, coherence, complexity, and verbosity using integer levels 0 through 4; higher means more of the named attribute, and complexity or verbosity need not mean better quality.",
    "ultrafeedback": "ultrafeedback-four-attribute-v1: Rate candidate 0 on instruction_following, truthfulness, honesty, and helpfulness using integer levels 1 through 5; higher means stronger performance. The context preserves the system prompt used when generating the candidate.",
}
PAGE_SIZE = 100
WORKERS = 4
RETRIES = 6
LOCAL = threading.local()
REQUEST_LOCK = threading.Lock()
LAST_REQUEST_AT = 0.0


def session() -> requests.Session:
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        LOCAL.session.headers["User-Agent"] = "judge-cache-trace-pilot/1.0"
    return LOCAL.session


def request_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    global LAST_REQUEST_AT
    for attempt in range(RETRIES):
        try:
            # Keep concurrent page workers below the Hub's burst limits.
            with REQUEST_LOCK:
                delay = 0.35 - (time.monotonic() - LAST_REQUEST_AT)
                if delay > 0:
                    time.sleep(delay)
                LAST_REQUEST_AT = time.monotonic()
            response = session().get(url, params=params, timeout=(20, 120))
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait = min(120.0, float(retry_after)) if retry_after and retry_after.isdigit() else min(60, 5 * (2 ** attempt))
                time.sleep(wait + random.Random(attempt).random())
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt + 1 == RETRIES:
                raise
            # Bounded exponential retry; request order is restored below.
            time.sleep(min(20, 2 ** attempt) + random.Random(attempt).random())
    raise AssertionError("unreachable")


def json_text(obj: Any) -> str:
    # Escape Unicode line separators so JSON strings remain one physical JSONL line.
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hub_metadata(dataset_id: str) -> dict[str, Any]:
    info = request_json(f"https://huggingface.co/api/datasets/{dataset_id}")
    return {"revision": info["sha"], "last_modified": info.get("lastModified"),
            "gated": info.get("gated", False)}


def fetch_page(key: str, offset: int, output_dir: Path, offline: bool,
               reuse_existing: bool = False) -> tuple[str, int, dict[str, Any]]:
    spec = DATASETS[key]
    path = output_dir / "raw" / key / f"rows_{offset:06d}.json"
    if offline or (reuse_existing and path.exists()):
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = request_json("https://datasets-server.huggingface.co/rows", {
            "dataset": spec["id"], "config": "default", "split": "train",
            "offset": offset, "length": PAGE_SIZE,
        })
        write_json(path, payload)
    expected = PAGE_SIZE
    if payload.get("partial"):
        raise RuntimeError(f"{key} at {offset}: dataset view is marked partial")
    if len(payload.get("rows", [])) != expected:
        raise RuntimeError(f"{key} at {offset}: expected {expected} rows, received {len(payload.get('rows', []))}")
    for idx, row in enumerate(payload["rows"], offset):
        if row["row_idx"] != idx or row.get("truncated_cells"):
            raise RuntimeError(f"{key} at {idx}: noncontiguous or truncated row: {row.get('truncated_cells')}")
    return key, offset, payload


def split_row_count(dataset_id: str) -> int:
    payload = request_json("https://datasets-server.huggingface.co/size", {"dataset": dataset_id})
    matches = [item["num_rows"] for item in payload["size"]["splits"]
               if item.get("config") == "default" and item.get("split") == "train"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one default/train row count for {dataset_id}; got {matches}")
    return int(matches[0])


def page_offsets(total_rows: int, count: int, sampling: str, prefix_rows: int) -> list[int]:
    if sampling == "prefix":
        return list(range(0, prefix_rows, PAGE_SIZE))
    if total_rows < PAGE_SIZE:
        raise RuntimeError(f"Split has fewer than {PAGE_SIZE} rows: {total_rows}")
    last_start = total_rows - PAGE_SIZE
    starts = []
    for i in range(count):
        # Equivalent to rounded linspace(0, total_rows - PAGE_SIZE, count),
        # then align each selected page to the datasets-server page boundary.
        point = last_start * i / max(1, count - 1)
        starts.append(min((last_start // PAGE_SIZE) * PAGE_SIZE,
                          int(math.floor(point / PAGE_SIZE + 0.5)) * PAGE_SIZE))
    if len(set(starts)) != count:
        raise RuntimeError(f"Could not select {count} distinct page offsets from {total_rows} rows: {starts}")
    return sorted(starts)


def make_record(key: str, source_index: int, instruction: Any, candidates: list[Any],
                context: dict[str, Any], oracle: dict[str, Any],
                completion_index: int | None = None) -> dict[str, Any]:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{key}:{source_index}: missing instruction")
    if any(not isinstance(x, str) for x in candidates):
        raise ValueError(f"{key}:{source_index}: non-string candidate")
    judge_input = {"rubric": RUBRICS[key], "context": context,
                   "instruction": instruction, "candidates": candidates}
    record = {"dataset": key, "source_index": source_index, **judge_input,
              "oracle": oracle, "prompt": json_text(judge_input)}
    if completion_index is not None:
        record["completion_index"] = completion_index
    # The string given to an embedding or judge has these exact four fields.
    assert set(json.loads(record["prompt"])) == {"rubric", "context", "instruction", "candidates"}
    assert json.loads(record["prompt"]) == judge_input
    return record


def parse_uf_rating(raw: Any, key: str, idx: int, completion: int,
                    anomalies: list[dict[str, Any]]) -> int | None:
    if isinstance(raw, bool):
        value = None
    else:
        try:
            number = float(raw)
            value = int(number) if math.isfinite(number) and number.is_integer() and 1 <= number <= 5 else None
        except (TypeError, ValueError):
            value = None
    if value is None:
        anomalies.append({"source_index": idx, "completion_index": completion,
                          "dimension": key, "raw_rating": raw,
                          "reason": "missing_or_invalid_1_to_5_rating", "action": "stored_as_null"})
    return value


def transform(key: str, source_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    anomalies: list[dict[str, Any]] = []
    completion_counts: collections.Counter[int] = collections.Counter()
    uf_source_counts: collections.Counter[str] = collections.Counter()
    for indexed in source_rows:
        idx, row = indexed["row_idx"], indexed["row"]
        if key == "pku":
            oracle = {dim: row[dim] for dim in ("better_response_id", "safer_response_id")}
            if any(type(value) is not int or value not in (0, 1) for value in oracle.values()):
                raise ValueError(f"pku:{idx}: invalid preference {oracle}")
            records.append(make_record(key, idx, row["prompt"], [row["response_0"], row["response_1"]], {}, oracle))
        elif key == "helpsteer2":
            oracle = {dim: row[dim] for dim in SCORE_DIMS}
            if any(type(value) is not int or value not in range(5) for value in oracle.values()):
                raise ValueError(f"helpsteer2:{idx}: invalid score {oracle}")
            records.append(make_record(key, idx, row["prompt"], [row["response"]], {}, oracle))
        else:
            uf_source_counts[str(row.get("source") or "<missing>")] += 1
            completions = row["completions"]
            completion_counts[len(completions)] += 1
            for ci, candidate in enumerate(completions):
                annotations = candidate.get("annotations") or {}
                oracle = {dim: parse_uf_rating((annotations.get(dim) or {}).get("Rating"), dim,
                                               idx, ci, anomalies) for dim in UF_DIMS}
                # Preserve generating context, never model scores or critiques.
                context = {"custom_system_prompt": candidate.get("custom_system_prompt") or ""}
                records.append(make_record(key, idx, row["instruction"], [candidate["response"]], context, oracle, ci))
    nulls = collections.Counter(dim for record in records for dim, value in record["oracle"].items() if value is None)
    empty_candidates = [{"source_index": record["source_index"],
                         "completion_index": record.get("completion_index"),
                         "candidate_index": candidate_index, "action": "preserved_as_empty_string"}
                        for record in records for candidate_index, candidate in enumerate(record["candidates"])
                        if not candidate.strip()]
    return records, {"source_rows": len(source_rows), "requests": len(records),
                     "records_with_complete_oracle": sum(all(v is not None for v in r["oracle"].values()) for r in records),
                     "null_oracle_fields_by_dimension": dict(nulls), "rating_anomalies": anomalies,
                     "source_rows_by_completion_count": dict(completion_counts),
                     "empty_candidates": empty_candidates,
                     "ultrafeedback_instruction_source_distribution": dict(sorted(uf_source_counts.items())),
                     "empty_custom_system_prompts": sum(not r["context"].get("custom_system_prompt", "") for r in records) if key == "ultrafeedback" else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work" / "data")
    parser.add_argument("--offline", action="store_true", help="Rebuild from stored raw pages and source snapshot metadata")
    parser.add_argument("--sampling", choices=("prefix", "spread"), default="spread",
                        help="prefix reproduces the leading-row pilot; spread samples pages systematically across each split")
    args = parser.parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "source_snapshot.json"
    if args.offline:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        before = snapshot["hub_before"]
        sampling_plan = snapshot.get("sampling_plan")
        if sampling_plan is None:
            # Backward compatibility for the original prefix snapshot.
            sampling_plan = {key: {"num_rows_total": json.loads(
                                    (output_dir / "raw" / key / "rows_000000.json").read_text(encoding="utf-8")
                                  )["num_rows_total"],
                                  "page_offsets": list(range(0, spec["rows"], PAGE_SIZE))}
                             for key, spec in DATASETS.items()}
        sampling_mode = snapshot.get("sampling", "prefix")
    else:
        prior_snapshot = (json.loads(snapshot_path.read_text(encoding="utf-8"))
                          if snapshot_path.exists() else None)
        before = {key: hub_metadata(spec["id"]) for key, spec in DATASETS.items()}
        totals = {key: split_row_count(spec["id"]) for key, spec in DATASETS.items()}
        sampling_plan = {key: {"num_rows_total": totals[key],
                               "page_offsets": page_offsets(totals[key], math.ceil(spec["rows"] / PAGE_SIZE),
                                                             args.sampling, spec["rows"])}
                         for key, spec in DATASETS.items()}
        sampling_mode = args.sampling
        reuse_keys = {key for key in DATASETS
                      if prior_snapshot
                      and prior_snapshot.get("hub_before", {}).get(key, {}).get("revision") == before[key]["revision"]
                      and prior_snapshot.get("sampling_plan", {}).get(key, {}).get("page_offsets") == sampling_plan[key]["page_offsets"]}
        snapshot = {"download_started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "hub_before": before, "sampling": sampling_mode, "sampling_plan": sampling_plan}
        write_json(snapshot_path, snapshot)

    jobs = [(key, offset) for key in DATASETS for offset in sampling_plan[key]["page_offsets"]]
    pages: dict[str, dict[int, dict[str, Any]]] = {key: {} for key in DATASETS}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch_page, key, offset, output_dir, args.offline,
                               not args.offline and key in reuse_keys) for key, offset in jobs]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            key, offset, payload = future.result()
            pages[key][offset] = payload
            print(f"Fetched {completed}/{len(jobs)} pages: {key} offset={offset}", flush=True)

    after = before if args.offline else {key: hub_metadata(spec["id"]) for key, spec in DATASETS.items()}
    if any(before[key]["revision"] != after[key]["revision"] for key in DATASETS):
        raise RuntimeError("A Hub revision changed during download; rerun to collect a stable snapshot")
    if not args.offline:
        snapshot.update({"download_finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "hub_after": after})
        write_json(snapshot_path, snapshot)

    dataset_metadata = {}
    for key, spec in DATASETS.items():
        source_rows = sorted((row for offset in sorted(pages[key]) for row in pages[key][offset]["rows"]),
                             key=lambda row: row["row_idx"])
        records, stats = transform(key, source_rows)
        path = output_dir / f"{key}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json_text(record) + "\n")
        counts = collections.Counter(r["prompt"] for r in records)
        dataset_metadata[key] = {"hf_dataset": spec["id"], "url": f"https://huggingface.co/datasets/{spec['id']}",
                                 "hf_revision": before[key]["revision"], "hf_revision_after_download": after[key]["revision"],
                                 "config": "default", "split": "train", "license": spec["license"],
                                 "label_origin": spec["label_origin"],
                                 "sample_rule": (f"first {spec['rows']} source rows in datasets-server order" if sampling_mode == "prefix"
                                                 else f"systematic page sample: {len(sampling_plan[key]['page_offsets'])} evenly spaced, page-aligned pages; not IID"),
                                 "sampling": sampling_mode,
                                 "source_num_rows_total": sampling_plan[key]["num_rows_total"],
                                 "page_offsets": sampling_plan[key]["page_offsets"],
                                 "source_row_indices": [r["row_idx"] for r in source_rows], **stats,
                                 "exact_unique_requests": len(counts),
                                 "exact_duplicate_requests": len(records) - len(counts),
                                 "trace_file": str(path.relative_to(output_dir)), "trace_sha256": sha256(path),
                                 "raw_page_sha256": {str(offset): sha256(output_dir / "raw" / key / f"rows_{offset:06d}.json") for offset in sorted(pages[key])}}
        print(f"Prepared {key}: {len(records)} requests; {stats['records_with_complete_oracle']} complete oracle labels", flush=True)

    metadata = {"format_version": 1, "script_sha256": sha256(Path(__file__)),
                "prepared_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "download_started_at_utc": snapshot["download_started_at_utc"],
                "download_finished_at_utc": snapshot.get("download_finished_at_utc"),
                "elapsed_seconds": round(time.perf_counter() - started, 3), "offline_rebuild": args.offline,
                "sampling": sampling_mode,
                "transport": {"api": "https://datasets-server.huggingface.co/rows", "page_size": PAGE_SIZE,
                              "workers": WORKERS, "max_attempts": RETRIES, "truncated_rows_allowed": False},
                "revision_caveat": "The /rows API serves its current processed view rather than a revision-pinned request. Hub revisions were unchanged before and after download; raw pages are saved for exact offline reconstruction.",
                "prompt_contract": {"fields": ["rubric", "context", "instruction", "candidates"],
                                    "serialization": "UTF-8 JSON, sorted keys, compact separators, unescaped Unicode",
                                    "label_leakage": "oracle, ratings, preference ids, critique and rationales are excluded from every prompt"},
                "rubric_caveat": "Rubrics are concise normalized evaluation tasks, not the original full annotation/API prompts. Human labels in PKU/HelpSteer2 and GPT-4 labels in UltraFeedback are offline oracle values. This is an offline request replay pilot; it does not measure an online policy distribution or judge consistency.",
                "missing_rating_policy": "Missing/invalid UltraFeedback Rating is stored as null; full position, dimension and raw value are recorded. Never impute a score. Accuracy evaluators must skip null dimensions or report complete-case coverage.",
                "empty_candidate_policy": "An original empty string is a valid empty-answer input and is preserved without imputation or removal; positions are recorded per dataset.",
                "datasets": dataset_metadata}
    write_json(output_dir / "metadata.json", metadata)
    print(f"Done in {metadata['elapsed_seconds']:.3f} seconds. Metadata: {output_dir / 'metadata.json'}", flush=True)


if __name__ == "__main__":
    main()

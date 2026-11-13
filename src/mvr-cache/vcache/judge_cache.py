"""Framework-agnostic, field-aware cache adapter for structured LLM Judge rewards."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import threading
from typing import Any, Callable, Mapping, Sequence

import numpy as np


EmbeddingFn = Callable[[str], Sequence[float]]
ChunkFn = Callable[[str, int, int], Sequence[str]]
JudgeFn = Callable[["JudgeRequest"], "JudgeResult | Mapping[str, Any]"]


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RewardSchema:
    """Validation contract for one structured Judge reward format."""

    name: str
    numeric_ranges: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    categorical_values: Mapping[str, frozenset[str | int]] = field(default_factory=dict)

    @property
    def fields(self) -> frozenset[str]:
        return frozenset(self.numeric_ranges) | frozenset(self.categorical_values)

    def validate(self, values: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(values, Mapping) or set(values) != self.fields:
            raise ValueError(f"reward fields must be exactly {sorted(self.fields)}")
        clean: dict[str, Any] = {}
        for key, (low, high) in self.numeric_ranges.items():
            value = values[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"reward field {key!r} must be numeric")
            number = float(value)
            if not math.isfinite(number) or number < low or number > high:
                raise ValueError(f"reward field {key!r} must be in [{low}, {high}]")
            clean[key] = number
        for key, allowed in self.categorical_values.items():
            value = values[key]
            if value not in allowed:
                raise ValueError(f"reward field {key!r} must be one of {sorted(allowed, key=str)}")
            clean[key] = value
        return clean


PKU_PREFERENCE = RewardSchema(
    name="pku_preference_v1",
    categorical_values={
        "better_response_id": frozenset({0, 1}),
        "safer_response_id": frozenset({0, 1}),
    },
)
HELPSTEER2_REWARD = RewardSchema(
    name="helpsteer2_five_attribute_v1",
    numeric_ranges={
        name: (0, 4)
        for name in ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
    },
)
ULTRAFEEDBACK_REWARD = RewardSchema(
    name="ultrafeedback_four_attribute_v1",
    numeric_ranges={
        name: (1, 5)
        for name in ("instruction_following", "truthfulness", "honesty", "helpfulness")
    },
)


@dataclass(frozen=True)
class JudgeRequest:
    instruction: str
    candidates: tuple[str, ...]
    reward_schema: RewardSchema
    judge_model: str
    prompt_version: str
    generation_config: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    rubric: str = ""

    def __post_init__(self) -> None:
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")
        if not self.candidates or any(not isinstance(item, str) for item in self.candidates):
            raise ValueError("candidates must contain one or more strings")
        if not self.judge_model or not self.prompt_version:
            raise ValueError("judge_model and prompt_version are required")

    def compatibility_key(self) -> str:
        return _digest(
            {
                "schema": self.reward_schema.name,
                "fields": sorted(self.reward_schema.fields),
                "judge_model": self.judge_model,
                "prompt_version": self.prompt_version,
                "generation_config": self.generation_config,
            }
        )

    def exact_key(self) -> str:
        return _digest({"compatibility": self.compatibility_key(), "content": self.content_payload()})

    def content_payload(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "context": self.context,
            "rubric": self.rubric,
            "candidates": list(self.candidates),
        }

    def field_texts(self) -> dict[str, str]:
        fields = {
            "instruction": self.instruction,
            "context": _canonical(self.context),
            "rubric": self.rubric,
        }
        fields.update({f"candidate_{i}": text for i, text in enumerate(self.candidates)})
        return fields


@dataclass(frozen=True)
class JudgeResult:
    values: Mapping[str, Any]

    def validated(self, schema: RewardSchema) -> "JudgeResult":
        return JudgeResult(schema.validate(self.values))


@dataclass
class _Entry:
    request: JudgeRequest
    result: JudgeResult
    vectors: dict[str, np.ndarray]
    observations: int = 1


def _normalize(vector: Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError("embedding function must return a non-empty finite vector")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("embedding function returned a zero vector")
    return value / norm


def _text_chunks(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text or " "]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = text.rfind(" ", start + max_chars // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _field_vectors(
    request: JudgeRequest,
    embed: EmbeddingFn,
    *,
    max_chunk_chars: int,
    chunk_overlap: int,
    chunker: ChunkFn | None = None,
) -> dict[str, np.ndarray]:
    vectors = {}
    for name, text in request.field_texts().items():
        chunks = (
            list(chunker(text, max_chunk_chars, chunk_overlap))
            if chunker is not None
            else _text_chunks(text, max_chunk_chars, chunk_overlap)
        )
        if not chunks:
            chunks = [" "]
        chunk_vectors = [_normalize(embed(chunk)) for chunk in chunks]
        if len({vector.size for vector in chunk_vectors}) != 1:
            raise ValueError("embedding function returned inconsistent dimensions")
        vectors[name] = _normalize(np.mean(chunk_vectors, axis=0))
    return vectors


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None or left.shape != right.shape:
        return 0.0
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def _similarity_features(query: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]) -> dict[str, float]:
    instruction = _cosine(query.get("instruction"), candidate.get("instruction"))
    context = _cosine(query.get("context"), candidate.get("context"))
    rubric = _cosine(query.get("rubric"), candidate.get("rubric"))
    candidate_names = sorted(name for name in query if name.startswith("candidate_"))
    candidate_sims = [
        _cosine(query.get(name), candidate.get(name)) for name in candidate_names
    ]
    candidate_similarity = float(np.mean(candidate_sims)) if candidate_sims else 0.0
    overall = 0.18 * instruction + 0.08 * context + 0.14 * rubric + 0.60 * candidate_similarity
    return {
        "overall_similarity": overall,
        "instruction_similarity": instruction,
        "context_similarity": context,
        "rubric_similarity": rubric,
        "candidate_similarity": candidate_similarity,
    }


class RiskModel:
    """Small regularized logistic model over cache-match diagnostics."""

    FEATURE_NAMES = (
        "overall_similarity",
        "instruction_similarity",
        "context_similarity",
        "rubric_similarity",
        "candidate_similarity",
        "similarity_margin",
        "log_observations",
    )

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.threshold: float | None = None

    def state_dict(self) -> dict[str, Any]:
        if self.mean is None or self.scale is None or self.weights is None:
            raise RuntimeError("cannot serialize an unfitted risk model")
        return {
            "feature_names": list(self.FEATURE_NAMES),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "threshold": self.threshold,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "RiskModel":
        if list(state.get("feature_names", [])) != list(cls.FEATURE_NAMES):
            raise ValueError("risk model feature schema does not match this implementation")
        model = cls()
        model.mean = np.asarray(state["mean"], dtype=np.float64)
        model.scale = np.asarray(state["scale"], dtype=np.float64)
        model.weights = np.asarray(state["weights"], dtype=np.float64)
        if model.mean.shape != (len(cls.FEATURE_NAMES),):
            raise ValueError("invalid risk model mean shape")
        if model.scale.shape != model.mean.shape or model.weights.shape != (len(cls.FEATURE_NAMES) + 1,):
            raise ValueError("invalid risk model parameter shape")
        model.threshold = None if state.get("threshold") is None else float(state["threshold"])
        if not np.isfinite(model.mean).all() or not np.isfinite(model.scale).all() or not np.isfinite(model.weights).all():
            raise ValueError("risk model parameters must be finite")
        if (model.scale <= 0).any():
            raise ValueError("risk model scales must be positive")
        if model.threshold is not None and not 0.0 <= model.threshold <= 1.0 and model.threshold != -1.0:
            raise ValueError("risk model threshold must be in [0, 1] or -1 for fail-closed")
        return model

    @staticmethod
    def vector(features: Mapping[str, float]) -> np.ndarray:
        return np.asarray(
            [float(features[name]) for name in RiskModel.FEATURE_NAMES], dtype=np.float64
        )

    def fit(self, features: Sequence[Mapping[str, float]], unsafe: Sequence[bool]) -> None:
        if len(features) != len(unsafe) or not features:
            raise ValueError("risk training requires equally sized non-empty features and labels")
        x = np.vstack([self.vector(item) for item in features])
        y = np.asarray(unsafe, dtype=np.float64)
        if np.unique(y).size < 2:
            raise ValueError("risk training requires both safe and unsafe examples")
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        z = (x - self.mean) / self.scale
        design = np.column_stack([np.ones(len(z)), z])
        weights = np.zeros(design.shape[1], dtype=np.float64)
        for _ in range(1500):
            logits = np.clip(design @ weights, -30, 30)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            gradient = design.T @ (probabilities - y) / len(y)
            gradient[1:] += 1e-3 * weights[1:]
            weights -= 0.08 * gradient
        self.weights = weights

    def predict_risk(self, features: Mapping[str, float]) -> float:
        if self.mean is None or self.scale is None or self.weights is None:
            raise RuntimeError("risk model is not fitted")
        z = (self.vector(features) - self.mean) / self.scale
        probability = 1.0 / (1.0 + math.exp(-float(np.clip(self.weights[0] + z @ self.weights[1:], -30, 30))))
        return probability

    def calibrate_threshold(
        self,
        probabilities: Sequence[float],
        unsafe: Sequence[bool],
        *,
        max_unsafe_rate: float,
        require_zero_unsafe: bool = False,
    ) -> float:
        if not 0.0 <= max_unsafe_rate <= 1.0:
            raise ValueError("max_unsafe_rate must be between 0 and 1")
        if len(probabilities) != len(unsafe) or not probabilities:
            raise ValueError("threshold calibration requires equally sized non-empty inputs")
        values = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(unsafe, dtype=bool)
        candidates = sorted(set(values.tolist()) | {0.0, 1.0})
        eligible = []
        for threshold in candidates:
            accepted = values <= threshold
            count = int(accepted.sum())
            if not count:
                continue
            errors = int((labels & accepted).sum())
            if require_zero_unsafe and errors:
                continue
            if errors / count <= max_unsafe_rate:
                eligible.append((count, threshold))
        self.threshold = max(eligible)[1] if eligible else -1.0
        return self.threshold


class JudgeCache:
    """Field-aware semantic reward cache with a fail-closed risk gate."""

    def __init__(
        self,
        embed: EmbeddingFn,
        risk_model: RiskModel,
        *,
        max_chunk_chars: int = 1600,
        chunk_overlap: int = 120,
        chunker: ChunkFn | None = None,
        max_candidates: int = 10000,
    ) -> None:
        if max_chunk_chars < 32 or chunk_overlap < 0 or chunk_overlap >= max_chunk_chars:
            raise ValueError("invalid long-text chunk settings")
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        self.embed = embed
        self.risk_model = risk_model
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap = chunk_overlap
        self.chunker = chunker
        self.max_candidates = max_candidates
        self._entries: list[_Entry] = []
        self._lock = threading.RLock()
        self.trace: list[dict[str, Any]] = []

    def _vectors(self, request: JudgeRequest) -> dict[str, np.ndarray]:
        return _field_vectors(
            request,
            self.embed,
            max_chunk_chars=self.max_chunk_chars,
            chunk_overlap=self.chunk_overlap,
            chunker=self.chunker,
        )

    def _compatible(self, query: JudgeRequest, entry: _Entry) -> bool:
        return query.compatibility_key() == entry.request.compatibility_key()

    def _insert(self, request: JudgeRequest, result: JudgeResult, vectors=None) -> None:
        entry = _Entry(request, result, vectors or self._vectors(request))
        with self._lock:
            self._entries.append(entry)

    def _fallback(self, request: JudgeRequest, judge_fn: JudgeFn, reason: str, trace: dict) -> JudgeResult:
        trace.update({"decision": "judge", "reason": reason, "judge_called": True})
        query_vectors = trace.pop("_query_vectors", None)
        try:
            raw_result = judge_fn(request)
            result = raw_result if isinstance(raw_result, JudgeResult) else JudgeResult(raw_result)
            result = result.validated(request.reward_schema)
        except Exception as exc:
            trace["error"] = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self.trace.append(trace)
            raise
        self._insert(request, result, query_vectors)
        trace["result"] = dict(result.values)
        with self._lock:
            self.trace.append(trace)
        return result

    def evaluate(self, request: JudgeRequest, judge_fn: JudgeFn) -> JudgeResult:
        query_vectors = self._vectors(request)
        trace: dict[str, Any] = {
            "request_key": request.exact_key(),
            "compatibility_key": request.compatibility_key(),
            "candidate_count": len(request.candidates),
            "field_names": sorted(query_vectors),
            "judge_called": False,
        }
        with self._lock:
            entries = tuple(self._entries)
        for entry in entries:
            if entry.request.exact_key() == request.exact_key():
                with self._lock:
                    entry.observations += 1
                trace.update({"decision": "hit", "reason": "exact_match", "risk": 0.0,
                              "cache_key": entry.request.exact_key(), "features": {}})
                with self._lock:
                    self.trace.append(trace)
                return entry.result

        ranked = []
        compatible_count_mismatch = 0
        for entry in entries:
            if self._compatible(request, entry):
                if len(request.candidates) != len(entry.request.candidates):
                    compatible_count_mismatch += 1
                    continue
                features = _similarity_features(query_vectors, entry.vectors)
                ranked.append((features["overall_similarity"], entry, features))
        ranked.sort(key=lambda item: item[0], reverse=True)
        ranked = ranked[: max(1, int(self.max_candidates))]
        if not ranked:
            trace["_query_vectors"] = query_vectors
            reason = "candidate_count_mismatch" if compatible_count_mismatch else "no_compatible_candidate"
            trace["compatible_candidate_count_mismatch"] = compatible_count_mismatch
            return self._fallback(request, judge_fn, reason, trace)

        score, entry, features = ranked[0]
        margin = score - ranked[1][0] if len(ranked) > 1 else score
        trace["candidate_diagnostics"] = [
            {
                "cache_key": candidate.request.exact_key(),
                "overall_similarity": float(candidate_score),
                **candidate_features,
            }
            for candidate_score, candidate, candidate_features in ranked[:5]
        ]
        features = {
            **features,
            "similarity_margin": float(margin),
            "log_observations": math.log1p(entry.observations),
        }
        if (
            self.risk_model.mean is None
            or self.risk_model.scale is None
            or self.risk_model.weights is None
            or self.risk_model.threshold is None
        ):
            trace.update({
                "top_candidate_key": entry.request.exact_key(),
                "top_similarity": float(score),
                "candidate_margin": float(margin),
                "observation_count": entry.observations,
                "features": features,
                "risk": None,
                "risk_threshold": self.risk_model.threshold,
                "_query_vectors": query_vectors,
            })
            return self._fallback(request, judge_fn, "risk_model_uncalibrated", trace)
        risk = self.risk_model.predict_risk(features)
        trace.update({
            "top_candidate_key": entry.request.exact_key(),
            "top_similarity": float(score),
            "candidate_margin": float(margin),
            "observation_count": entry.observations,
            "risk": risk,
            "risk_threshold": self.risk_model.threshold,
            "features": features,
        })
        threshold = self.risk_model.threshold
        if risk > threshold:
            trace["_query_vectors"] = query_vectors
            return self._fallback(request, judge_fn, "risk_above_threshold", trace)

        with self._lock:
            entry.observations += 1
        trace.update({"decision": "hit", "reason": "semantic_risk_accepted",
                      "judge_called": False, "cache_key": entry.request.exact_key()})
        with self._lock:
            self.trace.append(trace)
        return entry.result


def fit_risk_threshold(
    model: RiskModel,
    train_features: Sequence[Mapping[str, float]],
    train_unsafe: Sequence[bool],
    validation_features: Sequence[Mapping[str, float]],
    validation_unsafe: Sequence[bool],
    *,
    max_unsafe_rate: float,
    require_zero_unsafe: bool = False,
) -> float:
    """Fit on train examples and select the acceptance threshold on validation only."""
    model.fit(train_features, train_unsafe)
    risks = [model.predict_risk(features) for features in validation_features]
    return model.calibrate_threshold(
        risks,
        validation_unsafe,
        max_unsafe_rate=max_unsafe_rate,
        require_zero_unsafe=require_zero_unsafe,
    )

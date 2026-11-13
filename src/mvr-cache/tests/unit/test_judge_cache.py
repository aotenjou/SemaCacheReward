import hashlib
import re

import numpy as np
import pytest

from vcache.judge_cache import (
    HELPSTEER2_REWARD,
    PKU_PREFERENCE,
    ULTRAFEEDBACK_REWARD,
    JudgeCache,
    JudgeRequest,
    JudgeResult,
    RiskModel,
)


def embed(text):
    vector = [0.0] * 64
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        index = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % len(vector)
        vector[index] += 1.0
    return vector


def request(*, instruction="explain gravity", candidates=("gravity attracts mass",),
            rubric="score correctness", model="judge-v1", prompt_version="p1",
            schema=HELPSTEER2_REWARD, generation_config=None):
    return JudgeRequest(
        instruction=instruction,
        candidates=tuple(candidates),
        reward_schema=schema,
        judge_model=model,
        prompt_version=prompt_version,
        generation_config=generation_config or {"temperature": 0},
        context={"system": "helpful"},
        rubric=rubric,
    )


def score(value=3):
    return JudgeResult({name: value for name in HELPSTEER2_REWARD.fields})


def ready_model(threshold):
    model = RiskModel()
    width = len(model.FEATURE_NAMES)
    model.mean = np.zeros(width)
    model.scale = np.ones(width)
    model.weights = np.zeros(width + 1)
    model.threshold = threshold
    return model


def test_miss_calls_judge_validates_and_writes_through():
    cache = JudgeCache(embed, RiskModel())
    calls = []
    result = cache.evaluate(request(), lambda req: calls.append(req) or score())

    assert result.values["helpfulness"] == 3.0
    assert len(calls) == 1
    assert cache.trace[-1]["reason"] == "no_compatible_candidate"
    assert cache.trace[-1]["judge_called"] is True


def test_exact_hit_skips_judge():
    cache = JudgeCache(embed, RiskModel())
    calls = []
    req = request()
    cache.evaluate(req, lambda _req: calls.append(1) or score())
    result = cache.evaluate(req, lambda _req: calls.append(2) or score(4))

    assert result.values["helpfulness"] == 3.0
    assert calls == [1]
    assert cache.trace[-1]["reason"] == "exact_match"


def test_calibrated_semantic_hit_and_risk_rejection_fallback():
    cache = JudgeCache(embed, RiskModel())
    base = request()
    cache.evaluate(base, lambda _req: score())
    cache.risk_model = ready_model(1.0)
    similar = request(instruction="please explain gravity")
    result = cache.evaluate(similar, lambda _req: score(4))
    assert result.values["helpfulness"] == 3.0
    assert cache.trace[-1]["reason"] == "semantic_risk_accepted"
    assert cache.trace[-1]["candidate_diagnostics"][0]["instruction_similarity"] > 0

    cache.risk_model.threshold = 0.0
    changed = request(instruction="explain gravity for a child")
    calls = []
    result = cache.evaluate(changed, lambda _req: calls.append(1) or score(4))
    assert result.values["helpfulness"] == 4.0
    assert calls == [1]
    assert cache.trace[-1]["reason"] == "risk_above_threshold"


@pytest.mark.parametrize("change", [
    {"model": "judge-v2"},
    {"prompt_version": "p2"},
    {"generation_config": {"temperature": 0.7}},
])
def test_judge_configuration_mismatch_never_reuses_entry(change):
    cache = JudgeCache(embed, RiskModel())
    cache.evaluate(request(), lambda _req: score())
    calls = []
    cache.evaluate(request(**change), lambda _req: calls.append(1) or score(2))
    assert calls == [1]
    assert cache.trace[-1]["reason"] == "no_compatible_candidate"


def test_candidate_order_and_rubric_are_part_of_content():
    cache = JudgeCache(embed, RiskModel())
    original = request(candidates=("answer A", "answer B"))
    cache.evaluate(original, lambda _req: score())
    reversed_candidates = request(candidates=("answer B", "answer A"))
    assert original.exact_key() != reversed_candidates.exact_key()
    cache.risk_model = ready_model(0.0)
    calls = []
    cache.evaluate(reversed_candidates, lambda _req: calls.append(1) or score(2))
    assert calls == [1]
    assert cache.trace[-1]["reason"] == "risk_above_threshold"

    changed_rubric = request(candidates=("answer A", "answer B"), rubric="score safety")
    calls.clear()
    cache.evaluate(changed_rubric, lambda _req: calls.append(1) or score(2))
    assert calls == [1]


def test_candidate_count_mismatch_falls_back_and_records_diagnostics():
    cache = JudgeCache(embed, ready_model(1.0))
    cache.evaluate(request(candidates=("answer A", "answer B")), lambda _req: score())
    calls = []
    cache.evaluate(request(candidates=("answer A",)),
                   lambda _req: calls.append(1) or score(4))

    assert calls == [1]
    trace = cache.trace[-1]
    assert trace["reason"] == "candidate_count_mismatch"
    assert trace["compatible_candidate_count_mismatch"] == 1
    assert "candidate_diagnostics" not in trace


def test_invalid_result_and_judge_exception_are_not_cached():
    cache = JudgeCache(embed, RiskModel())
    with pytest.raises(ValueError, match="reward fields"):
        cache.evaluate(request(), lambda _req: {"helpfulness": 1})
    assert not cache._entries

    with pytest.raises(RuntimeError, match="judge down"):
        cache.evaluate(request(), lambda _req: (_ for _ in ()).throw(RuntimeError("judge down")))
    assert not cache._entries
    assert cache.trace[-1]["judge_called"] is True


def test_reward_schemas_validate_all_supported_granularities():
    assert PKU_PREFERENCE.validate(
        {"better_response_id": 0, "safer_response_id": 1}
    )["safer_response_id"] == 1
    with pytest.raises(ValueError, match="one of"):
        PKU_PREFERENCE.validate({"better_response_id": 2, "safer_response_id": 1})
    assert HELPSTEER2_REWARD.validate({name: 4 for name in HELPSTEER2_REWARD.fields})
    assert ULTRAFEEDBACK_REWARD.validate({name: 5 for name in ULTRAFEEDBACK_REWARD.fields})
    with pytest.raises(ValueError, match="must be in"):
        ULTRAFEEDBACK_REWARD.validate({name: 6 for name in ULTRAFEEDBACK_REWARD.fields})


def test_long_text_is_chunked_without_dropping_tail():
    seen = []
    cache = JudgeCache(embed=lambda text: (seen.append(text) or embed(text)),
                      risk_model=RiskModel(), max_chunk_chars=64, chunk_overlap=8)
    text = "alpha " * 30 + "unique-tail-token"
    cache.evaluate(request(candidates=(text,)), lambda _req: score())
    chunks_for_candidate = [chunk for chunk in seen if "unique-tail-token" in chunk]
    assert chunks_for_candidate
    assert sum(len(chunk) for chunk in seen) > len(text)


def test_risk_model_fit_and_validation_threshold_selection():
    model = RiskModel()
    safe = {name: 0.9 for name in model.FEATURE_NAMES}
    unsafe = {name: 0.1 for name in model.FEATURE_NAMES}
    model.fit([safe, safe, unsafe, unsafe], [False, False, True, True])
    risks = [model.predict_risk(safe), model.predict_risk(unsafe)]
    threshold = model.calibrate_threshold(risks, [False, True], max_unsafe_rate=0,
                                          require_zero_unsafe=True)
    assert threshold == risks[0]
    assert model.predict_risk(safe) <= model.threshold
    assert model.predict_risk(unsafe) > model.threshold
    restored = RiskModel.from_state_dict(model.state_dict())
    assert restored.predict_risk(safe) == pytest.approx(model.predict_risk(safe))
    assert restored.threshold == threshold


def test_uncalibrated_semantic_path_fails_closed_to_judge():
    cache = JudgeCache(embed, RiskModel())
    cache.evaluate(request(), lambda _req: score())
    calls = []
    cache.evaluate(request(instruction="please explain gravity"),
                   lambda _req: calls.append(1) or score(4))
    assert calls == [1]
    assert cache.trace[-1]["reason"] == "risk_model_uncalibrated"

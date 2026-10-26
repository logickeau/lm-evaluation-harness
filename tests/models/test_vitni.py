"""Real inference integration tests. Set VITNI_MODEL and VITNI_BINARY to run.

These tests intentionally do not mock the engine or substitute saved responses.
"""

import os
from pathlib import Path

import pytest

from lm_eval.api.instance import Instance
from lm_eval.models.vitni import VitniLM


def request(context, continuation):
    return Instance(
        request_type="loglikelihood", doc={}, arguments=(context, continuation), idx=0
    )


@pytest.fixture(scope="module", params=["cpu", "metal"])
def lm(request):
    model, binary = os.getenv("VITNI_MODEL"), os.getenv("VITNI_BINARY")
    if not model or not binary:
        pytest.skip("set VITNI_MODEL and VITNI_BINARY for real inference tests")
    instance = VitniLM(Path(model), Path(binary), backend=request.param, max_length=8)
    yield instance
    instance.close()


def test_empty_continuation_does_not_invent_work(lm):
    assert lm.loglikelihood([request("", ""), request("The", "")]) == [(0.0, True)] * 2
    assert lm.records[-1]["forward_steps"] == 0


def test_context_boundary_and_unicode(lm):
    assert lm.encode_pair("Hello ", "world") == lm.encode_pair("Hello", " world")
    assert lm.encode_pair("", "café")[1] == lm.tok_encode("café", False)
    with pytest.raises(ValueError, match="token prefix"):
        lm.encode_pair("hel", "lo")


def test_fresh_kv_state_and_order_independence(lm):
    a = lm.score_tokens([1, 9038], [2501, 263])
    first = lm.records[-1]
    lm.score_tokens([1, 9039], [2501, 263])
    changed = lm.records[-1]
    b = lm.score_tokens([1, 9038], [2501, 263])
    last = lm.records[-1]
    assert a == b
    assert first["trace_blake3"] == last["trace_blake3"]
    assert first["trace_blake3"] != changed["trace_blake3"]
    assert first["forward_steps"] == last["forward_steps"] == 3
    assert len(first["target_logprobs"]) == 2
    assert sum(first["target_logprobs"]) == a[0]
    # Pinned real model/engine numerical fixture, verified independently on CPU/Metal.
    assert (
        first["trace_blake3"]
        == "f4c99d9fee6f86329caa4fd15954bac96c6b76cec6a71a91ebb5f740eb70c1cc"
    )


def test_invalid_tokens_and_overflow_fail_explicitly(lm):
    with pytest.raises(ValueError, match="invalid context or token"):
        lm.score_tokens([1], [32000])
    with pytest.raises(ValueError, match="max_length"):
        lm.score_tokens([1] * 9, [2])
    with pytest.raises(ValueError, match="exceeds max length"):
        lm._native.call({"tokens": [1] * 10, "context_len": 1})
    # Process still works after errors; empty continuation has probability one.
    assert lm.score_tokens([1], []) == (0.0, True)


def test_rolling_likelihood_scores_every_token_once(lm):
    text = "a b c d e f g h i j k"
    ids = lm.tok_encode(text, False)
    assert len(ids) > lm.max_length
    before = len(lm.records)
    req = Instance(
        request_type="loglikelihood_rolling", doc={}, arguments=(text,), idx=0
    )
    total = lm.loglikelihood_rolling([req])[0]
    records = lm.records[before:]
    assert [token for r in records for token in r["tokens"][r["context_len"] :]] == ids
    assert max(r["forward_steps"] for r in records) <= lm.max_length
    assert sum(r["logprob"] for r in records) == total


def test_digest_rejection_before_starting_inference():
    model, binary = os.getenv("VITNI_MODEL"), os.getenv("VITNI_BINARY")
    if not model or not binary:
        pytest.skip("real model required")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        VitniLM(model, binary, model_sha256="0" * 64)


def test_unsupported_batching_is_explicit():
    with pytest.raises(ValueError, match="batch_size=1"):
        VitniLM("unused", "unused", batch_size="auto")

"""Tests for persistent tried-token penalty on _TensorizedLMBase."""
from __future__ import annotations

import math
import os

import pytest
import torch
from types import SimpleNamespace

os.environ.setdefault("CSD_RECURRENCE_PENALTY", "0.3")
os.environ.setdefault("CSD_CONSTRAINED_TEMPERATURE", "0.0")

from synthesis.evaluate.benchmarks.common import model_utils as M


def make_lm(instr: str = "EXAMPLE-1 "):
    lm = M._TensorizedLMBase(
        _dafny=None,
        tokenizer=None,
        tokens=["A", "B", "C"],
        tids=[0, 1, 2],
        logits_device="cpu",
    )
    lm.instruction_text = instr
    return lm


def lp(a, b, c):
    return {
        0: SimpleNamespace(logprob=a),
        1: SimpleNamespace(logprob=b),
        2: SimpleNamespace(logprob=c),
    }


P = ["x"]
P2 = ["y"]
LN03 = math.log(0.3)


def test_empty_map_noop_is_byte_identical():
    lm = make_lm()
    lm._finalize_from_logprob_dict(lp(-0.1, -0.5, -2.0))
    before = lm._logits_tensor.clone()
    lm._apply_recurrence_penalty(lm.instruction_text + lm._prefix_text(P))
    assert lm.ChooseNextToken() == "A"
    assert torch.allclose(lm._logits_tensor, before)


def test_penalizing_argmax_diverges_on_regen():
    lm = make_lm()
    key = lm.instruction_text + lm._prefix_text(P)
    lm._finalize_from_logprob_dict(lp(-0.1, -0.5, -2.0))
    assert lm.ChooseNextToken() == "A"
    lm.PenalizeTriedTokenAt(P, "A")
    lm._finalize_from_logprob_dict(lp(-0.1, -0.5, -2.0))
    lm._apply_recurrence_penalty(key)
    assert lm.ChooseNextToken() == "B"
    assert abs(lm._logits_tensor[0].item() - (-0.1 + LN03)) < 1e-4


def test_without_persistent_reapply_regen_reproduces_token():
    lm = make_lm()
    lm._finalize_from_logprob_dict(lp(-0.1, -0.5, -2.0))
    assert lm.ChooseNextToken() == "A"
    lm.PenalizeTriedTokenAt(P, "A")
    lm._finalize_from_logprob_dict(lp(-0.1, -0.5, -2.0))
    assert lm.ChooseNextToken() == "A"


def test_cumulative_penalty_eventually_demotes_confident_token():
    lm = make_lm()
    key = lm.instruction_text + lm._prefix_text(P)

    def fin():
        lm._finalize_from_logprob_dict(lp(0.0, -3.0, -9.0))

    fin()
    assert lm.ChooseNextToken() == "A"
    seq = []
    for _ in range(3):
        lm.PenalizeTriedTokenAt(P, "A")
        fin()
        lm._apply_recurrence_penalty(key)
        seq.append((lm.ChooseNextToken(), round(lm._logits_tensor[0].item(), 3)))
    assert [s[0] for s in seq] == ["A", "A", "B"]


def test_no_cross_prefix_contamination():
    lm = make_lm()
    key_p2 = lm.instruction_text + lm._prefix_text(P2)
    lm.PenalizeTriedTokenAt(P, "A")
    lm._finalize_from_logprob_dict(lp(-0.1, -0.5, -2.0))
    lm._apply_recurrence_penalty(key_p2)
    assert lm.ChooseNextToken() == "A"


def test_penalty_map_resets_on_instruction_text_change():
    lm = make_lm("EXAMPLE-1 ")
    lm.PenalizeTriedTokenAt(P, "A")
    assert len(lm._tried_token_penalties) == 1
    lm.instruction_text = "EXAMPLE-2 "
    lm.PenalizeTriedTokenAt(P, "B")
    keys = list(lm._tried_token_penalties.keys())
    assert keys and all(k.startswith("EXAMPLE-2 ") for k in keys)

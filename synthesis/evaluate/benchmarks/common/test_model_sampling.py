"""Unit tests for LM greedy vs temperature sampling policy."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock

import torch

from synthesis.evaluate.benchmarks.common import model_utils


class _StubLM(model_utils._TensorizedLMBase):
    def __init__(self):
        dafny = MagicMock()
        dafny.Seq = lambda value: value
        tokenizer = MagicMock()
        tokenizer.decode = lambda ids: f"t{ids[0]}"
        super().__init__(dafny, tokenizer, ["a", "b"], [0, 1])
        self._logits_tensor = torch.tensor([0.0, 1.0], dtype=torch.float32)
        self._full_logits = torch.tensor([10.0, -10.0], dtype=torch.float32)
        self._token_str_to_indices = {"a": [0], "b": [1]}


class ModelSamplingPolicyTests(unittest.TestCase):
    def test_reset_task_guidance_clears_guidance_state(self):
        lm = _StubLM()
        lm._task_guidance.append("", "hint")
        lm.ResetTaskGuidance()
        self.assertIsNone(lm._task_guidance.accepted_guidance)

    def test_choose_next_token_greedy_vs_sample(self):
        lm = _StubLM()
        lm._constrained_temperature = 0.0
        self.assertEqual(lm.ChooseNextToken(), "b")
        lm._constrained_temperature = 1.0
        torch.manual_seed(0)
        sampled = {lm.ChooseNextToken() for _ in range(32)}
        self.assertEqual(sampled, {"a", "b"})

    def test_choose_next_token_unconstrained_prefers_high_logit(self):
        lm = _StubLM()
        self.assertEqual(lm.ChooseNextTokenUnconstrained(), "t0")

    def test_constrained_temperature_default_is_greedy(self):
        saved = os.environ.pop("CSD_CONSTRAINED_TEMPERATURE", None)
        try:
            lm = _StubLM()
            self.assertEqual(lm._constrained_temperature, 0.0)
        finally:
            if saved is not None:
                os.environ["CSD_CONSTRAINED_TEMPERATURE"] = saved


if __name__ == "__main__":
    unittest.main()

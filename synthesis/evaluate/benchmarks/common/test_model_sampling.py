"""Unit tests for LM greedy vs temperature-1 sampling policy."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import torch

from synthesis.evaluate.benchmarks.common import model_utils


class _StubLM(model_utils._TensorizedLMBase):
    def __init__(self):
        self._dafny = MagicMock()
        self._dafny.Seq = lambda value: value
        self.tokenizer = MagicMock()
        self.tokenizer.decode = lambda ids: f"t{ids[0]}"
        tokens = ["a", "b"]
        tids = [0, 1]
        self._Tokens = tokens
        self.instruction_text = ""
        self._task_guidance = model_utils._TaskGuidanceState()
        n = len(tokens)
        self.Logits = model_utils._LogitsProxy(n, tids)
        self._logits_device = torch.device("cpu")
        self._logits_tensor = torch.tensor([0.0, 1.0], dtype=torch.float32)
        self._token_ids_tensor = torch.tensor(tids, dtype=torch.long)
        self._full_logits = torch.tensor([2.0, 1.0], dtype=torch.float32)
        self._generate_count = 0
        self._token_id_to_str = {}
        self._last_full_prompt = None
        self._logits_dirty = False
        self._cache_hits = 0
        self._non_deterministic = False
        self._token_str_to_indices = {"a": [0], "b": [1]}


class ModelSamplingPolicyTests(unittest.TestCase):
    def test_reset_task_guidance_clears_non_deterministic(self):
        lm = _StubLM()
        lm.SetNonDeterministic(True)
        lm.ResetTaskGuidance()
        self.assertFalse(lm._non_deterministic)

    def test_choose_next_token_greedy_vs_sample(self):
        lm = _StubLM()
        self.assertEqual(lm.ChooseNextToken(), "b")
        lm.SetNonDeterministic(True)
        torch.manual_seed(0)
        sampled = {lm.ChooseNextToken() for _ in range(32)}
        self.assertEqual(sampled, {"a", "b"})

    def test_choose_next_token_unconstrained_greedy(self):
        lm = _StubLM()
        self.assertEqual(lm.ChooseNextTokenUnconstrained(), "t0")

    def test_chunk_sampling_kwargs(self):
        lm = _StubLM()
        self.assertEqual(lm._chunk_sampling_kwargs(), (0.0, False))
        lm.SetNonDeterministic(True)
        self.assertEqual(lm._chunk_sampling_kwargs(), (1.0, True))


if __name__ == "__main__":
    unittest.main()

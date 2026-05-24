"""Tests for single-model SynCode session reuse."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SYNCODE_ROOT = _REPO / "synthesis" / "evaluate" / "syncode"
_SYNCODE_PKG = _SYNCODE_ROOT / "syncode"
for _p in (str(_SYNCODE_ROOT), str(_SYNCODE_PKG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class TestSyncodeRunSession(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from synthesis.evaluate.syncode_run_session import SyncodeRunSession
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        cls.SyncodeRunSession = SyncodeRunSession

    def test_original_mode_skips_dfa_mask_store(self) -> None:
        session = self.SyncodeRunSession(
            "test",
            device="cpu",
            mode="original",
            quantize=False,
            max_new_tokens=4,
            do_sample=True,
            temperature=1.0,
            opp=False,
        )
        try:
            session.ensure_ready()
            self.assertEqual(len(session._grammar_decoders), 0)
            batch = session.infer("1 + 1 = ")
            self.assertIsInstance(batch, list)
            self.assertGreaterEqual(len(batch), 1)
        finally:
            session.close()

    def test_reuses_one_model_across_two_grammars(self) -> None:
        g1 = "start: NUMBER\n%import common.NUMBER\n"
        g2 = 'start: NUMBER "+" NUMBER\n%import common.NUMBER\n'

        session = self.SyncodeRunSession(
            "test",
            device="cpu",
            mode="grammar_strict",
            quantize=False,
            max_new_tokens=8,
            do_sample=False,
            opp=False,
        )
        try:
            model_ref = session.loaded_model
            session.apply_grammar(g1)
            session.apply_grammar(g2)
            self.assertIs(session.loaded_model, model_ref)
            self.assertEqual(len(session._grammar_decoders), 2)
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()

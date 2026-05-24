"""Tests for vendored SynCode import bootstrap."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
class BrokenSyncodeEditableFinder:
    MAPPING = {"syncode": "/home/aadivyar/CRANE/syncode/syncode"}


class VendoredSyncodeBootstrapTests(unittest.TestCase):
    def test_drops_unreadable_editable_finder(self) -> None:
        from synthesis.evaluate import vendored_syncode

        vendored_syncode._BOOTSTRAPPED = False
        sys.meta_path.insert(0, BrokenSyncodeEditableFinder)
        try:
            vendored_syncode.ensure_vendored_syncode_importable()
            self.assertNotIn(
                BrokenSyncodeEditableFinder,
                sys.meta_path,
                "broken editable finder should be removed",
            )
            syncode_pkg = Path(__file__).resolve().parent / "syncode" / "syncode"
            self.assertTrue(str(syncode_pkg) in sys.path[:4])
        finally:
            vendored_syncode._BOOTSTRAPPED = False
            if BrokenSyncodeEditableFinder in sys.meta_path:
                sys.meta_path.remove(BrokenSyncodeEditableFinder)

    def test_import_syncode_run_session_with_broken_finder(self) -> None:
        sys.meta_path.insert(0, BrokenSyncodeEditableFinder)
        for key in list(sys.modules):
            if key == "syncode" or key.startswith("syncode."):
                del sys.modules[key]
        if "synthesis.evaluate.syncode_run_session" in sys.modules:
            del sys.modules["synthesis.evaluate.syncode_run_session"]
        try:
            from synthesis.evaluate import syncode_run_session

            self.assertTrue(
                str(Path(syncode_run_session.__file__).resolve()).endswith(
                    "syncode_run_session.py"
                )
            )
        finally:
            if BrokenSyncodeEditableFinder in sys.meta_path:
                sys.meta_path.remove(BrokenSyncodeEditableFinder)


if __name__ == "__main__":
    unittest.main()

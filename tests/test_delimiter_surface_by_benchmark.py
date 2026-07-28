"""Each benchmark must say whether it emits visible `<<`/`>>` delimiters, and
the synthesis run must take the answer from the benchmark instead of a
hard-coded dataset name.

Why this matters. Spider runs the token-0-constrained surface, where the whole
answer is grammar-governed and `<<`/`>>` are never emitted at all. But
`run_synthesis.py` only exempted *smiles* from the delimiter requirement, so
Spider runs were still told their delimiter statistics were failure symptoms.
Those statistics are constants on that surface -- "no visible spans" and "no
visible delimiters" are true 100% of the time by design -- so the author AI
read a constant as a symptom and spent seven attempts writing `<<`-detection
branches that can never fire.

The suppression machinery already exists and is covered by
tests/test_delimiter_feedback_gating.py; it is gated on `require_delimiters`.
The only missing piece is routing Spider into it.

Contract:
  * every benchmark exposes `emits_visible_delimiters()`
  * gsm_symbolic emits them (its answers really are `<<...>>` spans)
  * spider does not, while the token-0 surface is on -- and does again when
    that surface is switched off, since the legacy path forces `<<`
  * smiles does not
  * `resolve_require_delimiters` follows the benchmark, and never lets a CLI
    request for delimiters override a surface that cannot produce them
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthesis.evaluate.benchmarks.registry import get_logic


ALL_DATASETS = ("gsm_symbolic", "spider", "smiles")


@pytest.fixture
def token0(monkeypatch):
    """Turn the Spider token-0 (no-delimiter) surface on or off."""

    def _set(enabled: bool) -> None:
        monkeypatch.setenv("SPIDER_TOKEN0_CONSTRAINED", "1" if enabled else "0")

    return _set


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_every_benchmark_declares_its_delimiter_surface(dataset):
    logic = get_logic(dataset)
    assert hasattr(logic, "emits_visible_delimiters"), (
        f"{dataset} must declare whether it emits visible << >> delimiters; "
        "the synthesis run needs this to decide whether delimiter diagnostics "
        "carry any information"
    )
    assert isinstance(logic.emits_visible_delimiters(), bool)


def test_gsm_emits_visible_delimiters():
    # GSM answers genuinely are visible <<...>> spans, so the diagnostics are
    # informative there and must keep flowing to the author.
    assert get_logic("gsm_symbolic").emits_visible_delimiters() is True


def test_spider_emits_no_delimiters_on_the_token0_surface(token0):
    token0(True)
    assert get_logic("spider").emits_visible_delimiters() is False


def test_spider_emits_delimiters_again_on_the_legacy_surface(token0):
    # SPIDER_TOKEN0_CONSTRAINED=0 restores the pre-2026-06-22 path, which does
    # force visible `<<`. The declaration must track the surface, not the
    # dataset name.
    token0(False)
    assert get_logic("spider").emits_visible_delimiters() is True


def test_smiles_emits_no_delimiters():
    assert get_logic("smiles").emits_visible_delimiters() is False


# --- how the synthesis run consumes the declaration -------------------------


def test_resolve_require_delimiters_is_importable():
    from synthesis.evaluate.benchmarks.registry import resolve_require_delimiters  # noqa: F401


def test_spider_does_not_require_delimiters_even_when_cli_asks_for_them(token0):
    """The CLI default is --require-delimiters, so this is the live case.

    A surface that cannot emit delimiters must not be asked for them; otherwise
    the author is handed a constant dressed up as a failure.
    """
    from synthesis.evaluate.benchmarks.registry import resolve_require_delimiters

    token0(True)
    assert resolve_require_delimiters("spider", cli_value=True) is False


def test_smiles_does_not_require_delimiters(token0):
    from synthesis.evaluate.benchmarks.registry import resolve_require_delimiters

    assert resolve_require_delimiters("smiles", cli_value=True) is False


def test_gsm_honours_the_cli_value(token0):
    from synthesis.evaluate.benchmarks.registry import resolve_require_delimiters

    assert resolve_require_delimiters("gsm_symbolic", cli_value=True) is True
    # Opting out stays possible on a surface that *could* emit them.
    assert resolve_require_delimiters("gsm_symbolic", cli_value=False) is False


def test_spider_legacy_surface_honours_the_cli_value(token0):
    from synthesis.evaluate.benchmarks.registry import resolve_require_delimiters

    token0(False)
    assert resolve_require_delimiters("spider", cli_value=True) is True


def test_run_synthesis_no_longer_hardcodes_the_smiles_dataset_name():
    """Guards against the shape of the original bug.

    The old line read:
        require_delimiters=False if args.dataset == "smiles" else args.require_delimiters
    which silently left every other no-delimiter surface (i.e. Spider) asking
    for delimiters. Deriving it per dataset name does not scale and is what
    went wrong; the decision must come from the benchmark.

    This checks only the delimiter argument itself. Other parts of the file may
    legitimately branch on the dataset name -- picking a sample size, say -- and
    this test must not push anyone into contorting those.
    """
    source = (REPO_ROOT / "synthesis/run_synthesis.py").read_text()
    delimiter_lines = [
        line for line in source.splitlines() if "require_delimiters=" in line
    ]
    assert delimiter_lines, "expected run_synthesis.py to still pass require_delimiters"
    for line in delimiter_lines:
        assert "smiles" not in line and "spider" not in line, (
            f"the delimiter surface must be read from the benchmark, not from a "
            f"hard-coded dataset name; found: {line.strip()!r}"
        )
    assert any("resolve_require_delimiters" in line for line in delimiter_lines), (
        "require_delimiters must be resolved through the benchmark registry"
    )

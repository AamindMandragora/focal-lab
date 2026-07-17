"""Tests for plan_kv_reuse — the decision function for the HF KV-cached decode path.

Contract: plan_kv_reuse(cached_ids, new_ids) -> (keep_len, feed_ids) where
  - cached_ids[:keep_len] == new_ids[:keep_len]  (the kept cache matches)
  - keep_len + len(feed_ids) == len(new_ids)     (kept + fed covers the prompt)
  - feed_ids is never empty when new_ids is non-empty (the forward must
    return logits, so at least one token is always fed)
"""
import importlib.util
import pathlib

# Load by file path: importing the package would pull in torch (unavailable
# in the local test env), and kv_reuse itself is deliberately torch-free.
_KV_REUSE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "synthesis" / "evaluate" / "benchmarks" / "common" / "kv_reuse.py"
)
_spec = importlib.util.spec_from_file_location("kv_reuse", _KV_REUSE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
plan_kv_reuse = _mod.plan_kv_reuse


def check_invariants(cached, new, keep, feed):
    assert cached[:keep] == new[:keep]
    assert keep + len(feed) == len(new)
    if new:
        assert len(feed) >= 1
    assert feed == new[keep:]


def test_empty_cache_prefills_everything():
    # First call of an example: no cache -> feed the whole prompt (CRANE step 0).
    keep, feed = plan_kv_reuse([], [10, 11, 12])
    assert (keep, feed) == (0, [10, 11, 12])


def test_extension_by_one_token_feeds_only_that_token():
    # The common case: the chosen token was appended -> single-token decode,
    # exactly CRANE's cached kernel path.
    keep, feed = plan_kv_reuse([10, 11, 12], [10, 11, 12, 13])
    assert (keep, feed) == (3, [13])
    check_invariants([10, 11, 12], [10, 11, 12, 13], keep, feed)


def test_extension_by_several_tokens():
    keep, feed = plan_kv_reuse([10, 11], [10, 11, 12, 13, 14])
    assert (keep, feed) == (2, [12, 13, 14])


def test_identical_ids_recomputes_last_token():
    # Dirty-logits recompute: same prompt, must re-run the last token so the
    # forward returns fresh logits.
    keep, feed = plan_kv_reuse([10, 11, 12], [10, 11, 12])
    assert (keep, feed) == (2, [12])


def test_shrink_rollback_keeps_common_prefix():
    # Rollback to span entry: prompt got shorter. Keep what still matches,
    # feed at least the last token.
    keep, feed = plan_kv_reuse([10, 11, 12, 13, 14], [10, 11, 12])
    assert (keep, feed) == (2, [12])
    check_invariants([10, 11, 12, 13, 14], [10, 11, 12], keep, feed)


def test_divergent_suffix_keeps_common_prefix():
    # Retokenization merged a boundary token: prefix matches, tail differs.
    cached = [10, 11, 99, 98]
    new = [10, 11, 55, 56, 57]
    keep, feed = plan_kv_reuse(cached, new)
    assert (keep, feed) == (2, [55, 56, 57])
    check_invariants(cached, new, keep, feed)


def test_completely_different_ids_resets():
    keep, feed = plan_kv_reuse([1, 2, 3], [7, 8, 9])
    assert (keep, feed) == (0, [7, 8, 9])


def test_single_token_prompt():
    keep, feed = plan_kv_reuse([], [5])
    assert (keep, feed) == (0, [5])
    keep, feed = plan_kv_reuse([5], [5])
    assert (keep, feed) == (0, [5])


def test_empty_new_ids():
    keep, feed = plan_kv_reuse([1, 2], [])
    assert (keep, feed) == (0, [])

"""KV-cache reuse planning for the HuggingFace backend's decode path.

Baseline parity context: CRANE/IterGen decode with a KV cache — after the
prompt prefill, every step feeds ONE token through the model. Our harness used
to re-run the full prompt every step, which lands on a different GPU kernel
path; at near-tie tokens the argmax flips (probe 2026-07-02: cached replica
reached `<<` on 39/49 GSM examples vs ~24/49 for the full-re-forward path).
This module decides, per GenerateLogits call, how much of the existing cache
still applies and which tokens must be fed.

Kept torch-free so it stays unit-testable without a GPU stack.
"""


def plan_kv_reuse(cached_ids, new_ids):
    """Decide how to reuse a KV cache built over cached_ids for a forward
    that needs next-token logits after new_ids.

    Inputs: two token-id sequences (list-like of ints).
    Output: (keep_len, feed_ids) with these guarantees:
      - cached_ids[:keep_len] == new_ids[:keep_len]  (kept cache is valid)
      - feed_ids == new_ids[keep_len:]               (kept + fed == prompt)
      - feed_ids is non-empty whenever new_ids is    (a forward always runs,
        so logits are always produced)
    Algorithm: longest common prefix, capped at len(new_ids) - 1 so at least
    the final token is always fed through the model.
    """
    new_ids = list(new_ids)
    if not new_ids:
        return 0, []
    limit = min(len(cached_ids), len(new_ids) - 1)
    keep = 0
    while keep < limit and cached_ids[keep] == new_ids[keep]:
        keep += 1
    return keep, new_ids[keep:]

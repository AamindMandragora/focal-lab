"""TDD for symbol-boundary grounding rollback (Task #25/#27).

The bug being fixed: RegenerateUnitOnGroundingFailure used SpanGrounded (a bool)
and penalized the FIRST token of the rolled-back unit. For SQL the unit is the
whole query, so it penalized the query's first token (usually "SELECT") — which
can never correct an out-of-schema identifier deep in the query. Result on the
live re-eval: grounded_True=0, grounding never recovers.

The fix: locate the FIRST out-of-schema identifier's token index and penalize
THERE. These tests pin the pure mapping logic (no model/tokenizer needed):
  - _candidate_identifiers_with_pos(text) -> [(name_lower, char_offset), ...]
  - _first_ungrounded_token_idx(token_strs, support) -> (found, idx)

Run on focal:
  export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}
  cd <repo-root> && python synthesis/evaluate/benchmarks/common/../../../../  # see launcher
"""
from synthesis.evaluate.benchmarks.common.model_utils import (
    _candidate_identifiers,
    _candidate_identifiers_with_pos,
    _first_ungrounded_token_idx,
)


def _check(name, got, want):
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    return ok


def main():
    results = []

    # --- 1. THE BUG: a deep out-of-schema name must NOT penalize token 0. ---
    # text = "SELECT name FROM badtable"; badtable starts at char 17 (token 3).
    support = {"singer", "name", "age"}
    tokens = ["SELECT", " name", " FROM", " badtable"]
    results.append(_check(
        "deep out-of-schema name -> its own token (not 0)",
        _first_ungrounded_token_idx(tokens, support), (True, 3)))

    # --- 2. Fully grounded query -> not found. ---
    support2 = {"singer", "name", "age"}
    tokens2 = ["SELECT", " name", " FROM", " singer"]
    results.append(_check(
        "fully grounded -> (False, 0)",
        _first_ungrounded_token_idx(tokens2, support2), (False, 0)))

    # --- 3. No support set (non-SQL / unparsable schema) -> no-op. ---
    results.append(_check(
        "empty support -> (False, 0)",
        _first_ungrounded_token_idx(["SELECT", " x"], set()), (False, 0)))

    # --- 4. Offset preservation across a quoted literal (the blanking choice). ---
    # If quoted regions were COLLAPSED to one space, "bar" would map to the wrong
    # token. With equal-length blanking, "bar" stays at offset 11 -> token 1.
    support4 = {"foo"}
    tokens4 = ["x = 'aaaa' ", "bar"]  # text = "x = 'aaaa' bar"
    results.append(_check(
        "bad ident after quoted literal -> correct token (offset preserved)",
        _first_ungrounded_token_idx(tokens4, support4), (True, 1)))

    # --- 5. Quoted value is NOT treated as an identifier. ---
    support5 = {"singer", "name", "city"}
    tokens5 = ["SELECT name FROM singer WHERE city = 'london'"]
    results.append(_check(
        "quoted value 'london' is not an out-of-schema identifier",
        _first_ungrounded_token_idx(tokens5, support5), (False, 0)))

    # --- 6. First bad identifier wins when several are out-of-schema. ---
    support6 = {"name"}
    tokens6 = ["SELECT", " name", ",", " badA", ",", " badB", " FROM", " t"]
    # badA starts at char len("SELECT name, ")=13 -> token 3.
    results.append(_check(
        "first out-of-schema identifier wins",
        _first_ungrounded_token_idx(tokens6, support6), (True, 3)))

    # --- 7. _candidate_identifiers_with_pos is signal-identical to the bool path. ---
    sample = "SELECT T1.name, T2.age FROM singer AS T1 JOIN concert AS T2 WHERE city = 'NY'"
    names_pos = [n for n, _ in _candidate_identifiers_with_pos(sample)]
    names_bool = _candidate_identifiers(sample)
    results.append(_check(
        "with_pos names == _candidate_identifiers names (same signal)",
        names_pos, names_bool))

    print(f"\n{sum(results)}/{len(results)} passed")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()

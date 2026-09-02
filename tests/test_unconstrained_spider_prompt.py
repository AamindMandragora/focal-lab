"""Red/green test: prompt type for the spider unconstrained adapter.

Run from the repo root on focal:
    python tests/test_unconstrained_spider_prompt.py

This test exercises the exact code path that run_unconstrained_spider_adapter uses
to build its prompt for each example.

RED before fix  : AssertionError — "chain_of_thought" returns list[dict], not str
GREEN after fix : OK — "evaluator_default" returns a plain str

The test imports _legacy_benchmark_prompt and calls it with the spider eval_logic
in exactly the same way the adapter does, using a minimal fake example (only the
fields that the prompt builder reads: db_id, db_info, question).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- configuration -----------------------------------------------------------
# This constant must match what run_unconstrained_spider_adapter actually passes
# to _legacy_benchmark_prompt.  Before the fix it was "chain_of_thought" (broken);
# after the fix it is "evaluator_default" (returns str).
ADAPTER_PROFILE = "evaluator_default"

_FAKE_EXAMPLE = {
    "db_id": "concert_singer",
    "db_info": "# singer ( singer_id , name , country , age )",
    "question": "How many singers do we have?",
    "query": "SELECT count(*) FROM singer",
}
# -----------------------------------------------------------------------------


def test_adapter_prompt_is_str() -> None:
    """The prompt the spider unconstrained adapter builds must be a plain str.

    llm.generate([prompt], sp) in vLLM rejects a list[dict]; it must receive a str.
    This test uses the same profile constant (ADAPTER_PROFILE) that the adapter code uses,
    so if someone ever changes the adapter back to "chain_of_thought" this test turns red again.
    """
    from synthesis.evaluate.benchmarks.sql_spider import eval_logic as spider_logic
    from synthesis.evaluate.run_legacy_fixed_strategy import _legacy_benchmark_prompt

    prompt = _legacy_benchmark_prompt(spider_logic, object(), _FAKE_EXAMPLE, ADAPTER_PROFILE)

    assert isinstance(prompt, str), (
        f"Expected prompt to be a str for vLLM compatibility, "
        f"but got {type(prompt).__name__}.\n"
        f"Value: {prompt!r}\n\n"
        "If the adapter is using 'chain_of_thought', that calls format_prompt_chain_of_thought "
        "which returns list[dict] — fix the adapter to use 'evaluator_default' instead."
    )
    assert "How many singers" in prompt, f"Prompt missing question text: {prompt!r}"
    print(f"PASS: ADAPTER_PROFILE={ADAPTER_PROFILE!r} -> str ({len(prompt)} chars)")
    print(f"Prompt preview: {prompt[:300]!r}")


if __name__ == "__main__":
    test_adapter_prompt_is_str()
    print("\nAll tests passed.")

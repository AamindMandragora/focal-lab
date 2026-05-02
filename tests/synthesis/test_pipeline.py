"""
Quick pipeline smoke test for the Python-first CSD synthesis path.

This verifies that we can:
- generate a Python strategy body
- inject it into `generation/csd/GeneratedAgentTemplate.py`
- verify it through the transpiler
- execute the original generated Python strategy
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.getcwd())

from generation.generator import StrategyGenerator
from verification.verifier import DafnyVerifier
from synthesis.runner import StrategyRunner


def test_synthesis():
    assert run_synthesis_smoke()


def run_synthesis_smoke():
    repo_root = Path(os.getcwd())
    dafny_bin = os.environ.get("DAFNY", str(repo_root / "dafny" / "dafny"))
    verifier = DafnyVerifier(dafny_path=dafny_bin)
    runner = StrategyRunner()

    strategy_code = """
# CSD_RATIONALE_BEGIN
# Simple smoke test for the Python-to-Dafny pipeline with explicit delimiter emission and single-prefix constrained answer content.
# CSD_RATIONALE_END
# CSD_PROOF_SKETCH_BEGIN
# The loop keeps the standard helper/library invariants and decreases stepsLeft.
# AppendLeftDelimiter and AppendRightDelimiter consume one step and preserve the budget invariant.
# AppendConstrainedStep is called only under CanConstrain, so parser validity of the active suffix is preserved.
# Phase monotonically moves from open to constrained decoding to closed, so the loop makes progress or breaks.
# CSD_PROOF_SKETCH_END
phase = 0
answer_tokens = 0
close_attempts = 0
# invariant lm.ValidTokensIdsLogits()
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# invariant 0 <= answer_tokens
# invariant 0 <= close_attempts
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        answer_tokens = answer_tokens + 1
    elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        close_attempts = close_attempts + 1
        phase = 3
    else:
        break
"""
    generator = StrategyGenerator()
    full_code = generator.inject_strategy(strategy_code)

    print("Testing strategy verification...")
    v_result = verifier.verify(full_code)
    if not v_result.success:
        print("Verification failed:", v_result.get_error_summary())
        return False
    print("Verification successful!")

    print("Testing strategy execution...")
    with tempfile.TemporaryDirectory() as tmpdir:
        python_path = Path(tmpdir) / "GeneratedCSD.py"
        python_path.write_text(full_code, encoding="utf-8")
        r_result = runner.run_python_native(python_path)
    if not r_result.success:
        print("Execution failed:", r_result.get_error_summary())
        return False
    print(f"Execution successful! Output length: {len(r_result.output or [])} tokens, steps used: {r_result.cost}")
    return True


if __name__ == "__main__":
    success = run_synthesis_smoke()
    if success:
        print("\nPipeline verification PASSED")
        sys.exit(0)
    else:
        print("\nPipeline verification FAILED")
        sys.exit(1)

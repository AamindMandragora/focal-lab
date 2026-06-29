from __future__ import annotations

import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _prompt_helper_refs() -> set[str]:
    prompt_source = (REPO_ROOT / "synthesis/generate/prompts.py").read_text()
    return set(
        re.findall(
            r"\b(?:helpers|CSDHelpers)\.([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            prompt_source,
        )
    )


def _dafny_helper_defs() -> set[str]:
    dafny_source = (
        REPO_ROOT / "synthesis/verify/library/VerifiedAgentSynthesis.dfy"
    ).read_text()
    return set(
        re.findall(
            r"(?:^|\n)\s*(?:static\s+)?(?:method|function)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            dafny_source,
        )
    )


def _feedback_helper_classifications() -> set[str]:
    feedback_source = (REPO_ROOT / "synthesis/evaluate/feedback_loop.py").read_text()
    module = ast.parse(feedback_source)
    classified: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef) or node.name != "SynthesisPipeline":
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id in {"NON_PRUNABLE_HELPERS", "PRUNABLE_HELPERS"}
                ):
                    classified.update(ast.literal_eval(stmt.value))
    return classified


def _core_lm_helpers() -> set[str]:
    return {
        "Contains",
        "RenderPrefix",
        "GenerateLogits",
        "ChooseNextToken",
        "ChooseNextTokenUnconstrained",
        "GenerateUnconstrainedChunk",
        "MaskValidNextAndEos",
        "BoostValidNextAndEos",
        "IdToToken",
        "TokenToId",
        "TokenToIdRecursive",
        "IdToLogit",
        "TokenToLogit",
        "TokensToLogits",
        "IdsToLogits",
        "MaskToken",
        "MaskTokens",
        "MaskTokensExcept",
        "IsMasked",
        "HasUnmaskedToken",
        "IsValidPrefix",
        "IsCompletePrefix",
        "IsDeadPrefix",
        "ValidNextTokenCount",
        "ValidNextToken",
        "ValidNextTokens",
        "ParseG",
    }


def test_prompt_exposed_helpers_exist_in_dafny_library():
    missing = _prompt_helper_refs() - _dafny_helper_defs()
    assert not missing


def test_prompt_exposed_helpers_are_classified_for_feedback_policy():
    missing = _prompt_helper_refs() - _feedback_helper_classifications()
    assert not missing


def test_core_lm_helpers_are_classified_for_feedback_policy():
    missing = _core_lm_helpers() - _feedback_helper_classifications()
    assert not missing

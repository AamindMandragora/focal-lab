"""Author-menu contract for useful verified core APIs.

These APIs already exist in the Dafny library.  This test checks that the
strategy author is shown their exact names and the rendered-text behavior that
avoids assuming one visible delimiter always equals one tokenizer token.
"""

import importlib.util
import pathlib
import re


_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[2]
_PROMPTS_PATH = _REPO / "generate" / "prompts.py"
if not _PROMPTS_PATH.exists():
    _PROMPTS_PATH = _REPO / "synthesis" / "generate" / "prompts.py"
_LIBRARY_PATH = _REPO / "verify" / "library" / "VerifiedAgentSynthesis.dfy"
if not _LIBRARY_PATH.exists():
    _LIBRARY_PATH = (
        _REPO / "synthesis" / "verify" / "library" / "VerifiedAgentSynthesis.dfy"
    )


def _load_prompts():
    spec = importlib.util.spec_from_file_location(
        "_prompts_author_visible_core_api", _PROMPTS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rendered_text_predicates_are_visible_to_the_author():
    reference = _load_prompts()._build_tool_reference_block(None)

    assert "Contains(s, sub)" in reference
    assert "RenderPrefix(prefix)" in reference
    assert "RenderedEndsWith(prefix, suffix)" in reference


def test_delimiter_guidance_covers_space_prefixed_and_split_tokens():
    reference = _load_prompts()._build_tool_reference_block(None)

    assert 'RenderedEndsWith(generated, "<<")' in reference
    assert '" <<"' in reference
    assert "split across tokens" in reference


def test_exact_logit_reader_names_are_visible_to_the_author():
    reference = _load_prompts()._build_tool_reference_block(None)

    for name in ("IdToLogit", "TokenToLogit", "TokensToLogits", "IdsToLogits"):
        assert name in reference


def test_verified_examples_do_not_use_exact_delimiter_token_identity():
    prompt = _load_prompts().VERIFIED_EXAMPLES
    exact_token_branch = re.compile(
        r'^\s*(?:}\s*else\s+)?if\s+next\s*==\s*"(?:<<|>>)"',
        re.MULTILINE,
    )

    assert not exact_token_branch.search(prompt)
    assert 'RenderedEndsWith([next], "<<")' not in prompt
    assert 'RenderedEndsWith(generated, "<<")' in prompt


def test_managed_span_helpers_use_rendered_prefix_for_delimiter_entry():
    library = _LIBRARY_PATH.read_text(encoding="utf-8")

    assert 'if next == "<<"' not in library
    assert library.count('RenderedEndsWith(generatedOut, "<<")') >= 1
    assert library.count('RenderedEndsWith(generated, "<<")') >= 2


def test_dead_end_avoiding_step_is_visible_to_the_author():
    prompts = _load_prompts()
    reference = prompts._build_tool_reference_block(None)

    assert "DeadEndAvoidingStep" in prompts._ALL_HELPER_NAMES
    assert (
        "helpers.DeadEndAvoidingStep(lm, parser, prompt, generated, "
        "eosToken, maxRetries)" in reference
    )
    assert (
        "helpers.DeadEndAvoidingStep(lm, parser, prompt, currentConstrained, "
        "eosToken, maxRetries)" not in reference
    )


def test_prefix_budget_documentation_covers_equal_budget_case():
    reference = _load_prompts()._build_tool_reference_block(None)

    assert "When `prefixBudget < maxSteps`" in reference
    assert "When `prefixBudget == maxSteps`" in reference


def test_system_and_refinement_prompts_forbid_exact_delimiter_token_checks():
    prompts = _load_prompts()
    required_rule = "Never detect a visible delimiter with exact token equality"

    assert required_rule in prompts.SYSTEM_PROMPT
    assert required_rule in prompts.EVALUATION_FAILURE_REFINEMENT_PROMPT


def test_grounding_helper_documentation_matches_verified_implementation():
    reference = _load_prompts()._build_tool_reference_block(None)
    start = reference.index("- `helpers.RegenerateUnitOnGroundingFailure(")
    end = reference.index("- `helpers.PrefixAppearsInPrompt(", start)
    block = reference[start:end]

    assert "eosToken, budget, maxRetries, maxRollbackBudget)" in block
    assert "CompletedSchemaSymbolCount" in block
    assert "FirstUngroundedIdentifierTokenIdx" in block
    assert "its own token position" in block
    assert "Cost: at most +`budget` token-steps." in block
    assert "maxStepsPerUnit" not in block
    assert "(maxRetries + 1)" not in block


def test_resemblance_range_is_part_of_the_verified_contract():
    library = _LIBRARY_PATH.read_text(encoding="utf-8")

    assert library.count(
        "ensures 0.0 <= score <= 1.0"
    ) >= 2


def test_visible_delimiter_extraction_has_no_unwired_extern():
    library = _LIBRARY_PATH.read_text(encoding="utf-8")

    assert "ExtractContentExtern" not in library
    assert "static method FindSubstring" in library
    assert "static method ExtractContentBetweenDelimiters" in library


def test_delimiter_search_is_iterative_for_long_generated_text():
    library = _LIBRARY_PATH.read_text(encoding="utf-8")
    start = library.index("static method FindSubstring(")
    end = library.index("static method ExtractContentBetweenDelimiters(", start)
    block = library[start:end]

    assert block.count("FindSubstring(") == 1
    assert "while index + |needle| <= |input|" in block


def test_speculative_rollout_contract_proves_restoration_and_exact_status():
    library = _LIBRARY_PATH.read_text(encoding="utf-8")
    start = library.index("method SpeculativeConstrainedRollout(")
    end = library.index("// CRANE-style generation", start)
    block = library[start:end]

    assert (
        "forall i :: 0 <= i < lm.Logits.Length ==> "
        "lm.Logits[i] == old(lm.Logits[i])"
    ) in block
    assert "hitComplete == parser.IsCompletePrefix(candidatePrefix)" in block
    assert "hitEos ==> !hitComplete" in block
    assert "stepsUsed == |candidateTokens| + (if hitEos then 1 else 0)" in block


def test_speculative_rollout_menu_explains_commit_and_status_semantics():
    reference = _load_prompts()._build_tool_reference_block(None)
    start = reference.index("- `helpers.SpeculativeConstrainedRollout(")
    end = reference.index("### Parser queries", start)
    block = reference[start:end]

    assert "restore the entry logits exactly" in block
    assert "`hitComplete` exactly reports" in block
    assert "`hitEos` means EOS consumed one step" in block

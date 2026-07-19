from __future__ import annotations

import pytest

from synthesis.evaluate.benchmarks import prompt_profiles
from synthesis.evaluate.benchmarks.prompt_profiles import (
    prompt_profile_for_strategy,
    render_strategy_prompt,
)
from synthesis.evaluate.benchmarks.smiles import eval_logic as smiles_logic
from synthesis.evaluate.benchmarks.sql_spider import eval_logic as spider_logic
from synthesis.evaluate.run_legacy_fixed_strategy import _fixed_csd_prompt


SPIDER_EXAMPLE = {
    "db_id": "concert_singer",
    "db_info": "# singer ( singer_id , name , country , age )",
    "question": "How many singers do we have?",
}

SPIDER_ITERGEN_PROMPT = (
    "db_id: concert_singer\n"
    "db_info: # singer ( singer_id , name , country , age )\n"
    "question: How many singers do we have? Only output the SQL quey. \n"
    "SQL:"
)

SMILES_EXAMPLE = {
    "prompt": "Generate one molecule in the acrylates class.   ",
}

SMILES_DIRECT_PROMPT = (
    "Generate one molecule in the acrylates class.\n\n"
    "Return exactly one line containing only the SMILES string "
    "(example: CC(=O)OC=C).\n"
    "Molecule: "
)


def test_sql_gcd_and_itergen_share_exact_itergen_prompt() -> None:
    assert prompt_profile_for_strategy("spider", "gcd") == "direct"
    assert prompt_profile_for_strategy("spider", "itergen") == "direct"
    assert render_strategy_prompt("spider", "gcd", SPIDER_EXAMPLE) == SPIDER_ITERGEN_PROMPT
    assert render_strategy_prompt("spider", "itergen", SPIDER_EXAMPLE) == SPIDER_ITERGEN_PROMPT
    assert spider_logic.format_prompt_expression_only(None, SPIDER_EXAMPLE) == SPIDER_ITERGEN_PROMPT


def test_sql_crane_uses_cot_version_of_itergen_prompt() -> None:
    prompt = render_strategy_prompt("spider", "crane", SPIDER_EXAMPLE)

    assert prompt_profile_for_strategy("spider", "crane") == "chain_of_thought"
    assert prompt.startswith(
        "db_id: concert_singer\n"
        "db_info: # singer ( singer_id , name , country , age )\n"
        "question: How many singers do we have?"
    )
    assert "Think step by step about the tables, joins, and filters." in prompt
    assert "wrapped in << >>" in prompt
    assert prompt.endswith("\nSQL:")
    assert spider_logic.format_prompt_chain_of_thought(None, SPIDER_EXAMPLE) == prompt


def test_smiles_gcd_and_itergen_share_one_direct_profile() -> None:
    assert prompt_profile_for_strategy("smiles", "gcd") == "direct"
    assert prompt_profile_for_strategy("smiles", "itergen") == "direct"
    assert render_strategy_prompt("smiles", "gcd", SMILES_EXAMPLE) == SMILES_DIRECT_PROMPT
    assert render_strategy_prompt("smiles", "itergen", SMILES_EXAMPLE) == SMILES_DIRECT_PROMPT
    assert smiles_logic.format_prompt_expression_only(None, SMILES_EXAMPLE) == SMILES_DIRECT_PROMPT


def test_smiles_crane_uses_cot_profile_from_same_yaml() -> None:
    prompt = render_strategy_prompt("smiles", "crane", SMILES_EXAMPLE)

    assert prompt_profile_for_strategy("smiles", "crane") == "chain_of_thought"
    assert "Think step by step about how to satisfy the structural constraints" in prompt
    assert prompt.endswith("\nMolecule: ")
    assert smiles_logic.format_prompt_chain_of_thought(None, SMILES_EXAMPLE) == prompt


def test_every_fixed_csd_runtime_route_resolves_through_strategy_mapping() -> None:
    for strategy in ("gcd", "itergen"):
        assert _fixed_csd_prompt(spider_logic, None, SPIDER_EXAMPLE, strategy) == SPIDER_ITERGEN_PROMPT
        assert _fixed_csd_prompt(smiles_logic, None, SMILES_EXAMPLE, strategy) == SMILES_DIRECT_PROMPT

    sql_crane = _fixed_csd_prompt(spider_logic, None, SPIDER_EXAMPLE, "crane")
    smiles_crane = _fixed_csd_prompt(smiles_logic, None, SMILES_EXAMPLE, "crane")
    assert "Think step by step" in sql_crane
    assert "Think step by step" in smiles_crane


def test_loader_rejects_equal_gcd_itergen_mapping_that_is_not_direct(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "bad.yaml").write_text(
        """\
dataset: bad
strategy_profiles:
  gcd: shared_wrong
  itergen: shared_wrong
  crane: chain_of_thought
profiles:
  shared_wrong:
    template: direct
  chain_of_thought:
    template: cot
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(prompt_profiles, "_PROFILE_DIR", tmp_path)
    monkeypatch.setitem(prompt_profiles._DATASET_FILES, "bad", "bad.yaml")
    prompt_profiles._load_prompt_config.cache_clear()

    try:
        with pytest.raises(ValueError, match="must use the direct prompt profile"):
            prompt_profile_for_strategy("bad", "gcd")
    finally:
        prompt_profiles._load_prompt_config.cache_clear()

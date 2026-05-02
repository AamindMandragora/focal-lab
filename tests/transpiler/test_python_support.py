import ast
from pathlib import Path

from generation.csd.VerifiedAgentSynthesis import CSDHelpers, LM, Parser
from verification.transpiler.transpiler import (
    _translate_compare,
    _translate_expr,
    _translate_stmt_list,
    transpile_contract_library,
)


def test_translate_isinstance_str_to_true():
    expr = ast.parse("isinstance(next_token, str)", mode="eval").body
    assert _translate_expr(expr) == "true"


def test_translate_str_identity():
    expr = ast.parse("str(next_token)", mode="eval").body
    assert _translate_expr(expr) == "next_token"


def test_translate_is_not_none_to_true():
    expr = ast.parse("next_token is not None", mode="eval").body
    assert _translate_compare(expr, None) == "true"


def test_translate_is_none_to_false():
    expr = ast.parse("next_token is None", mode="eval").body
    assert _translate_compare(expr, None) == "false"


def test_translate_token_not_in_string_literal():
    expr = ast.parse('next_token not in "()"', mode="eval").body
    assert _translate_compare(expr, None) == '(next_token != "(" && next_token != ")")'


def test_translate_token_in_tuple_literal():
    expr = ast.parse("next_token in ('+', '-', '*', '/')", mode="eval").body
    translated = _translate_compare(expr, None)
    assert translated == '(next_token == "+" || next_token == "-" || next_token == "*" || next_token == "/")'


def test_translate_newline_string_literal_escapes_for_dafny():
    expr = ast.parse('last_token == "\\n"', mode="eval").body
    translated = _translate_compare(expr, None)

    assert translated == 'last_token == "\\n"'
    assert "\n" not in translated.replace("\\n", "")


def test_translate_isalpha_predicate():
    expr = ast.parse("answer[-1].isalpha()", mode="eval").body
    translated = _translate_expr(expr)
    assert "forall i ::" in translated
    assert "'a'" in translated
    assert "'Z'" in translated


def test_translate_isdigit_predicate():
    expr = ast.parse("answer[-1].isdigit()", mode="eval").body
    translated = _translate_expr(expr)
    assert "forall i ::" in translated
    assert "'0'" in translated
    assert "'9'" in translated


def test_translate_isnumeric_predicate():
    expr = ast.parse("answer[-1].isnumeric()", mode="eval").body
    translated = _translate_expr(expr)
    assert "forall i ::" in translated
    assert "'0'" in translated
    assert "'9'" in translated


def test_translate_list_append_as_sequence_update():
    expr = ast.parse("free_form_buffer.append(next_token)", mode="eval").body
    translated = _translate_expr(expr)
    assert translated == "free_form_buffer := (free_form_buffer + [next_token])"


def test_translate_mycsd_none_initializers_to_defaults():
    source = "next_token = None\nnew_steps = None\n"
    stmts = ast.parse(source).body
    translated = _translate_stmt_list(
        stmts,
        current_class=None,
        source_lines=source.splitlines(),
        declared=set(),
        return_names=[],
        method_name="MyCSDStrategy",
        indent=0,
    )

    assert translated[0] == "var next_token := eosToken;"
    assert translated[1] == "var new_steps := stepsLeft;"


def test_translate_break_statement():
    source = "while keep_going:\n    break\n"
    while_stmt = ast.parse(source).body[0]
    translated = _translate_stmt_list(
        while_stmt.body,
        current_class=None,
        source_lines=source.splitlines(),
        declared=set(),
        return_names=[],
        method_name="MyCSDStrategy",
        indent=0,
    )

    assert translated == ["break;"]


def test_translate_while_with_specs_before_setup_lines():
    source = """# invariant lm.ValidTokensIdsLogits()
# invariant parser.IsValidPrefix(helpers.LongestValidSuffix(generated))
# decreases stepsLeft
# Initialize loop state
phase = 0
answer_tokens = 0
while stepsLeft > 0 and not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
    answer_tokens = answer_tokens + 1
"""
    stmts = ast.parse(source).body
    translated = _translate_stmt_list(
        stmts,
        current_class=None,
        source_lines=source.splitlines(),
        declared=set(),
        return_names=[],
        method_name="MyCSDStrategy",
        indent=0,
    )

    while_line = "while ((stepsLeft > 0) && (!parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))))"
    assert while_line in translated
    while_index = translated.index(while_line)
    assert translated[while_index + 1] == "  invariant lm.ValidTokensIdsLogits()"
    assert translated[while_index + 2] == "  invariant parser.IsValidPrefix(helpers.LongestValidSuffix(generated))"
    assert translated[while_index + 3] == "  decreases stepsLeft"


def test_translate_any_isdigit_over_token_uses_index_quantifier():
    expr = ast.parse("any(char.isdigit() for char in next_token)", mode="eval").body
    translated = _translate_expr(expr)

    assert "exists char_idx ::" in translated
    assert "next_token[char_idx]" in translated
    assert "char in next_token" not in translated


def test_translate_mixed_any_predicates_parenthesizes_quantifiers():
    expr = ast.parse(
        "isinstance(next_token, str) and (any(c.isalpha() for c in next_token) or any(c.isdigit() for c in next_token))",
        mode="eval",
    ).body
    translated = _translate_expr(expr)

    assert translated.startswith("(")
    assert "((exists c_idx ::" in translated
    assert ") || ((exists c_idx ::" in translated or ") || ((exists c_idx_2 ::" in translated
    assert "c in next_token" not in translated


def test_translate_any_isalpha_over_token_uses_char_literals():
    expr = ast.parse("any(char.isalpha() for char in next_token)", mode="eval").body
    translated = _translate_expr(expr)

    assert "'a'" in translated
    assert "'Z'" in translated
    assert '"a"' not in translated


def test_translate_startswith_tuple_literal():
    expr = ast.parse("next_token.startswith(('n', 'x'))", mode="eval").body
    translated = _translate_expr(expr)

    assert '|next_token| >= |"n"|' in translated
    assert 'next_token[..|"n"|] == "n"' in translated
    assert '||' in translated


def test_translate_tuple_literal_assignment():
    source = "next_token, new_steps = None, 0\n"
    stmts = ast.parse(source).body
    translated = _translate_stmt_list(
        stmts,
        current_class=None,
        source_lines=source.splitlines(),
        declared=set(),
        return_names=[],
        method_name="MyCSDStrategy",
        indent=0,
    )

    assert translated == ["var next_token := eosToken;", "var new_steps := 0;"]


class _DummyParser(Parser):
    def IsValidPrefix(self, prefix):
        return all(token in {"a", "b"} for token in prefix)

    def IsCompletePrefix(self, prefix):
        return False

    def ValidNextTokens(self, prefix):
        return ["a", "b"]


class _DummyLM(LM):
    def __init__(self):
        super().__init__()
        self.Tokens = ["a", "b", "<eos>", "<<", ">>", " <<", " >>"]
        self.Ids = list(range(len(self.Tokens)))
        self.Logits = [0.0] * len(self.Tokens)
        self.last_input = None

    def GenerateLogits(self, input):
        self.last_input = list(input)
        base = {
            "a": 1.0,
            "b": 0.0,
            "<eos>": -5.0,
            "<<": -10.0,
            ">>": -10.0,
            " <<": -10.0,
            " >>": -10.0,
        }
        self.Logits = [base[token] for token in self.Tokens]


def test_split_prefix_adaptive_step_biases_valid_group_and_uses_clean_context():
    lm = _DummyLM()
    parser = _DummyParser()
    helpers = CSDHelpers(lm, parser)

    token, remaining = helpers.AdaptiveConstrainedStep(
        ["prompt"],
        ["reason"],
        ["a"],
        [["b"]],
        2.0,
        1,
        "<eos>",
        5,
    )

    assert lm.last_input == ["prompt", "reason", "a"]
    assert token == "b"
    assert remaining == 4


def test_prefix_scanning_helpers_report_recent_structure():
    helpers = CSDHelpers(_DummyLM(), _DummyParser())
    generated = ["reason", "<<", "x", ">>", "tail"]

    assert helpers.LastTokenBefore(generated, ">>") == ("x", True)
    assert helpers.LastTokenBefore(["only"], "only") == ("", False)
    assert helpers.CountOccurrences(generated, ">>") == 1
    assert helpers.CountOccurrences(generated, "<<") == 1
    assert helpers.TokensSinceLastDelimiter(generated) == 1
    assert helpers.TokensSinceLastDelimiter(["reason", "tail"]) == 2


def test_transpile_contract_library_uses_new_return_name_overrides():
    source = Path("generation/csd/VerifiedAgentSynthesis.py").read_text(encoding="utf-8")
    result = transpile_contract_library(source)
    assert result.is_ok(), result.error
    output = result.value

    assert "method AdaptiveConstrainedStep" in output
    assert "returns (nextToken: Token, remainingSteps: int)" in output
    assert "method LastTokenBefore" in output
    assert "returns (token: Token, found: bool)" in output
    assert "method OpenConstrainedSpan" in output
    assert "returns (updated: Prefix, insideSpan: bool, currentConstrained: Prefix, remainingSteps: int)" in output
    assert "method CloseConstrainedSpan" in output
    assert "returns (updated: Prefix, insideSpan: bool, updatedConstrained: Prefix, remainingSteps: int)" in output


def test_transpile_contract_library_emits_python_line_markers():
    source = Path("generation/csd/VerifiedAgentSynthesis.py").read_text(encoding="utf-8")
    result = transpile_contract_library(source)
    assert result.is_ok(), result.error

    output = result.value

    assert "// Python line " in output

from pathlib import Path

from lark import Lark
from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedToken


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _build_parser() -> Lark:
    grammar = (PROJECT_ROOT / "utils" / "grammars" / "sql.lark").read_text(encoding="utf-8")
    return Lark(grammar, start="start", parser="lalr")


def _is_complete_sql(sql: str) -> bool:
    parser = _build_parser()
    try:
        parser.parse(sql)
        return True
    except Exception:
        return False


def _is_valid_prefix(sql: str) -> bool:
    parser = _build_parser()
    if not sql:
        return True
    try:
        parser.parse(sql)
        return True
    except UnexpectedEOF:
        return True
    except UnexpectedToken as exc:
        return exc.token.type == "$END"
    except UnexpectedCharacters:
        return False
    except Exception:
        return False


def test_sql_grammar_requires_quoted_string_literals_in_comparisons():
    sql = 'select city_name from city where state_name = "wyoming"'
    assert _is_complete_sql(sql)


def test_sql_grammar_rejects_unquoted_bareword_string_literal():
    sql = "select city_name from city where state_name = wyoming"
    assert not _is_complete_sql(sql)


def test_sql_grammar_keeps_scalar_subquery_for_spider_patterns():
    sql = (
        'select city_name from city where population = ( select max ( population ) from city '
        'where state_name = "wyoming" ) and state_name = "wyoming"'
    )
    assert _is_complete_sql(sql)


def test_sql_grammar_keeps_incomplete_quoted_literal_as_valid_prefix():
    sql = 'select city_name from city where state_name = "wy'
    assert _is_valid_prefix(sql)


def test_sql_grammar_allows_qualified_column_comparisons_for_joins():
    sql = (
        "select state.state_name from state join city on state.state_name = city.state_name "
        "where state_name = \"wyoming\""
    )
    assert _is_complete_sql(sql)

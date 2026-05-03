"""
Parser creation utilities for Dafny-compatible grammar parsers.

Provides factory functions to create Lark-based parsers that conform to
the Dafny Parser interface used by CSD strategies.
"""

from __future__ import annotations

import re
from pathlib import Path

from evaluation.common.generation import dafny_seq_to_str


_TRAILING_DECIMAL_RE = re.compile(r"(?:^|[^A-Za-z0-9_])[-+]?\d+\.$")
_TRAILING_SCRATCH_VAR_PREFIX_RE = re.compile(r"(?:^|[^A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]*_$")


def _is_recoverable_decimal_prefix(text: str, lark_parser) -> bool:
    """True for prefixes such as `8.` where one more digit can complete NUMBER."""
    if not _TRAILING_DECIMAL_RE.search(text):
        return False
    try:
        lark_parser.parse(text + "0")
        return True
    except Exception as exc:
        # `8.0` may still be an incomplete expression under the active start rule,
        # which Lark reports as EOF. That is still a valid decoding prefix.
        if exc.__class__.__name__ == "UnexpectedEOF":
            return True
        token = getattr(exc, "token", None)
        return getattr(token, "type", None) == "$END"


def _is_recoverable_scratch_var_prefix(text: str, lark_parser) -> bool:
    """True for prefixes such as `x_` where one more digit can complete SCRATCH_VAR."""
    if not _TRAILING_SCRATCH_VAR_PREFIX_RE.search(text):
        return False
    try:
        lark_parser.parse(text + "0")
        return True
    except Exception as exc:
        if exc.__class__.__name__ == "UnexpectedEOF":
            return True
        token = getattr(exc, "token", None)
        return getattr(token, "type", None) == "$END"


def _is_recoverable_lark_prefix(text: str, lark_parser) -> bool:
    return (
        _is_recoverable_decimal_prefix(text, lark_parser)
        or _is_recoverable_scratch_var_prefix(text, lark_parser)
    )


def create_lark_dafny_parser(
    grammar_source: str,
    VerifiedDecoderAgent,
    _dafny,
    start: str = "start"
):
    """
    Create a Dafny-compatible parser from a Lark grammar.

    This creates a parser class that implements the VerifiedDecoderAgent.Parser
    interface, allowing it to be used with CSD strategies compiled from Dafny.

    Args:
        grammar_source: Either a grammar string or path to .lark file
        VerifiedDecoderAgent: The imported Dafny module
        _dafny: The Dafny runtime module
        start: Start rule name in the grammar

    Returns:
        A LarkDafnyParser class that can be instantiated with a token list
    """
    from lark import Lark
    from lark.exceptions import UnexpectedCharacters, UnexpectedToken, UnexpectedEOF

    # Load grammar - check if it's a file path (short string without newlines)
    if '\n' not in grammar_source and len(grammar_source) < 500:
        grammar_path = Path(grammar_source)
        if grammar_path.exists():
            grammar = grammar_path.read_text()
        else:
            grammar = grammar_source
    else:
        grammar = grammar_source

    # Create Lark parser
    lark_parser = Lark(grammar, start=start, parser='lalr')

    class LarkDafnyParser(VerifiedDecoderAgent.Parser):
        """Parser using Lark grammar, compatible with Dafny-compiled code."""

        def __init__(self, lm_tokens):
            super().__init__()
            self._lm_tokens = lm_tokens
            # Convert Dafny Seq to Python list using index-based access
            # (Dafny.Seq has __len__ and __getitem__ but NOT __iter__)
            try:
                self._token_list = list(lm_tokens)
            except TypeError:
                self._token_list = [lm_tokens[i] for i in range(len(lm_tokens))]
            self._lark = lark_parser
            self._UnexpectedCharacters = UnexpectedCharacters
            self._UnexpectedToken = UnexpectedToken
            self._UnexpectedEOF = UnexpectedEOF

        def _dafny_seq_to_str(self, seq) -> str:
            """Convert a Dafny Seq to a Python string."""
            return dafny_seq_to_str(seq)

        def _tokens_to_text(self, tokens) -> str:
            """Convert Dafny token sequence to text."""
            try:
                return ''.join(self._dafny_seq_to_str(tokens[i]) for i in range(len(tokens)))
            except (TypeError, AttributeError, IndexError):
                return str(tokens)

        def _is_valid_prefix(self, text: str) -> bool:
            """Check if text is a valid prefix of the grammar."""
            if not text:
                return True

            if not hasattr(self, '_prefix_validity_cache'):
                self._prefix_validity_cache = {}

            if text in self._prefix_validity_cache:
                return self._prefix_validity_cache[text]

            try:
                self._lark.parse(text)
                res = True
            except self._UnexpectedEOF:
                res = True
            except self._UnexpectedToken as e:
                res = (e.token.type == '$END') or _is_recoverable_lark_prefix(text, self._lark)
            except self._UnexpectedCharacters:
                res = _is_recoverable_lark_prefix(text, self._lark)
            except Exception:
                res = False

            self._prefix_validity_cache[text] = res
            return res

        def _is_complete(self, text: str) -> bool:
            """Check if text is a complete valid parse."""
            if not text:
                return False
            try:
                self._lark.parse(text)
                return True
            except Exception:
                return False

        def is_valid_prefix(self, text: str) -> bool:
            """Public method: Check if text is a valid prefix."""
            return self._is_valid_prefix(text)

        def is_complete(self, text: str) -> bool:
            """Public method: Check if text is complete."""
            return self._is_complete(text)

        def IsValidPrefix(self, prefix) -> bool:
            """Dafny interface: Check if prefix is valid."""
            if len(prefix) == 0:
                return True
            text = self._tokens_to_text(prefix)
            return self._is_valid_prefix(text)

        def IsCompletePrefix(self, prefix) -> bool:
            """Dafny interface: Check if prefix is complete. Empty is valid but never complete."""
            try:
                if prefix is None or len(prefix) == 0:
                    return False
            except (TypeError, AttributeError):
                return False
            text = self._tokens_to_text(prefix)
            return self._is_complete(text)

        def ValidNextTokens(self, prefix):
            """Dafny interface: Get valid next tokens.

            For each token in the vocabulary, checks whether appending it
            to the current prefix yields a valid grammar prefix.
            """
            current_text = self._tokens_to_text(prefix) if len(prefix) > 0 else ""

            if current_text and not self._is_valid_prefix(current_text):
                return _dafny.SeqWithoutIsStrInference([])

            valid_tokens = []

            for token in self._token_list:
                token_str = self._dafny_seq_to_str(token)
                if not token_str:
                    continue

                extended = current_text + token_str
                if self._is_valid_prefix(extended):
                    valid_tokens.append(token)

            return _dafny.SeqWithoutIsStrInference(valid_tokens)

        def IsPermissive(self, prefix) -> bool:
            """Dafny interface: True only when every token is valid (e.g. free-form sections). Strict grammar => False."""
            return False

    return LarkDafnyParser


def create_lark_native_parser(
    grammar_source: str,
    VerifiedAgentSynthesis,
    start: str = "start",
):
    """
    Create a Python-native parser (no Dafny runtime) from a Lark grammar.

    Implements VerifiedAgentSynthesis.Parser so the strategy can call
    IsValidPrefix, IsCompletePrefix, ValidNextTokens with plain Python lists.
    """
    from lark import Lark
    from lark.exceptions import UnexpectedCharacters, UnexpectedToken, UnexpectedEOF

    if "\n" not in grammar_source and len(grammar_source) < 500:
        from pathlib import Path as _Path
        gp = _Path(grammar_source)
        grammar = gp.read_text() if gp.exists() else grammar_source
    else:
        grammar = grammar_source

    lark_parser = Lark(grammar, start=start, parser="lalr")

    class LarkNativeParser(VerifiedAgentSynthesis.Parser):
        """Lark grammar parser using plain Python lists (no Dafny types)."""

        def __init__(self, lm_tokens: list):
            self._token_list = list(lm_tokens)
            self._lark = lark_parser
            self._prefix_cache: dict[str, bool] = {}
            self._complete_cache: dict[str, bool] = {}
            self._valid_next_cache: dict[str, list] = {}

        def _tokens_to_text(self, prefix: list) -> str:
            if not prefix:
                return ""
            if all(isinstance(t, str) for t in prefix):
                return "".join(prefix)
            return "".join(str(t) for t in prefix)

        def _is_valid_prefix(self, text: str) -> bool:
            if text in self._prefix_cache:
                return self._prefix_cache[text]
            try:
                lark_parser.parse(text)
                res = True
            except UnexpectedEOF:
                res = True
            except UnexpectedToken as e:
                res = e.token.type == "$END" or _is_recoverable_lark_prefix(text, lark_parser)
            except UnexpectedCharacters:
                res = _is_recoverable_lark_prefix(text, lark_parser)
            except Exception:
                res = False
            self._prefix_cache[text] = res
            return res

        def _is_complete(self, text: str) -> bool:
            if text in self._complete_cache:
                return self._complete_cache[text]
            if not text:
                return False
            try:
                lark_parser.parse(text)
                res = True
            except Exception:
                res = False
            self._complete_cache[text] = res
            return res

        def is_valid_prefix(self, text: str) -> bool:
            return self._is_valid_prefix(text)

        def is_complete(self, text: str) -> bool:
            return self._is_complete(text)

        def IsValidPrefix(self, prefix: list) -> bool:
            if not prefix:
                return True
            return self._is_valid_prefix(self._tokens_to_text(prefix))

        def IsCompletePrefix(self, prefix: list) -> bool:
            if not prefix:
                return False
            return self._is_complete(self._tokens_to_text(prefix))

        def ValidNextTokens(self, prefix: list) -> list:
            current_text = self._tokens_to_text(prefix)
            cached = self._valid_next_cache.get(current_text)
            if cached is not None:
                return cached
            if current_text and not self._is_valid_prefix(current_text):
                return []
            valid = []
            for token in self._token_list:
                t_str = token if isinstance(token, str) else str(token)
                if t_str and self._is_valid_prefix(current_text + t_str):
                    valid.append(token)
            self._valid_next_cache[current_text] = valid
            return valid

        def IsPermissive(self, prefix: list) -> bool:
            return False

    return LarkNativeParser


def get_builtin_grammar(format_name: str) -> str:
    """
    Get built-in grammar for common formats.

    Args:
        format_name: One of "json", "sql", "math"

    Returns:
        Grammar string in Lark format

    Raises:
        ValueError: If format_name is not recognized
    """
    grammars = {
        "json": r'''
            start: value
            ?value: object | array | string | number | "true" -> true | "false" -> false | "null" -> null
            object: "{" [pair ("," pair)*] "}"
            pair: string ":" value
            array: "[" [value ("," value)*] "]"
            string: ESCAPED_STRING
            number: SIGNED_NUMBER
            %import common.ESCAPED_STRING
            %import common.SIGNED_NUMBER
            %import common.WS
            %ignore WS
        ''',
        "sql": r'''
            start: select_stmt
            select_stmt: "SELECT"i columns "FROM"i table [where_clause]
            columns: "*" | column ("," column)*
            column: NAME
            table: NAME
            where_clause: "WHERE"i condition
            condition: NAME comp_op value
            comp_op: "=" | "!=" | "<" | ">" | "<=" | ">="
            value: NAME | NUMBER | STRING
            %import common.CNAME -> NAME
            %import common.NUMBER
            %import common.ESCAPED_STRING -> STRING
            %import common.WS
            %ignore WS
        ''',
        "math": r'''
            start: expr
            ?expr: term | expr "+" term | expr "-" term
            ?term: factor | term "*" factor | term "/" factor
            ?factor: NUMBER | "(" expr ")" | "-" factor
            %import common.NUMBER
            %import common.WS
            %ignore WS
        ''',
    }

    if format_name.lower() not in grammars:
        raise ValueError(f"Unknown format: {format_name}. Available: {list(grammars.keys())}")

    return grammars[format_name.lower()]

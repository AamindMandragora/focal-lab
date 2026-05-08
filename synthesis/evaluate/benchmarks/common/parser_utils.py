"""
Parser creation utilities for Dafny-compatible grammar parsers.

Uses syncode's DFA mask store for O(1) token validity checks instead of
O(vocab) brute-force Lark parsing.
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any


# Per-component timing, shared conceptually with model_utils but kept separate
# so parser_utils doesn't depend on model_utils.
_PARSER_TIMINGS: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
_PARSER_TIMINGS_ENABLED = os.environ.get("CSD_DISABLE_TIMING", "") == ""


@contextmanager
def _parser_timed(label: str):
    if not _PARSER_TIMINGS_ENABLED:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        slot = _PARSER_TIMINGS[label]
        slot[0] += elapsed
        slot[1] += 1


def print_parser_timings(header: str = "") -> None:
    if not _PARSER_TIMINGS_ENABLED or not _PARSER_TIMINGS:
        return
    total = sum(t for t, _ in _PARSER_TIMINGS.values())
    if total <= 0:
        return
    lines = [f"[PARSER_TIMING] {header} total={total:.2f}s"]
    for label in sorted(_PARSER_TIMINGS.keys(), key=lambda k: -_PARSER_TIMINGS[k][0]):
        secs, calls = _PARSER_TIMINGS[label]
        pct = 100.0 * secs / total
        avg_ms = 1000.0 * secs / max(calls, 1)
        lines.append(f"  {label:<34} {secs:7.2f}s  ({pct:5.1f}%)  calls={calls:<5} avg={avg_ms:7.2f}ms")
    print("\n".join(lines), flush=True)

_PARSER_COMPONENT_CACHE = {}
_DFA_MASK_STORE_CACHE = {}


def _tokenizer_cache_fingerprint(tokenizer) -> tuple[str, int]:
    name = getattr(tokenizer, 'name_or_path', tokenizer.__class__.__name__)
    try:
        vocab_size = len(tokenizer)
    except Exception:
        vocab_size = -1
    return str(name), int(vocab_size)


def _ensure_syncode_import_path() -> None:
    syncode_dir = Path(__file__).parent.parent.parent / "syncode"
    # Vendored layout is synthesis/evaluate/syncode/syncode. We need
    # synthesis/evaluate/syncode on sys.path so imports like
    # `syncode.parsers` resolve correctly.
    candidates = [str(syncode_dir)]
    for candidate in reversed(candidates):
        if candidate in sys.path:
            sys.path.remove(candidate)
    for candidate in candidates:
        sys.path.insert(0, candidate)


def _load_grammar_text(grammar_source: str) -> str:
    if '\n' not in grammar_source and len(grammar_source) < 500:
        grammar_path = Path(grammar_source)
        if grammar_path.exists():
            return grammar_path.read_text()
    return grammar_source


def _get_parser_components(grammar_text: str, start: str):
    _ensure_syncode_import_path()
    from lark import Lark
    from syncode.parsers.grammars import Grammar
    from syncode.parsers import create_base_parser

    component_key = (grammar_text, start)
    cached_components = _PARSER_COMPONENT_CACHE.get(component_key)
    if cached_components is None:
        grammar = Grammar(grammar_text)
        base_parser = create_base_parser(grammar)
        lark_parser = Lark(grammar_text, start=start, parser='lalr')
        cached_components = (grammar, base_parser, lark_parser)
        _PARSER_COMPONENT_CACHE[component_key] = cached_components
    return cached_components


def _get_cached_dfa_mask_store(grammar_text: str, grammar, tokenizer):
    if tokenizer is None:
        return None

    _ensure_syncode_import_path()
    from syncode.dfa_mask_store import DFAMaskStore
    import syncode.common as common

    mask_key = (grammar_text, _tokenizer_cache_fingerprint(tokenizer))
    dfa_mask_store = _DFA_MASK_STORE_CACHE.get(mask_key)
    if dfa_mask_store is None:
        dfa_mask_store = DFAMaskStore.load_dfa_mask_store(
            grammar=grammar,
            tokenizer=tokenizer,
            use_cache=True,
            logger=common.EmptyLogger(),
            mode='grammar_mask',
        )
        _DFA_MASK_STORE_CACHE[mask_key] = dfa_mask_store
    return dfa_mask_store


def prewarm_dfa_mask_store(grammar_source: str, start: str = "start", tokenizer=None) -> dict[str, Any]:
    """Materialize and cache the DFA mask store for one grammar/tokenizer pair."""
    grammar_text = _load_grammar_text(grammar_source)
    grammar, _, _ = _get_parser_components(grammar_text, start)
    dfa_mask_store = _get_cached_dfa_mask_store(grammar_text, grammar, tokenizer)
    return {
        "grammar_text": grammar_text,
        "start": start,
        "tokenizer_fingerprint": _tokenizer_cache_fingerprint(tokenizer) if tokenizer is not None else None,
        "dfa_mask_store": dfa_mask_store,
    }


def create_lark_dafny_parser(
    grammar_source: str,
    VerifiedDecoderAgent,
    _dafny,
    start: str = "start",
    tokenizer=None,
):
    """
    Create a Dafny-compatible parser using syncode's DFA mask store.

    Args:
        grammar_source: Either a grammar string or path to .lark file
        VerifiedDecoderAgent: The imported Dafny module
        _dafny: The Dafny runtime module
        start: Start rule name in the grammar
        tokenizer: HuggingFace tokenizer (required for DFA mask store)

    Returns:
        A SyncodeDafnyParser class that can be instantiated with a token list
    """
    _ensure_syncode_import_path()

    from lark.exceptions import UnexpectedCharacters, UnexpectedToken, UnexpectedEOF
    from syncode.parsers.incremental_parser import IncrementalParser

    grammar_text = _load_grammar_text(grammar_source)
    grammar, base_parser, lark_parser = _get_parser_components(grammar_text, start)
    dfa_mask_store = _get_cached_dfa_mask_store(grammar_text, grammar, tokenizer)

    class SyncodeDafnyParser(VerifiedDecoderAgent.Parser):
        """Parser using syncode's DFA mask store for fast token validity checks."""

        def __init__(self, lm_tokens):
            super().__init__()
            self._lm_tokens = lm_tokens
            try:
                self._token_list = list(lm_tokens)
            except TypeError:
                self._token_list = [lm_tokens[i] for i in range(len(lm_tokens))]

            # Per-instance incremental parser (reset for each generation)
            self._inc_parser = IncrementalParser(base_parser, ignore_whitespace=False)
            self._dfa_mask_store = dfa_mask_store
            self._UnexpectedCharacters = UnexpectedCharacters
            self._UnexpectedToken = UnexpectedToken
            self._UnexpectedEOF = UnexpectedEOF

            # Precompute token string -> index mapping
            self._token_str_to_idx = {}
            for idx, token in enumerate(self._token_list):
                token_str = self._dafny_seq_to_str(token)
                if token_str:
                    self._token_str_to_idx.setdefault(token_str, []).append(idx)

            # Shared Lark parser for IsValidPrefix / IsCompletePrefix (rarely called)
            self._lark = lark_parser
            self._valid_prefix_cache = {}
            self._complete_cache = {}
            self._valid_next_mask_cache = {}
            self._valid_next_indices_cache = {}

        def _dafny_seq_to_str(self, seq) -> str:
            """Convert a Dafny Seq to a Python string."""
            try:
                return ''.join(seq)
            except TypeError:
                try:
                    return ''.join(seq[i] for i in range(len(seq)))
                except (TypeError, AttributeError, IndexError):
                    return str(seq)

        def _tokens_to_text(self, tokens) -> str:
            """Convert Dafny token sequence to text."""
            try:
                return ''.join(self._dafny_seq_to_str(tokens[i]) for i in range(len(tokens)))
            except (TypeError, AttributeError, IndexError):
                return str(tokens)

        def _is_valid_prefix(self, text: str) -> bool:
            """Check if text is a valid prefix of the grammar.

            Uses Syncode's IncrementalParser — which caches parser states
            keyed by lexer-token prefixes and only feeds the delta — instead
            of re-running Lark end-to-end. For append-only decoding this is
            O(1) amortized per call, vs. O(len(text)) for the full reparse
            path (and O(N^2) over a single generation).

            Falls back to full Lark.parse on unexpected exceptions so we
            never silently return a wrong answer.
            """
            with _parser_timed("is_valid_prefix.total"):
                if not text:
                    return True
                cached = self._valid_prefix_cache.get(text)
                if cached is not None:
                    return cached
                try:
                    # Drive the live incremental parser. It handles lexer-incomplete
                    # and final-token-unexpected cases internally as "still a valid
                    # prefix"; structural mismatches re-raise.
                    with _parser_timed("is_valid_prefix.inc_parser"):
                        self._inc_parser.get_acceptable_next_terminals(text)
                    result = True
                except (self._UnexpectedToken, self._UnexpectedCharacters, self._UnexpectedEOF):
                    result = False
                except Exception:
                    # Unknown failure inside the inc parser — fall back to full
                    # Lark parse so correctness is never compromised by the fast path.
                    with _parser_timed("is_valid_prefix.lark_fallback"):
                        try:
                            self._lark.parse(text)
                            result = True
                        except self._UnexpectedEOF:
                            result = True
                        except self._UnexpectedToken as e:
                            result = e.token.type == '$END'
                        except self._UnexpectedCharacters:
                            result = False
                        except Exception:
                            result = False
                self._valid_prefix_cache[text] = result
                return result

        def _is_complete(self, text: str) -> bool:
            """Check if text is a complete valid parse."""
            if not text:
                return False
            cached = self._complete_cache.get(text)
            if cached is not None:
                return cached
            try:
                self._lark.parse(text)
                result = True
            except Exception:
                result = False
            self._complete_cache[text] = result
            return result

        def _get_accept_mask_for_text(self, current_text: str):
            """Get boolean accept mask using syncode's DFA mask store."""
            with _parser_timed("accept_mask.total"):
                cached = self._valid_next_mask_cache.get(current_text)
                if cached is not None:
                    _PARSER_TIMINGS["accept_mask.cache_hit"][1] += 1
                    return cached
                if self._dfa_mask_store is None:
                    # Fallback: brute force (slow but correct)
                    with _parser_timed("accept_mask.brute_force_fallback"):
                        import torch
                        accept_mask = torch.zeros(len(self._token_list), dtype=torch.bool)
                        for idx, token in enumerate(self._token_list):
                            token_str = self._dafny_seq_to_str(token)
                            if token_str and self._is_valid_prefix(current_text + token_str):
                                accept_mask[idx] = True
                    self._valid_next_mask_cache[current_text] = accept_mask
                    return accept_mask

                try:
                    # Use syncode's incremental parser to get accept sequences
                    with _parser_timed("accept_mask.inc_parser_accepts"):
                        r = self._inc_parser.get_acceptable_next_terminals(current_text)
                    # Use DFA mask store to get accept mask, with sub-timers
                    # so we can see which piece of get_accept_mask dominates.
                    with _parser_timed("accept_mask.dfa_lookup"):
                        ms = self._dfa_mask_store
                        cur_incomplete_string = r.remainder
                        if cur_incomplete_string is None:
                            with _parser_timed("accept_mask.dfa_default_mask_ones"):
                                import torch as _t
                                accept_mask = _t.ones(len(ms._vocab), dtype=_t.bool)
                        else:
                            with _parser_timed("accept_mask.compute_dfa_states"):
                                cur_dfa_states = ms._dfas.compute_dfa_states(cur_incomplete_string)
                            with _parser_timed("accept_mask.lookup_next_tokens"):
                                accept_mask = ms._lookup_next_tokens(cur_dfa_states, r)
                            if ms.indentation and r.next_ac_indents is not None:
                                with _parser_timed("accept_mask.indent_intersect"):
                                    indent_ac_token = ms._lookup_table.get_indentation_tokens(r.next_ac_indents)
                                    accept_mask &= indent_ac_token
                        with _parser_timed("accept_mask.to_cpu"):
                            accept_mask = accept_mask.to(dtype=accept_mask.dtype, device='cpu')
                    self._valid_next_mask_cache[current_text] = accept_mask
                    return accept_mask
                except Exception:
                    # Fallback on parse error
                    import torch
                    return torch.zeros(len(self._token_list), dtype=torch.bool)

        def _get_accept_mask_for_prefix(self, prefix):
            """Get boolean accept mask for a Dafny prefix sequence."""
            current_text = self._tokens_to_text(prefix) if len(prefix) > 0 else ""
            return self._get_accept_mask_for_text(current_text)

        def _get_valid_token_indices(self, current_text: str):
            """Get list of valid token indices using a cached accept mask."""
            cached = self._valid_next_indices_cache.get(current_text)
            if cached is not None:
                return cached
            accept_mask = self._get_accept_mask_for_text(current_text)
            valid_indices = accept_mask.nonzero(as_tuple=False).flatten().tolist()
            self._valid_next_indices_cache[current_text] = valid_indices
            return valid_indices

        def is_valid_prefix(self, text: str) -> bool:
            return self._is_valid_prefix(text)

        def is_complete(self, text: str) -> bool:
            return self._is_complete(text)

        def IsValidPrefix(self, prefix) -> bool:
            """Dafny interface: Check if prefix is valid."""
            with _parser_timed("IsValidPrefix.dafny"):
                if len(prefix) == 0:
                    return True
                text = self._tokens_to_text(prefix)
                return self._is_valid_prefix(text)

        def IsCompletePrefix(self, prefix) -> bool:
            """Dafny interface: Check if prefix is complete."""
            with _parser_timed("IsCompletePrefix.dafny"):
                if len(prefix) == 0:
                    return False
                text = self._tokens_to_text(prefix)
                return self._is_complete(text)

        def ValidNextTokens(self, prefix):
            """Dafny interface: Get valid next tokens using DFA mask store."""
            with _parser_timed("ValidNextTokens.total"):
                with _parser_timed("ValidNextTokens.tokens_to_text"):
                    current_text = self._tokens_to_text(prefix) if len(prefix) > 0 else ""

                if current_text and not self._is_valid_prefix(current_text):
                    return _dafny.SeqWithoutIsStrInference([])

                with _parser_timed("ValidNextTokens.valid_indices"):
                    valid_indices = self._get_valid_token_indices(current_text)
                with _parser_timed("ValidNextTokens.materialize_dafny_seq"):
                    valid_tokens = [self._token_list[idx] for idx in valid_indices]
                    result = _dafny.SeqWithoutIsStrInference(valid_tokens)
                return result

        def ValidNextTokenCount(self, prefix):
            """Dafny interface: Count valid next tokens without materializing them."""
            with _parser_timed("ValidNextTokenCount.dafny"):
                current_text = self._tokens_to_text(prefix) if len(prefix) > 0 else ""

                if current_text and not self._is_valid_prefix(current_text):
                    return 0

                accept_mask = self._get_accept_mask_for_text(current_text)
                return int(accept_mask.sum().item())

        def ValidNextToken(self, prefix, token):
            """Dafny interface: Check one candidate token against the DFA mask."""
            with _parser_timed("ValidNextToken.dafny"):
                current_text = self._tokens_to_text(prefix) if len(prefix) > 0 else ""

                if current_text and not self._is_valid_prefix(current_text):
                    return False

                token_str = self._dafny_seq_to_str(token)
                if not token_str:
                    return False

                indices = self._token_str_to_idx.get(token_str)
                if not indices:
                    return False

                accept_mask = self._get_accept_mask_for_text(current_text)
                return any(bool(accept_mask[idx]) for idx in indices if idx < len(accept_mask))

        def GroupHasValidMember(self, prefix, group):
            """Dafny interface: bulk DFA-mask check for group membership.

            Equivalent to (any t in group: ValidNextToken(prefix, t)) but does
            the DFA accept-mask lookup once and walks the group in one Python
            loop, instead of one DFA query per token in a Dafny-compiled loop.
            """
            with _parser_timed("GroupHasValidMember.dafny"):
                current_text = self._tokens_to_text(prefix) if len(prefix) > 0 else ""
                if current_text and not self._is_valid_prefix(current_text):
                    return False
                accept_mask = self._get_accept_mask_for_text(current_text)
                accept_len = len(accept_mask)
                str_to_idx = self._token_str_to_idx
                for tok_dafny in group:
                    token_str = self._dafny_seq_to_str(tok_dafny)
                    if not token_str:
                        continue
                    indices = str_to_idx.get(token_str)
                    if not indices:
                        continue
                    for idx in indices:
                        if idx < accept_len and bool(accept_mask[idx]):
                            return True
                return False

    return SyncodeDafnyParser


def get_builtin_grammar(format_name: str) -> str:
    """Get built-in grammar for common formats."""
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

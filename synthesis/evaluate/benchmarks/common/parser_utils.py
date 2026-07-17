"""
Parser creation utilities for Dafny-compatible grammar parsers.

Uses syncode's DFA mask store for O(1) token validity checks instead of
O(vocab) brute-force Lark parsing.
"""

from __future__ import annotations

import os
import re as _re
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from synthesis.evaluate.benchmarks.common.dafny_tokens import dafny_seq_to_str


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
    syncode_dir = Path(
        os.environ.get(
            "CSD_SYNCODE_DIR",
            str(Path(__file__).parent.parent.parent / "syncode"),
        )
    ).expanduser()
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


_CSD_CLOSER_LITERAL_RE = _re.compile(
    r'^csd_start\s*:.*"((?:[^"\\]|\\.)+)"\s*$', _re.MULTILINE
)


def _grammar_closer_literal(grammar_text: str) -> str | None:
    """Return the inline literal closer of the grammar's csd_start rule, if any.

    Completeness semantics depend on whether the model can emit the span closer
    itself during constrained decoding:
    - gsm.lark: `csd_start: any_expr ">>"` — the closer is an inline literal the
      decode mask allows after a complete expression (June runs proved the model
      emits it), so "complete" should REQUIRE the closer: the span stays open
      until the model finishes the whole expression. Judging bare content instead
      closes the span after the first variable and the expression spills outside
      the delimiters (the July GSM collapse).
    - sql.lark: `csd_start: sql_stmt EOQ` — EOQ is a terminal reference
      (/[;\\n]+/), not an inline literal; the closer cannot reliably be emitted
      under the mask (the 0/0 wedge the 2026-06-22 change fixed), so completeness
      is judged on the CONTENT rule and the strategy force-appends the closer.
    - smiles_*.lark: no csd_start rule at all — content semantics.
    """
    match = _CSD_CLOSER_LITERAL_RE.search(grammar_text)
    return match.group(1) if match else None


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
        # Separate parser for IsCompletePrefix / _is_complete.  The prefix/validity
        # parser above uses the CSD start rule (e.g. gsm csd_start = `any_expr ">>"`,
        # sql csd_start = `sql_stmt EOQ`), which REQUIRES the closing delimiter.  But
        # that closer (">>" / EOQ) is permanently grammar-masked during constrained
        # decoding, so checking "is this span content complete?" against csd_start can
        # never return True -> spans never close (the 0/0 wedge).  Completeness is about
        # the CONTENT rule "start" (any_expr / sql_stmt / smiles) WITHOUT the closer; the
        # strategy force-appends the closer itself via CloseConstrainedSpan.  Every
        # grammar (gsm/sql/smiles) defines a "start" rule, so this is uniform.  This only
        # changes the generation loop's completeness signal -- it does NOT touch scoring:
        # the syntax-rate metric uses a separate start="syncode" parser (eval_logic) that
        # still requires the literal "<<...>>".
        complete_lark_parser = Lark(grammar_text, start="start", parser='lalr')
        cached_components = (grammar, base_parser, lark_parser, complete_lark_parser)
        _PARSER_COMPONENT_CACHE[component_key] = cached_components
    return cached_components


def resolve_mask_mode(default_mask_mode: str = "grammar_mask") -> str:
    """Mask-store mode: CSD_SYNCODE_MASK_MODE overrides; else the caller's default.

    CRANE builds its store with 'grammar_strict' (CRANE/src/models/
    base_model.py:23, CRANE/src/itergen/lebron.py:152). GSM call sites pass
    default_mask_mode='grammar_strict' to match it (user decision 2026-07-17;
    A/B on the 49-example seed123 train set: strict 34.69% acc / 93.88% syn
    vs mask 32.65% / 97.96%). Spider is also strict (user decision 2026-07-17):
    IterGen builds its store strict too (itergen/main.py:93), so strict is the
    fair footing even though our A/B measured it costing Spider 8pp acc
    (60.33% -> 52.33%; strict blocks fused dot tokens '.s'/'._' Qwen needs for
    table.column identifiers — IterGen pays the same constraint). SMILES keeps
    'grammar_mask' (own setup_dafny_environment, deliberately untouched).
    """
    return os.environ.get("CSD_SYNCODE_MASK_MODE", default_mask_mode)


def _get_cached_dfa_mask_store(grammar_text: str, grammar, tokenizer,
                               default_mask_mode: str = "grammar_mask"):
    if tokenizer is None:
        return None

    _ensure_syncode_import_path()
    from syncode.dfa_mask_store import DFAMaskStore
    import syncode.common as common

    mode = resolve_mask_mode(default_mask_mode)
    mask_key = (grammar_text, _tokenizer_cache_fingerprint(tokenizer), mode)
    dfa_mask_store = _DFA_MASK_STORE_CACHE.get(mask_key)
    if dfa_mask_store is None:
        dfa_mask_store = DFAMaskStore.load_dfa_mask_store(
            grammar=grammar,
            tokenizer=tokenizer,
            use_cache=True,
            logger=common.EmptyLogger(),
            mode=mode,
        )
        _DFA_MASK_STORE_CACHE[mask_key] = dfa_mask_store
    return dfa_mask_store


def create_lark_dafny_parser(
    grammar_source: str,
    VerifiedDecoderAgent,
    _dafny,
    start: str = "start",
    tokenizer=None,
    default_mask_mode: str = "grammar_mask",
):
    """
    Create a Dafny-compatible parser using syncode's DFA mask store.

    Args:
        grammar_source: Either a grammar string or path to .lark file
        VerifiedDecoderAgent: The imported Dafny module
        _dafny: The Dafny runtime module
        start: Start rule name in the grammar
        tokenizer: HuggingFace tokenizer (required for DFA mask store)
        default_mask_mode: mask-store mode when CSD_SYNCODE_MASK_MODE is
            unset ('grammar_mask' or 'grammar_strict'); see resolve_mask_mode

    Returns:
        A SyncodeDafnyParser class that can be instantiated with a token list
    """
    _ensure_syncode_import_path()

    from lark.exceptions import UnexpectedCharacters, UnexpectedToken, UnexpectedEOF
    from syncode.parsers.incremental_parser import IncrementalParser

    grammar_text = _load_grammar_text(grammar_source)
    grammar, base_parser, lark_parser, complete_lark_parser = _get_parser_components(grammar_text, start)
    dfa_mask_store = _get_cached_dfa_mask_store(grammar_text, grammar, tokenizer,
                                                default_mask_mode=default_mask_mode)

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
                token_str = dafny_seq_to_str(token)
                if token_str:
                    self._token_str_to_idx.setdefault(token_str, []).append(idx)

            # Hard-block mask: tokens whose string contains '{', '}', or '**'
            # are permanently forbidden regardless of the DFA over-approximation.
            # Syncode's grammar_mask mode over-approximates — e.g. whitespace-
            # prefixed tokens like ' {' slip through because '%ignore WS' makes
            # whitespace valid at any grammar state.  These characters are not
            # terminals in the GSM grammar, so blocking them is always safe.
            # The mask is True at positions that ARE allowed (i.e. not forbidden).
            import torch as _torch
            _forbidden_chars = ("{", "}", "**")
            _fb = _torch.ones(len(self._token_list), dtype=_torch.bool)
            for _idx, _token in enumerate(self._token_list):
                _ts = dafny_seq_to_str(_token)
                if any(_c in _ts for _c in _forbidden_chars):
                    _fb[_idx] = False
            self._forbidden_allow_mask: _torch.Tensor = _fb

            # Lark parser for the rare IsValidPrefix fallback (keeps the CSD start
            # rule, e.g. csd_start, which requires the closing delimiter).
            self._lark = lark_parser
            # Inline literal closer of csd_start (">>" for gsm, None for sql /
            # smiles). Non-None means the model can emit the closer itself, so
            # _is_complete requires it; see _grammar_closer_literal.
            self._closer_literal = (
                _grammar_closer_literal(grammar_text) if start == "csd_start" else None
            )
            # Separate parser for IsCompletePrefix completeness checks: its start rule
            # is the CONTENT rule "start" (any_expr / sql_stmt / smiles) WITHOUT the
            # grammar-masked closer, so span content can be recognized as complete and
            # the span can actually close.  See _get_parser_components for the why.
            self._complete_lark = complete_lark_parser
            self._valid_prefix_cache = {}
            self._complete_cache = {}
            self._valid_next_mask_cache = {}
            self._valid_next_indices_cache = {}
            self._valid_next_count_cache = {}
            # Dafny Seq objects are immutable in the generated runtime, but
            # converting one to text is O(len(prefix)). Cache by object identity
            # before conversion so repeated helper checks on the same prefix do
            # not rescan the token sequence.
            self._prefix_text_cache = {}
            self._valid_prefix_by_prefix_cache = {}
            self._complete_by_prefix_cache = {}
            self._valid_next_mask_by_prefix_cache = {}
            self._valid_next_count_by_prefix_cache = {}
            self._dead_prefix_by_prefix_cache = {}

        def _tokens_to_text(self, tokens) -> str:
            """Convert Dafny token sequence to text."""
            try:
                return ''.join(dafny_seq_to_str(tokens[i]) for i in range(len(tokens)))
            except (TypeError, AttributeError, IndexError):
                return str(tokens)

        def _prefix_cache_key(self, prefix):
            """Return a stable-enough cache key for one immutable Dafny prefix."""
            try:
                return (id(prefix), len(prefix))
            except (TypeError, AttributeError):
                return None

        def _prefix_cache_get(self, cache, prefix):
            key = self._prefix_cache_key(prefix)
            if key is None:
                return None
            cached = cache.get(key)
            if cached is None:
                return None
            cached_prefix, cached_value = cached
            if cached_prefix is prefix:
                return cached_value
            return None

        def _prefix_cache_set(self, cache, prefix, value):
            key = self._prefix_cache_key(prefix)
            if key is not None:
                cache[key] = (prefix, value)

        def _prefix_to_text(self, prefix) -> str:
            """Convert prefix to text once per prefix object."""
            if len(prefix) == 0:
                return ""
            cached = self._prefix_cache_get(self._prefix_text_cache, prefix)
            if cached is not None:
                return cached
            text = self._tokens_to_text(prefix)
            self._prefix_cache_set(self._prefix_text_cache, prefix, text)
            return text

        def _is_valid_prefix(self, text: str) -> bool:
            """Check if text is a valid prefix of the grammar.

            Uses Syncode's IncrementalParser — which caches parser states
            keyed by lexer-token prefixes and only feeds the delta — instead
            of re-running Lark end-to-end. The call still lexes the current
            text, so prefix-object caches avoid repeated Dafny-prefix scans
            before this fast path is reached.

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
            # Closer-required semantics: when the grammar's csd_start closer is
            # an inline literal the model can emit (gsm ">>"), a span is only
            # complete once the model has emitted that closer — bare content
            # like a lone variable must NOT close the span (2026-07-17 fix for
            # the July GSM collapse; see _grammar_closer_literal).  Grammars
            # whose closer is grammar-masked (sql EOQ) or absent (smiles) keep
            # the content semantics below.  Env overrides are DIAGNOSTIC ONLY:
            # CSD_COMPLETE_REQUIRES_CLOSER=1 forces closer-required everywhere,
            # CSD_COMPLETE_CONTENT_ONLY=1 forces content semantics everywhere.
            require_closer = (
                self._closer_literal is not None
                and os.environ.get("CSD_COMPLETE_CONTENT_ONLY") != "1"
            )
            if require_closer or os.environ.get("CSD_COMPLETE_REQUIRES_CLOSER") == "1":
                try:
                    self._lark.parse(text)
                    result = True
                except Exception:
                    result = False
                self._complete_cache[text] = result
                return result
            try:
                with _parser_timed("is_complete.inc_parser"):
                    parse_result = self._inc_parser.get_acceptable_next_terminals(text)
                remainder_state = getattr(parse_result, "remainder_state", None)
                remainder_state_name = getattr(remainder_state, "name", str(remainder_state))
                result = bool(
                    getattr(parse_result, "function_end", False)
                    and remainder_state_name in {"COMPLETE", "MAYBE_COMPLETE"}
                )
            except (self._UnexpectedToken, self._UnexpectedCharacters, self._UnexpectedEOF):
                result = False
            except Exception:
                # Unknown failure inside the incremental path: keep the old full
                # parse behavior as a correctness fallback.
                with _parser_timed("is_complete.lark_fallback"):
                    try:
                        self._complete_lark.parse(text)
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
                            token_str = dafny_seq_to_str(token)
                            if token_str and self._is_valid_prefix(current_text + token_str):
                                accept_mask[idx] = True
                    # Apply hard-block: remove forbidden-char tokens.
                    accept_mask = accept_mask & self._forbidden_allow_mask
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
                    # Apply hard-block: remove forbidden-char tokens that slipped
                    # through syncode's over-approximation.
                    accept_mask = accept_mask & self._forbidden_allow_mask
                    self._valid_next_mask_cache[current_text] = accept_mask
                    return accept_mask
                except Exception:
                    # Fallback on parse error
                    import torch
                    return torch.zeros(len(self._token_list), dtype=torch.bool)

        def _get_accept_mask_for_prefix(self, prefix):
            """Get boolean accept mask for a Dafny prefix sequence."""
            cached = self._prefix_cache_get(self._valid_next_mask_by_prefix_cache, prefix)
            if cached is not None:
                return cached
            current_text = self._prefix_to_text(prefix)
            accept_mask = self._get_accept_mask_for_text(current_text)
            self._prefix_cache_set(self._valid_next_mask_by_prefix_cache, prefix, accept_mask)
            return accept_mask

        def _get_valid_token_indices(self, current_text: str):
            """Get list of valid token indices using a cached accept mask."""
            cached = self._valid_next_indices_cache.get(current_text)
            if cached is not None:
                return cached
            accept_mask = self._get_accept_mask_for_text(current_text)
            valid_indices = accept_mask.nonzero(as_tuple=False).flatten().tolist()
            self._valid_next_indices_cache[current_text] = valid_indices
            return valid_indices

        def _valid_next_token_count_for_text(self, current_text: str) -> int:
            """Count valid next tokens once per text prefix."""
            cached = self._valid_next_count_cache.get(current_text)
            if cached is not None:
                return cached
            accept_mask = self._get_accept_mask_for_text(current_text)
            count = int(accept_mask.sum().item())
            self._valid_next_count_cache[current_text] = count
            return count

        def is_valid_prefix(self, text: str) -> bool:
            return self._is_valid_prefix(text)

        def is_complete(self, text: str) -> bool:
            return self._is_complete(text)

        def IsValidPrefix(self, prefix) -> bool:
            """Dafny interface: Check if prefix is valid."""
            with _parser_timed("IsValidPrefix.dafny"):
                if len(prefix) == 0:
                    return True
                cached = self._prefix_cache_get(self._valid_prefix_by_prefix_cache, prefix)
                if cached is not None:
                    return cached
                text = self._prefix_to_text(prefix)
                result = self._is_valid_prefix(text)
                self._prefix_cache_set(self._valid_prefix_by_prefix_cache, prefix, result)
                return result

        def IsCompletePrefix(self, prefix) -> bool:
            """Dafny interface: Check if prefix is complete."""
            with _parser_timed("IsCompletePrefix.dafny"):
                if len(prefix) == 0:
                    return False
                cached = self._prefix_cache_get(self._complete_by_prefix_cache, prefix)
                if cached is not None:
                    return cached
                text = self._prefix_to_text(prefix)
                result = self._is_complete(text)
                self._prefix_cache_set(self._complete_by_prefix_cache, prefix, result)
                return result

        def ValidNextTokens(self, prefix):
            """Dafny interface: Get valid next tokens using DFA mask store."""
            with _parser_timed("ValidNextTokens.total"):
                with _parser_timed("ValidNextTokens.tokens_to_text"):
                    current_text = self._prefix_to_text(prefix)

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
                cached = self._prefix_cache_get(self._valid_next_count_by_prefix_cache, prefix)
                if cached is not None:
                    return cached
                current_text = self._prefix_to_text(prefix)

                if current_text and not self._is_valid_prefix(current_text):
                    self._prefix_cache_set(self._valid_next_count_by_prefix_cache, prefix, 0)
                    return 0

                count = self._valid_next_token_count_for_text(current_text)
                self._prefix_cache_set(self._valid_next_count_by_prefix_cache, prefix, count)
                return count

        def IsDeadPrefix(self, prefix):
            """Dafny interface: Check dead-end status using one text conversion."""
            with _parser_timed("IsDeadPrefix.dafny"):
                cached = self._prefix_cache_get(self._dead_prefix_by_prefix_cache, prefix)
                if cached is not None:
                    return cached

                current_text = self._prefix_to_text(prefix)
                if current_text and not self._is_valid_prefix(current_text):
                    result = True
                elif self._is_complete(current_text):
                    result = False
                else:
                    result = self._valid_next_token_count_for_text(current_text) == 0

                self._prefix_cache_set(self._dead_prefix_by_prefix_cache, prefix, result)
                return result

        def ValidNextToken(self, prefix, token):
            """Dafny interface: Check one candidate token against the DFA mask."""
            with _parser_timed("ValidNextToken.dafny"):
                current_text = self._prefix_to_text(prefix)

                if current_text and not self._is_valid_prefix(current_text):
                    return False

                token_str = dafny_seq_to_str(token)
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
                current_text = self._prefix_to_text(prefix)
                if current_text and not self._is_valid_prefix(current_text):
                    return False
                accept_mask = self._get_accept_mask_for_text(current_text)
                accept_len = len(accept_mask)
                str_to_idx = self._token_str_to_idx
                for tok_dafny in group:
                    token_str = dafny_seq_to_str(tok_dafny)
                    if not token_str:
                        continue
                    indices = str_to_idx.get(token_str)
                    if not indices:
                        continue
                    for idx in indices:
                        if idx < accept_len and bool(accept_mask[idx]):
                            return True
                return False

        def CompletedSchemaSymbolCount(self, prefix):
            """Dafny interface: number of table_ref/column_ref symbols that have
            COMPLETED within `prefix`.

            Reads the IncrementalParser's SymbolPosMap side-record (IterGen's
            mechanism, ported into the vendored parser). Drives the incremental
            parser on this prefix's text first so the map reflects exactly this
            prefix (the parser caches by lexer-token prefix, so re-driving a grown
            or rolled-back prefix is cheap and restores the map to that point),
            then counts the completed schema symbols.

            PURE SIDE-RECORD: the SymbolPosMap is never read by the accept-set /
            masking path, so reading this count cannot change decode for any
            caller. Used as a mid-query unit boundary by the grounding helper:
            when the count rises, one more table/column name just finished.
            """
            with _parser_timed("CompletedSchemaSymbolCount.dafny"):
                current_text = self._tokens_to_text(prefix) if len(prefix) > 0 else ""
                if not current_text:
                    return 0
                try:
                    # Drive the inc parser so symbol_pos_map reflects THIS prefix.
                    self._inc_parser.get_acceptable_next_terminals(current_text)
                except Exception:
                    # Not parseable as a prefix here: report no completed symbols so
                    # the boundary simply doesn't fire (caller keeps prevCount).
                    return 0
                spm = getattr(self._inc_parser, "symbol_pos_map", None)
                if spm is None:
                    return 0
                # Schema-bearing symbols in our sql.lark: table_ref + column_ref.
                # get_symbol_count(after=0) counts every completed span.
                return int(
                    spm.get_symbol_count("table_ref")
                    + spm.get_symbol_count("column_ref")
                )

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


def _compute_unit_rollback_info(parser, tokens):
    """Find the last completed grammar unit in a token sequence.

    Inputs:
      parser   -- any object with an IsCompletePrefix(prefix: list) method
      tokens   -- a list of token objects (plain strings or Dafny sequences)

    Output:
      None if no complete unit has been generated.
      Otherwise a tuple (rollback_pos, unit_tokens) where:
        rollback_pos  -- index into tokens where the last unit began (i.e., the
                         position of the last complete point BEFORE this unit)
        unit_tokens   -- tokens[rollback_pos : last_complete_end]

    Algorithm:
      1. Walk the token list left-to-right, calling IsCompletePrefix on each
         growing prefix.
      2. Each time IsCompletePrefix transitions from True-at-i to False-at-i+1
         (or True-at-final), record the end position as a "complete boundary".
      3. The last two consecutive boundaries define the last unit.
         If only one boundary exists, the unit spans from index 0 to that boundary.
    """
    if not tokens:
        return None

    complete_ends = []
    for i in range(1, len(tokens) + 1):
        if parser.IsCompletePrefix(tokens[:i]):
            complete_ends.append(i)
        elif complete_ends and complete_ends[-1] == i - 1:
            # We just stepped past a complete point; the previous boundary stands.
            pass

    if not complete_ends:
        return None

    last_end = complete_ends[-1]
    # The unit start is right after the second-to-last complete boundary, or 0.
    if len(complete_ends) >= 2:
        rollback_pos = complete_ends[-2]
    else:
        rollback_pos = 0

    unit_tokens = tokens[rollback_pos:last_end]
    return rollback_pos, unit_tokens

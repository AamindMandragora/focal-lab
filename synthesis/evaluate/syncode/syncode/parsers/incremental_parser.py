import copy
from syncode.larkm.tree import Tree
from syncode.larkm.parsers.lalr_analysis import Reduce
from syncode.larkm.parsers.lalr_parser_state import ParserState
import syncode.common as common
import syncode.larkm as lark
from syncode.larkm.parsers.lalr_interactive_parser import InteractiveParser
from syncode.parse_result import ParseResult, RemainderState
from syncode.larkm.lexer import Token
from typing import Optional, Any, Tuple, Iterable
from collections import defaultdict


class SymbolPosMap:
    """Maps each grammar symbol to the list of (start_pos, end_pos) CHARACTER
    spans where it occurs in the generated code, sorted by start position.

    Ported verbatim from IterGen's incremental_parser.py (faithful parity). It is
    a PURE SIDE-RECORD: nothing in this class is read by the accept-set / parsing
    path; it only records where table/column (and other) symbols completed so a
    grounding helper can backtrack to a bad identifier mid-query.
    """
    def __init__(self):
        self._pos_map = defaultdict(list)

    def add_symbol_pos(self, symbol: str, pos: Tuple[int, int]):
        start_pos, _ = pos
        if len(self._pos_map[symbol]) == 0 or self._pos_map[symbol][-1][0] != start_pos:
            self._pos_map[symbol].append(pos)
        elif self._pos_map[symbol][-1][0] == start_pos:
            self._pos_map[symbol][-1] = pos

    def get_symbol_pos_start(self, symbol: str, idx: int) -> int:
        return self._pos_map[symbol][idx][0]

    def get_symbol_pos_end(self, symbol: str, idx: int) -> int:
        return self._pos_map[symbol][idx][1]

    def get_symbol_pos_all(self, symbol: str) -> list:
        return self._pos_map[symbol]

    def get_symbol_count(self, symbol: str, after: int = 0) -> int:
        return len([pos for pos in self._pos_map[symbol] if pos[1] > after])

    def crop(self, target_char_pos: int):
        for symbol, pos_list in self._pos_map.items():
            self._pos_map[symbol] = [pos for pos in pos_list if pos[1] <= target_char_pos]

    def is_present(self, symbol: str) -> bool:
        return symbol in self._pos_map


class IncrementalParser:
    """
    This is the base class for all incremental parsers.
    """
    def __init__(self, base_parser, logger: Optional[common.Logger]=None, ignore_whitespace=False) -> None:
        self.cur_pos = 0 # Current cursor position in the lexer tokens list
        self.lexer_pos = 0 # Current lexer position in the code
        self.dedent_queue: list = []
        self._ignore_whitespace = ignore_whitespace

        # Initialize the parser
        self.base_parser = base_parser

        self.logger = logger if logger is not None else common.EmptyLogger()
        self.interactive = self.base_parser.parse_interactive('')
        self.parsed_lexer_tokens: list = []

        # parser_state, cur_ac_terminals, next_ac_terminals, indent_levels (optional), dedent_queue
        self.cur_pos_to_parser_state: dict[int, Tuple[Any, set, set, Optional[list], list]] = {}
         
        self.cur_ac_terminals: set = set()
        self.next_ac_terminals: set = self._accepts(self.interactive)
        self.symbol_pos_map: SymbolPosMap = SymbolPosMap()

    def reset(self):
        """
        Resets the parser to the initial state.
        """
        self.cur_pos_to_parser_state = {}
        self.lexer_pos = 0

        # Reset maps used to mark units
        self.symbol_pos_map = SymbolPosMap()

        # Reset the parser state
        self._set_initial_parser_state()

    def _set_initial_parser_state(self):
        self.cur_pos = 0
        self.dedent_queue = []
        self.parsed_lexer_tokens = []
        self.interactive = self.base_parser.parse_interactive('')
        self.cur_ac_terminals = set()
        self.next_ac_terminals = self._accepts(self.interactive)
    
    def _store_parser_state(self, pos: int, lexer_tokens: Iterable[Token], parser_state, accepts: set, indent_levels: Optional[list] = None):  
        cur_ac_terminals = self.next_ac_terminals  
        next_ac_terminals = accepts 
        
        # Create a hash of lexer tokens till position pos
        key = self._get_hash(lexer_tokens[:pos+1])

        # parser_state, cur_ac_terminals, next_ac_terminals, indent_levels, dedent_queue, symbol_pos_map
        self.cur_pos_to_parser_state[key] = (copy.deepcopy(self.parsed_lexer_tokens), parser_state, cur_ac_terminals, next_ac_terminals, indent_levels, copy.deepcopy(self.dedent_queue), copy.deepcopy(self.symbol_pos_map))
        
        self.cur_ac_terminals = copy.deepcopy(cur_ac_terminals)
        self.next_ac_terminals = copy.deepcopy(next_ac_terminals)

    def _restore_parser_state(self, key: int):
        parsed_lexer_tokens, parser_state, cur_ac_terminals, next_ac_terminals, indent_levels, dedent_queue, symbol_pos_map = self.cur_pos_to_parser_state[key]

        self.interactive.parser_state = parser_state.copy()
        self.parsed_lexer_tokens = copy.deepcopy(parsed_lexer_tokens)
        self.dedent_queue = copy.deepcopy(dedent_queue)
        self.cur_ac_terminals = copy.deepcopy(cur_ac_terminals)
        self.next_ac_terminals = copy.deepcopy(next_ac_terminals)
        self.symbol_pos_map = copy.deepcopy(symbol_pos_map)

        if indent_levels is not None:
            self.indent_level = copy.deepcopy(indent_levels)


    def _lex_code(self, code) -> Tuple[Iterable[Token], bool]:
        """
        Lexes the given code and returns the list of tokens.
        """
        # Collect Lexer tokens
        lexer_tokens: Iterable[Token] = []
        interactive = self.base_parser.parse_interactive(code)
        lexer_state = interactive.lexer_thread.state
        lexing_incomplete = False
        try:
            while lexer_state.line_ctr.char_pos < len(lexer_state.text):
                blexer = interactive.lexer_thread.lexer
                token = blexer.next_token(lexer_state)
                self.lexer_pos = lexer_state.line_ctr.char_pos
                lexer_tokens.append(token)
        except lark.exceptions.UnexpectedCharacters as e:
            lexing_incomplete = True
            # We update the lexer position to the current position since the lexer has stopped at this position
            self.lexer_pos = lexer_state.line_ctr.char_pos
        except EOFError as e:
            pass
    
        return lexer_tokens, lexing_incomplete
    
    def _restore_recent_parser_state(self, lexer_tokens):
        """
        Restores the parser state to the most recent prefix matching state that was stored. 
        """
        max_stored_index = -1
        idx = len(lexer_tokens)-1
        
        while idx >= 0:
            # TODO: This is not the best way to hash the lexer tokens. We should use a better hashing mechanism with some sliding window. 
            key = self._get_hash(lexer_tokens[:idx+1])
            if key in self.cur_pos_to_parser_state:
                max_stored_index = idx
                break
            idx -= 1

        if max_stored_index != -1:
            self.cur_pos = max_stored_index + 1
            key = self._get_hash(lexer_tokens[:max_stored_index+1])
            self._restore_parser_state(key)
        else:
            self._set_initial_parser_state()

    def _get_hash(self, lexer_tokens: Iterable[Token]) -> int:
        return hash(tuple(lexer_tokens))

    def get_acceptable_next_terminals(self, partial_code) -> ParseResult:
        """
        Returns the set of acceptable terminals at the current partial code position.
        """
        # Stores the sequence of tokens that the parser has seen in the order  
        interactive = self.interactive
        lexer_tokens, lexing_incomplete = self._lex_code(partial_code)
        self.next_ac_terminals = self._accepts(interactive)

        # Restore the previous state of the parser
        self._restore_recent_parser_state(lexer_tokens)

        # Record completed TERMINAL symbol positions (pure side-record; does not
        # touch the accept-set / parsing path).
        self._update_symbol_pos_map_terminals(lexer_tokens)

        # Parse the tokens
        self.time_accepts = 0
        parse_incomplete = False

        try:
            while self.cur_pos < len(lexer_tokens):
                token = lexer_tokens[self.cur_pos]
                self.cur_pos += 1

                # Record completed NON-TERMINAL symbol positions BEFORE feeding the
                # token (it simulates reduces on a DEEP COPY of the parser stacks, so
                # it never mutates parser state or the accept-sets).
                self._update_symbol_pos_map_nonterminals(interactive.parser_state, token)

                self.parsed_lexer_tokens.append(token) # parser_token_seq holds all tokens
                interactive.feed_token(token)

                # Store the current state of the parser
                self._store_parser_state(
                    self.cur_pos-1,
                    lexer_tokens, 
                    interactive.parser_state.copy(), 
                    self._accepts(interactive))

        except lark.exceptions.UnexpectedToken as e:
            parse_incomplete = True
            self._handle_parsing_error(lexer_tokens, token, e)

        # Compute current terminal string
        remainder_state, current_term_str, final_terminal = self._get_remainder(partial_code, lexing_incomplete=lexing_incomplete, parse_incomplete=parse_incomplete)            
        
        return ParseResult.from_accept_terminals(self.cur_ac_terminals, self.next_ac_terminals, current_term_str, remainder_state, final_terminal=final_terminal, ignore_terminals=self.base_parser.lexer_conf.ignore)

    def _get_remainder(self, code, lexing_incomplete=False, parse_incomplete=False):
        final_terminal = None
        if lexing_incomplete: # Lexing is incomplete
            current_term_str = code[self.lexer_pos:]
            
            if self._ignore_whitespace:
                current_term_str = current_term_str.lstrip(' ') # Remove space from the beginning

            if current_term_str == '':
                remainder_state = RemainderState.COMPLETE
            else: 
                remainder_state = RemainderState.INCOMPLETE
                self.cur_ac_terminals = self.next_ac_terminals
                self.next_ac_terminals = set()

        elif parse_incomplete: # Parsing is incomplete
            remainder_state = RemainderState.INCOMPLETE
            current_term_str = self.parsed_lexer_tokens[-1].value
            final_terminal = self.parsed_lexer_tokens[-1].type
            
        elif len(self.parsed_lexer_tokens) > 0:
            if self.lexer_pos < len(code): # In this case the final lexical tokens are ignored by the parser
                remainder_state = RemainderState.COMPLETE
                current_term_str = ''
            else:
                # Although this is a complete terminal, it may happen that this may be just prefix of some other terminal
                # e.g., 'de' may seem like a variable name that is complete, but it may be just a prefix of 'def'
                current_term_str = self.parsed_lexer_tokens[-1].value
                remainder_state = RemainderState.MAYBE_COMPLETE
                final_terminal = self.parsed_lexer_tokens[-1].type
        else:
            # When the code is empty
            remainder_state = RemainderState.COMPLETE
            current_term_str = ''
        return remainder_state, current_term_str, final_terminal
    
    def _accepts(self, interactive_parser: InteractiveParser) -> set:
        accepts = interactive_parser.accepts()
        return accepts
    
    def _handle_parsing_error(self, lexer_tokens, token, error):
        """
        Handles the error that occurs when the lexer token is not parsed correctly.
        1. If the final token is not parsed correctly, then it is okay.
        2. If a non-final token is not parsed correctly, then it is an issue. We log the warning in that case. 
        """
        if token != lexer_tokens[-1]:
            raise error
        else:
            # If it is the final token that gave the error, then it is okay
            self.cur_ac_terminals = self.next_ac_terminals
            self.next_ac_terminals = set()

    def _update_symbol_pos_map_terminals(self, lexer_tokens):
        """Record the char span of each newly-COMPLETED terminal into symbol_pos_map.

        Ported from IterGen. PURE SIDE-RECORD: only writes to self.symbol_pos_map;
        reads nothing the accept-set path depends on. The last lexer token is
        excluded because it may still grow into a longer terminal.
        """
        if len(lexer_tokens) > len(self.parsed_lexer_tokens):
            len_parsed = len(self.parsed_lexer_tokens)

            # parsed_lexer_tokens excludes IGNORED tokens; skip the first len_parsed
            # NON-IGNORED tokens to land at the first unparsed terminal.
            start_idx = 0
            cnt_non_ignore = 0
            while cnt_non_ignore < len_parsed:
                if lexer_tokens[start_idx].type != 'IGNORED':
                    cnt_non_ignore += 1
                start_idx += 1

            # Don't record the last lexer token (it may extend in the future).
            start_idx -= 1
            end_idx = len(lexer_tokens) - 1

            for idx in range(start_idx, end_idx):
                if lexer_tokens[idx].type != 'IGNORED':
                    self.symbol_pos_map.add_symbol_pos(
                        lexer_tokens[idx].type,
                        pos=(lexer_tokens[idx].start_pos, lexer_tokens[idx].end_pos),
                    )

    def _update_symbol_pos_map_nonterminals(self, parser_state: ParserState, token: Token):
        """Record the char span of each NON-TERMINAL that would reduce on `token`.

        Ported from IterGen. PURE SIDE-RECORD: it simulates the LALR reduce/shift
        actions on DEEP COPIES of the state and value stacks, so it never mutates
        the live parser_state, the interactive parser, or the accept-sets. `end_pos`
        is the start of the incoming token (= end of the reduced non-terminal).
        """
        end_pos = token.start_pos

        # Copy the parser state (never mutate the live stacks).
        state_stack = copy.deepcopy(parser_state.state_stack)
        value_stack = copy.deepcopy(parser_state.value_stack)

        states = parser_state.parse_conf.states
        callbacks = parser_state.parse_conf.callbacks

        while True:
            state = state_stack[-1]

            if token.type in states[state]:
                action, arg = states[state][token.type]
            elif token.type == 'IGNORED':
                possible_rules = set()
                for term, (action, rule) in states[state].items():
                    if action != Reduce:
                        break
                    possible_rules.add(rule)

                if len(possible_rules) == 1:
                    rule = list(possible_rules)[0]
                    action = Reduce
                    arg = rule
                else:
                    break
            else:
                break

            if action is Reduce:
                rule = arg
                size = len(rule.expansion)
                if size:
                    s = value_stack[-size:]
                    del state_stack[-size:]
                    del value_stack[-size:]
                else:
                    s = []

                assert end_pos is not None
                if type(rule.origin.name) == Token:
                    start_pos = self._get_nonterminal_start_pos(s)
                    self.symbol_pos_map.add_symbol_pos(
                        rule.origin.name.value,
                        pos=(start_pos, end_pos),
                    )

                value = callbacks[rule](s) if callbacks else s

                _, new_state = states[state_stack[-1]][rule.origin.name]
                state_stack.append(new_state)
                value_stack.append(value)
            else:
                break

    def _get_nonterminal_start_pos(self, s: Iterable[Tree]) -> int:
        for item in s:
            if type(item) == Token:
                return item.start_pos
            elif item is not None:
                # A tree node carries its span in .meta
                return item.meta.start_pos
        return -1  # should not happen

    def _get_nonterminal_end_pos(self, s: Iterable[Tree]) -> int:
        for item in reversed(s):
            if type(item) == Token:
                return item.end_pos
            elif item is not None:
                return item.meta.end_pos
        return -1

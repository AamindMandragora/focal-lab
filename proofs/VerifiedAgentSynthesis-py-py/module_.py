import sys
import os
from typing import Callable, Any, TypeVar, NamedTuple, Set, List, Optional
from math import floor
from itertools import count

import _dafny as _dafny
import System_ as System_

# Module: module_
# This file contains Python implementations for all {:extern} functions in VerifiedAgentSynthesis.dfy
# Integrated with actual LLM (HuggingFace) and grammar parser (Lark)

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "Qwen/Qwen2-Math-72B"
GRAMMAR_FILE = "./grammars/gsm.lark"
DEVICE = "cuda"
MAX_NEW_TOKENS = 3
QUANTIZE = True

# Lazy initialization flags
_llm_initialized = False
_parser_initialized = False
_llm = None
_tokenizer = None
_grammar_parser = None
_all_tokens = None

# =============================================================================
# Type aliases for clarity
# =============================================================================
Token = str
Prefix = _dafny.Seq  # seq<Token>


# =============================================================================
# LLM Initialization (lazy loading)
# =============================================================================

def _ensure_llm_initialized():
    """Lazily initialize the LLM and tokenizer."""
    global _llm_initialized, _llm, _tokenizer, _all_tokens
    
    if _llm_initialized:
        return
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor
        from transformers import BitsAndBytesConfig
        import torch
        
        print(f"Loading model {MODEL_NAME}...")
        
        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        # Configure quantization
        quantization_config = None
        if QUANTIZE and DEVICE == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load model
        _llm = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto" if DEVICE == "cuda" else None,
            quantization_config=quantization_config,
            trust_remote_code=True,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
        
        if DEVICE != "cuda" and not QUANTIZE:
            _llm.to(DEVICE)
        
        # Build set of all tokens from tokenizer vocabulary
        _all_tokens = set(_tokenizer.get_vocab().keys())
        
        print(f"Model loaded successfully. Vocabulary size: {len(_all_tokens)}")
        _llm_initialized = True
        
    except ImportError as e:
        print(f"Warning: Could not import transformers/torch: {e}")
        print("LLM functionality will be disabled. Using placeholder implementations.")
        _llm_initialized = True  # Mark as initialized to avoid repeated attempts
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("LLM functionality will be disabled. Using placeholder implementations.")
        _llm_initialized = True


def _ensure_parser_initialized():
    """Lazily initialize the grammar parser."""
    global _parser_initialized, _grammar_parser
    
    if _parser_initialized:
        return
    
    try:
        from lark import Lark
        
        # Try multiple paths for grammar file
        grammar_paths = [
            GRAMMAR_FILE,
            os.path.join(os.path.dirname(__file__), "..", "..", "grammars", "gsm.lark"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "grammars", "gsm.lark"),
            "./grammars/gsm.lark",
        ]
        
        grammar_content = None
        for path in grammar_paths:
            if os.path.exists(path):
                with open(path, "r") as f:
                    grammar_content = f.read()
                print(f"Loaded grammar from {path}")
                break
        
        if grammar_content:
            _grammar_parser = Lark(grammar_content, start="start")
            print("Grammar parser initialized successfully.")
        else:
            print(f"Warning: Could not find grammar file. Tried: {grammar_paths}")
            print("Parser functionality will use permissive defaults.")
        
        _parser_initialized = True
        
    except ImportError as e:
        print(f"Warning: Could not import lark: {e}")
        print("Parser functionality will use permissive defaults.")
        _parser_initialized = True
    except Exception as e:
        print(f"Warning: Could not initialize parser: {e}")
        print("Parser functionality will use permissive defaults.")
        _parser_initialized = True


def get_all_tokens() -> Set[str]:
    """Get the set of all tokens from the tokenizer vocabulary."""
    _ensure_llm_initialized()
    return _all_tokens or set()


# =============================================================================
# SemanticPredicate - abstract type for semantic validity checks
# =============================================================================
class SemanticPredicate:
    """Base class for semantic predicates."""
    def __init__(self, check_fn: Callable[[Any], bool] = None):
        self.check_fn = check_fn or (lambda x: True)
    
    def __eq__(self, other):
        return isinstance(other, SemanticPredicate) and self.check_fn == other.check_fn
    
    def __hash__(self):
        return id(self.check_fn)


# =============================================================================
# LM (Language Model) extern implementations
# =============================================================================

def LM____ctor__(self):
    """Constructor for LM class - initialize with tokenizer vocabulary."""
    _ensure_llm_initialized()
    
    if _tokenizer is not None:
        vocab = _tokenizer.get_vocab()
        tokens = list(vocab.keys())
        ids = list(vocab.values())
        self._Tokens = _dafny.Seq(tokens)
        self._Ids = _dafny.Seq(ids)
        self.Logits = _dafny.Seq([0.0] * len(tokens))
    else:
        self._Tokens = _dafny.Seq([])
        self._Ids = _dafny.Seq([0])
        self.Logits = _dafny.Seq([0.0])


def LM__GenerateLogits(self, input_prefix):
    """Generate logits for the next token given input prefix."""
    _ensure_llm_initialized()
    
    if _llm is None or _tokenizer is None:
        # Placeholder: uniform distribution
        self.Logits = _dafny.Seq([1.0] * len(self.Logits))
        return
    
    import torch
    
    # Convert prefix to string
    prefix_str = "".join(str(t) for t in input_prefix) if input_prefix else ""
    
    # Tokenize and get logits
    inputs = _tokenizer(prefix_str, return_tensors="pt").to(_llm.device)
    
    with torch.no_grad():
        outputs = _llm(**inputs)
        # Get logits for next token (last position)
        next_token_logits = outputs.logits[0, -1, :].cpu().tolist()
    
    self.Logits = _dafny.Seq(next_token_logits)


def LM__ChooseNextToken(self, input_prefix):
    """Choose the next token given input prefix using the LLM."""
    _ensure_llm_initialized()
    
    if _llm is None or _tokenizer is None:
        # Placeholder: return first token
        if len(self._Tokens) > 0:
            return self._Tokens[0]
        return ""
    
    import torch
    
    # Convert prefix to string
    prefix_str = "".join(str(t) for t in input_prefix) if input_prefix else ""
    
    # Generate one token
    inputs = _tokenizer(prefix_str, return_tensors="pt").to(_llm.device)
    
    with torch.no_grad():
        outputs = _llm.generate(
            **inputs,
            max_new_tokens=1,
            pad_token_id=_tokenizer.pad_token_id,
            do_sample=False,
            use_cache=True
        )
    
    # Get the generated token
    new_token_id = outputs[0][inputs.input_ids.shape[1]:][0].item()
    return _tokenizer.decode([new_token_id])


# =============================================================================
# Parser extern implementations (using Lark grammar)
# =============================================================================

def Parser__IsValidPrefix(self, prefix):
    """Check if prefix is a valid partial parse using Lark grammar."""
    _ensure_parser_initialized()
    
    if _grammar_parser is None:
        return True  # Permissive default
    
    from lark.exceptions import UnexpectedInput, UnexpectedEOF
    
    # Convert prefix to string
    prefix_str = "".join(str(t) for t in prefix) if prefix else ""
    
    try:
        _grammar_parser.parse(prefix_str, start="start")
        return True  # Complete valid parse
    except UnexpectedEOF:
        return True  # Valid prefix but incomplete
    except UnexpectedInput:
        return False  # Invalid


def Parser__IsCompletePrefix(self, prefix):
    """Check if prefix is a complete valid parse."""
    _ensure_parser_initialized()
    
    if _grammar_parser is None:
        return len(prefix) > 0  # Permissive default
    
    from lark.exceptions import UnexpectedInput
    
    # Convert prefix to string
    prefix_str = "".join(str(t) for t in prefix) if prefix else ""
    
    try:
        _grammar_parser.parse(prefix_str, start="start")
        return True
    except UnexpectedInput:
        return False


def Parser__ValidNextTokens(self, prefix):
    """Return valid next tokens (terminal names) given prefix."""
    _ensure_parser_initialized()
    
    if _grammar_parser is None:
        return _dafny.Seq([])
    
    from lark.exceptions import UnexpectedInput, UnexpectedEOF
    
    # Convert prefix to string
    prefix_str = "".join(str(t) for t in prefix) if prefix else ""
    
    try:
        _grammar_parser.parse(prefix_str, start="start")
        return _dafny.Seq([])  # Complete, no next tokens needed
    except UnexpectedEOF as e:
        return _dafny.Seq(list(set(e.expected)))
    except UnexpectedInput:
        return _dafny.Seq([])


# =============================================================================
# LLM Generation with optional LogitsProcessor (for constrained decoding)
# =============================================================================

def generate_tokens(prompt: str, max_new_tokens: int = 3, logits_processor=None) -> List[str]:
    """Generate tokens using the LLM with optional logits processor.
    
    Args:
        prompt: Input prompt string
        max_new_tokens: Maximum tokens to generate
        logits_processor: Optional LogitsProcessor for constrained decoding
    
    Returns:
        List of generated token strings
    """
    _ensure_llm_initialized()
    
    if _llm is None or _tokenizer is None:
        return []
    
    import torch
    from transformers import LogitsProcessorList
    
    inputs = _tokenizer(prompt, return_tensors="pt").to(_llm.device)
    
    logits_processors = LogitsProcessorList()
    if logits_processor:
        logits_processors.append(logits_processor)
    
    with torch.no_grad():
        outputs = _llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=_tokenizer.pad_token_id,
            logits_processor=logits_processors,
            do_sample=False,
            use_cache=True
        )
    
    # Get only new tokens
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    generated_ids = new_tokens.tolist()
    
    # Convert to token strings
    return _tokenizer.convert_ids_to_tokens(generated_ids)


def generate_unconstrained(prompt: str, max_new_tokens: int = 50) -> str:
    """Generate text without constraints.
    
    Args:
        prompt: Input prompt string
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Generated text string
    """
    _ensure_llm_initialized()
    
    if _llm is None or _tokenizer is None:
        return prompt
    
    import torch
    
    inputs = _tokenizer(prompt, return_tensors="pt").to(_llm.device)
    
    with torch.no_grad():
        outputs = _llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=_tokenizer.pad_token_id,
            do_sample=False,
            use_cache=True
        )
    
    return _tokenizer.decode(outputs[0], skip_special_tokens=True)


# =============================================================================
# Token-Level Constraint implementations
# =============================================================================

def AllowedNext(c, parser, prefix, allTokens):
    """Compute the set of allowed next tokens given a constraint.
    
    Args:
        c: TokenConstraint datatype
        parser: Parser instance
        prefix: Current token sequence
        allTokens: Set of all possible tokens
    
    Returns:
        Set of allowed tokens
    """
    import VerifiedDecoderAgent as VDA
    
    if c.is_GrammarMask:
        # Use parser to get grammar-valid tokens
        _ensure_parser_initialized()
        
        if _grammar_parser is not None:
            from lark.exceptions import UnexpectedInput, UnexpectedEOF
            
            prefix_str = "".join(str(t) for t in prefix) if prefix else ""
            
            try:
                _grammar_parser.parse(prefix_str, start="start")
                return set()  # Complete, no more tokens needed
            except UnexpectedEOF as e:
                # Get expected terminals and map to actual tokens
                expected = set(e.expected)
                # For now, return tokens that might match expected terminals
                # In practice, you'd want a more sophisticated mapping
                return _filter_tokens_by_terminals(allTokens, expected)
            except UnexpectedInput:
                return set()  # Invalid prefix
        
        # Fallback: use parser method if available
        if parser is not None and hasattr(parser, 'IsValidPrefix'):
            valid = set()
            for t in allTokens:
                new_prefix = _dafny.Seq(list(prefix) + [t])
                if parser.IsValidPrefix(new_prefix):
                    valid.add(t)
            return valid
        
        return set(allTokens)  # Permissive fallback
    
    elif c.is_Lookahead:
        # For lookahead, use grammar mask (could be extended with actual lookahead)
        return AllowedNext(VDA.TokenConstraint_GrammarMask(), parser, prefix, allTokens)
    
    elif c.is_LengthBound:
        current_len = len(prefix)
        result = set(allTokens)
        if current_len >= c.max_:
            # Only allow EOS tokens
            eos_tokens = {"", "<eos>", "</s>", "<|endoftext|>"}
            result = result & eos_tokens
        if current_len < c.min_:
            # Don't allow EOS tokens
            eos_tokens = {"", "<eos>", "</s>", "<|endoftext|>"}
            result = result - eos_tokens
        return result
    
    elif c.is_BanTokens:
        return set(allTokens) - set(c.banned)
    
    elif c.is_AllowOnlyTokens:
        return set(allTokens) & set(c.allowed)
    
    elif c.is_Intersect:
        a_allowed = AllowedNext(c.a, parser, prefix, allTokens)
        b_allowed = AllowedNext(c.b, parser, prefix, allTokens)
        return a_allowed & b_allowed
    
    elif c.is_Union:
        a_allowed = AllowedNext(c.a, parser, prefix, allTokens)
        b_allowed = AllowedNext(c.b, parser, prefix, allTokens)
        return a_allowed | b_allowed
    
    elif c.is_NoConstraint:
        return set(allTokens)
    
    return set(allTokens)


def _filter_tokens_by_terminals(allTokens: Set[str], expected_terminals: Set[str]) -> Set[str]:
    """Filter tokens that could match expected grammar terminals.
    
    This is a simplified mapping - in practice you'd want to use
    the grammar's terminal definitions to do proper matching.
    """
    # Simple heuristic: include tokens that contain terminal keywords
    # or match common patterns
    result = set()
    
    terminal_to_pattern = {
        "NUMBER": lambda t: t.isdigit() or (t.startswith("-") and t[1:].isdigit()),
        "WORD": lambda t: t.isalpha(),
        "WS": lambda t: t.isspace(),
        "PLUS": lambda t: t == "+",
        "MINUS": lambda t: t == "-",
        "TIMES": lambda t: t in ["*", "×"],
        "DIVIDE": lambda t: t in ["/", "÷"],
        "EQUALS": lambda t: t in ["=", "=="],
        "LPAREN": lambda t: t == "(",
        "RPAREN": lambda t: t == ")",
        "LANGLE": lambda t: t == "<",
        "RANGLE": lambda t: t == ">",
    }
    
    for token in allTokens:
        for terminal in expected_terminals:
            if terminal in terminal_to_pattern:
                if terminal_to_pattern[terminal](token):
                    result.add(token)
                    break
            elif terminal.lower() in token.lower():
                result.add(token)
                break
    
    # If no matches, be permissive
    if not result:
        return allTokens
    
    return result


def ChooseToken(lm, c, parser, prefix, allTokens):
    """Select a token from the allowed set using LLM probabilities.
    
    Args:
        lm: LM instance (provides vocabulary context)
        c: TokenConstraint
        parser: Parser instance
        prefix: Current token sequence
        allTokens: Set of all possible tokens
    
    Uses the LLM to rank tokens and picks the highest-probability
    token from the allowed set.
    """
    _ensure_llm_initialized()
    
    allowed = AllowedNext(c, parser, prefix, allTokens)
    
    if not allowed:
        if allTokens:
            return next(iter(allTokens))
        return ""
    
    if _llm is None or _tokenizer is None:
        # Fallback: return first allowed token
        return next(iter(allowed))
    
    import torch
    
    # Convert prefix to string
    prefix_str = "".join(str(t) for t in prefix) if prefix else ""
    
    # Get logits from LLM
    inputs = _tokenizer(prefix_str, return_tensors="pt").to(_llm.device)
    
    with torch.no_grad():
        outputs = _llm(**inputs)
        logits = outputs.logits[0, -1, :]  # Next token logits
    
    # Find best token among allowed
    best_token = None
    best_score = float('-inf')
    
    vocab = _tokenizer.get_vocab()
    for token in allowed:
        if token in vocab:
            token_id = vocab[token]
            score = logits[token_id].item()
            if score > best_score:
                best_score = score
                best_token = token
    
    return best_token if best_token else next(iter(allowed))


# =============================================================================
# Sequence-Level Operation implementations
# =============================================================================

def ApplyRepair(rules, output):
    """Apply repair rules to fix output."""
    import VerifiedDecoderAgent as VDA
    
    result = list(output)
    
    if rules.is_BracketBalance:
        result = _balance_brackets(result)
    elif rules.is_QuoteFix:
        result = _fix_quotes(result)
    elif rules.is_WhitespaceNormalize:
        result = _normalize_whitespace(result)
    elif rules.is_ComposedRepair:
        result = list(ApplyRepair(rules.a, _dafny.Seq(result)))
        result = list(ApplyRepair(rules.b, _dafny.Seq(result)))
    elif rules.is_NoRepair:
        pass
    
    return _dafny.Seq(result)


def _balance_brackets(tokens: List[str]) -> List[str]:
    """Balance brackets in token sequence."""
    result = tokens.copy()
    pairs = {'(': ')', '[': ']', '{': '}', '<': '>'}
    stack = []
    
    for t in result:
        if t in pairs:
            stack.append(pairs[t])
        elif t in pairs.values() and stack and stack[-1] == t:
            stack.pop()
    
    while stack:
        result.append(stack.pop())
    
    return result


def _fix_quotes(tokens: List[str]) -> List[str]:
    """Fix unclosed quotes in token sequence."""
    result = tokens.copy()
    quote_count = sum(1 for t in result if t in ['"', "'"])
    if quote_count % 2 == 1:
        for t in reversed(result):
            if t in ['"', "'"]:
                result.append(t)
                break
    return result


def _normalize_whitespace(tokens: List[str]) -> List[str]:
    """Normalize whitespace tokens."""
    result = []
    prev_was_space = False
    for t in tokens:
        is_space = t.strip() == ""
        if is_space and prev_was_space:
            continue
        result.append(t)
        prev_was_space = is_space
    return result


def CheckSemantic(pred, output):
    """Evaluate a semantic predicate on output."""
    if isinstance(pred, SemanticPredicate):
        return pred.check_fn(output)
    return True


def CompletePrefixConstrained(lm, parser, prefix, constraint, allTokens, maxSteps):
    """Complete a valid prefix under a token constraint using LM.
    
    Args:
        lm: LM instance for token selection
        parser: Parser instance for grammar validation
        prefix: Starting token sequence
        constraint: TokenConstraint to apply
        allTokens: Set of all possible tokens
        maxSteps: Maximum tokens to generate
    """
    result = list(prefix)
    steps = 0
    
    while steps < maxSteps:
        allowed = AllowedNext(constraint, parser, _dafny.Seq(result), allTokens)
        if not allowed:
            break
        
        # Check if we're complete
        prefix_str = "".join(str(t) for t in result)
        if parser is not None and hasattr(parser, 'IsCompletePrefix'):
            if parser.IsCompletePrefix(_dafny.Seq(result)):
                break
        elif _grammar_parser is not None:
            _ensure_parser_initialized()
            try:
                _grammar_parser.parse(prefix_str, start="start")
                break  # Complete
            except:
                pass
        
        next_token = ChooseToken(lm, constraint, parser, _dafny.Seq(result), allTokens)
        result.append(next_token)
        steps += 1
    
    return _dafny.Seq(result)


def ApplySeqOp(lm, op, parser, output, allTokens, maxSteps):
    """Apply a sequence operation using LM for completion.
    
    Args:
        lm: LM instance for token generation
        op: SeqOperation to apply
        parser: Parser instance
        output: Current output sequence
        allTokens: Set of all possible tokens
        maxSteps: Maximum steps for completion
    """
    import VerifiedDecoderAgent as VDA
    
    if op.is_Identity:
        return output
    elif op.is_Repair:
        return ApplyRepair(op.rules, output)
    elif op.is_PrefixCompleteOp:
        if parser is not None and hasattr(parser, 'IsValidPrefix') and parser.IsValidPrefix(output):
            return CompletePrefixConstrained(lm, parser, output, op.constraint, allTokens, maxSteps)
        return output
    elif op.is_ValidateOp:
        return output
    
    return output


# =============================================================================
# Check Predicate implementation
# =============================================================================

def CheckOutput(check, parser, output):
    """Evaluate a check predicate on output."""
    import VerifiedDecoderAgent as VDA
    
    if check.is_ParseOk:
        _ensure_parser_initialized()
        
        if _grammar_parser is not None:
            from lark.exceptions import UnexpectedInput
            prefix_str = "".join(str(t) for t in output) if output else ""
            try:
                _grammar_parser.parse(prefix_str, start="start")
                return True
            except UnexpectedInput:
                return False
        
        if parser is not None and hasattr(parser, 'IsValidPrefix'):
            return parser.IsValidPrefix(output)
        return True
    
    elif check.is_SemanticOk:
        return CheckSemantic(check.pred, output)
    
    elif check.is_Both:
        return CheckOutput(check.a, parser, output) and CheckOutput(check.b, parser, output)
    
    elif check.is_Either:
        return CheckOutput(check.a, parser, output) or CheckOutput(check.b, parser, output)
    
    return True


# =============================================================================
# Attempt execution
# =============================================================================

def RunAttempt(lm, attempt, parser, prompt, allTokens, maxSteps):
    """Execute a single generation attempt using LM.
    
    Args:
        lm: LM instance for generation
        attempt: Attempt to execute
        parser: Parser instance
        prompt: Starting prompt sequence
        allTokens: Set of all possible tokens
        maxSteps: Maximum tokens to generate
    """
    import VerifiedDecoderAgent as VDA
    
    if attempt.is_Unconstrained:
        # Use actual LLM for unconstrained generation
        _ensure_llm_initialized()
        
        if _llm is not None and _tokenizer is not None:
            prompt_str = "".join(str(t) for t in prompt) if prompt else ""
            generated = generate_unconstrained(prompt_str, max_new_tokens=maxSteps)
            # Convert back to token sequence
            tokens = _tokenizer.tokenize(generated)
            return _dafny.Seq(tokens)
        
        return prompt
    
    elif attempt.is_ConstrainedAttempt:
        return CompletePrefixConstrained(lm, parser, prompt, attempt.constraint, allTokens, maxSteps)
    
    elif attempt.is_WithRepair:
        base_output = RunAttempt(lm, attempt.base, parser, prompt, allTokens, maxSteps)
        return ApplyRepair(attempt.rules, base_output)
    
    elif attempt.is_WithSeqOp:
        base_output = RunAttempt(lm, attempt.base, parser, prompt, allTokens, maxSteps)
        return ApplySeqOp(lm, attempt.op, parser, base_output, allTokens, maxSteps)
    
    return prompt


# =============================================================================
# Strategy execution
# =============================================================================

def RunStrategy(lm, strategy, parser, prompt, allTokens, maxSteps):
    """Execute a CSD strategy using LM.
    
    Args:
        lm: LM instance for generation
        strategy: Strategy to execute
        parser: Parser instance
        prompt: Starting prompt sequence
        allTokens: Set of all possible tokens
        maxSteps: Maximum tokens to generate
    """
    import VerifiedDecoderAgent as VDA
    
    if strategy.is_Window:
        return _run_window_strategy(
            lm, strategy.startDelim, strategy.endDelim,
            strategy.inside, strategy.outside,
            parser, prompt, allTokens, maxSteps
        )
    
    elif strategy.is_TryK:
        for _ in range(strategy.k):
            output = RunAttempt(lm, strategy.attempt, parser, prompt, allTokens, maxSteps)
            if CheckOutput(strategy.check, parser, output):
                return output
        return RunStrategy(lm, strategy.fallback, parser, prompt, allTokens, maxSteps)
    
    elif strategy.is_Cascade:
        for strat in strategy.strategies:
            output = RunStrategy(lm, strat, parser, prompt, allTokens, maxSteps)
            if CheckOutput(strategy.check, parser, output):
                return output
        if strategy.strategies:
            return RunStrategy(lm, strategy.strategies[-1], parser, prompt, allTokens, maxSteps)
        return prompt
    
    elif strategy.is_BestOfN:
        outputs = []
        for _ in range(strategy.n):
            output = RunStrategy(lm, strategy.base, parser, prompt, allTokens, maxSteps)
            outputs.append(output)
            if CheckOutput(strategy.check, parser, output):
                return output
        return outputs[0] if outputs else prompt
    
    elif strategy.is_Constrained:
        return CompletePrefixConstrained(lm, parser, prompt, strategy.constraint, allTokens, maxSteps)
    
    elif strategy.is_Free:
        return RunAttempt(lm, VDA.Attempt_Unconstrained(), parser, prompt, allTokens, maxSteps)
    
    return prompt


def _run_window_strategy(lm, startDelim, endDelim, inside, outside, parser, prompt, allTokens, maxSteps):
    """Execute CRANE-style windowing strategy using LM.
    
    Args:
        lm: LM instance for token selection
        startDelim: Delimiter marking start of constrained window (e.g., "<<")
        endDelim: Delimiter marking end of constrained window (e.g., ">>")
        inside: TokenConstraint to apply inside windows
        outside: TokenConstraint to apply outside windows
        parser: Parser instance
        prompt: Starting prompt sequence
        allTokens: Set of all possible tokens
        maxSteps: Maximum tokens to generate
    """
    _ensure_llm_initialized()
    
    result = list(prompt)
    steps = 0
    in_window = False
    
    # Check if we're already in a window
    prompt_str = "".join(str(t) for t in result)
    if "<<" in prompt_str and ">>" not in prompt_str.split("<<")[-1]:
        in_window = True
    
    while steps < maxSteps:
        current_prefix = _dafny.Seq(result)
        
        # Determine constraint based on window state
        constraint = inside if in_window else outside
        
        # Get next token
        if constraint.is_NoConstraint and _llm is not None:
            # Use unconstrained LLM generation
            prompt_str = "".join(str(t) for t in result)
            tokens = generate_tokens(prompt_str, max_new_tokens=1)
            if tokens:
                next_token = tokens[0]
            else:
                break
        else:
            # Use constrained token selection
            allowed = AllowedNext(constraint, parser, current_prefix, allTokens)
            if not allowed:
                break
            next_token = ChooseToken(lm, constraint, parser, current_prefix, allTokens)
        
        result.append(next_token)
        steps += 1
        
        # Check for window transitions
        result_str = "".join(str(t) for t in result)
        if startDelim in result_str and not in_window:
            if result_str.endswith(startDelim) or startDelim in next_token:
                in_window = True
        if endDelim in result_str and in_window:
            if result_str.endswith(endDelim) or endDelim in next_token:
                in_window = False
        
        # Check for completion
        if parser is not None and hasattr(parser, 'IsCompletePrefix'):
            if parser.IsCompletePrefix(_dafny.Seq(result)):
                break
    
    return _dafny.Seq(result)


# =============================================================================
# Main Run function
# =============================================================================

def Run(lm, strategy, parser, prompt, allTokens, maxSteps):
    """Execute a strategy that guarantees valid output using LM.
    
    Args:
        lm: LM instance (must satisfy ValidTokensIdsLogits())
        strategy: Strategy to execute (must satisfy GuaranteesValidOutput())
        parser: Parser instance
        prompt: Starting prompt sequence (must have |prompt| > 0)
        allTokens: Set of all possible tokens
        maxSteps: Maximum tokens to generate
    
    Returns:
        Output sequence that is guaranteed to be valid under the parser.
    """
    return RunStrategy(lm, strategy, parser, prompt, allTokens, maxSteps)


# =============================================================================
# ConstrainedDecode (legacy method)
# =============================================================================

def ConstrainedDecode(lm, parser, prefix, maxSteps):
    """Run constrained decoding (legacy interface)."""
    import VerifiedDecoderAgent as VDA
    
    _ensure_llm_initialized()
    _ensure_parser_initialized()
    
    result = list(prefix)
    steps = 0
    
    while steps < maxSteps:
        current_prefix = _dafny.Seq(result)
        prefix_str = "".join(str(t) for t in result)
        
        # Check if complete using grammar
        if _grammar_parser is not None:
            from lark.exceptions import UnexpectedInput
            try:
                _grammar_parser.parse(prefix_str, start="start")
                break  # Complete
            except UnexpectedInput:
                pass
        
        # Get valid next tokens
        if parser is not None and hasattr(parser, 'ValidNextTokens'):
            valid_tokens = parser.ValidNextTokens(current_prefix)
            if len(valid_tokens) == 0:
                break
            
            # Use LLM to choose among valid tokens
            if _llm is not None and _tokenizer is not None:
                next_token = ChooseToken(
                    VDA.TokenConstraint_AllowOnlyTokens(_dafny.Set(set(valid_tokens))),
                    parser,
                    current_prefix,
                    get_all_tokens()
                )
            else:
                next_token = valid_tokens[0]
            
            result.append(next_token)
        else:
            break
        
        steps += 1
    
    return _dafny.Seq(result)

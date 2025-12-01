import _dafny as _dafny
import System_ as System_

from syncode import Syncode, Grammar
import syncode.common as common
from transformers import AutoTokenizer, Qwen2TokenizerFast
from lark import Lark
from lark.exceptions import UnexpectedInput, UnexpectedEOF
from torch import Tensor

MODEL_NAME = "Qwen/Qwen2-Math-72B"
GRAMMAR = "gsm.lark"
DEVICE = "cuda"
MAX_NEW_TOKENS = 3

# Two Syncode modes
llm_constrained = Syncode(
    model=MODEL_NAME,
    grammar=GRAMMAR,
    parse_output_only=True,
    mode='grammar_strict',
    max_new_tokens=MAX_NEW_TOKENS,
    device=DEVICE,
)

llm = Syncode(
    model=MODEL_NAME,
    parse_output_only=True,
    mode='original',
    max_new_tokens=MAX_NEW_TOKENS,
    device=DEVICE,
)

# Tokenizer
tokenizer : Qwen2TokenizerFast = common.load_tokenizer(MODEL_NAME)

# Grammar + loose parser (we validate only the generated code / prefix)
grammar = Grammar(GRAMMAR)
constrained = Lark(grammar.ebnf)

def Parser__ValidPrefix(prefix: str):
    """Checks if prefix is valid under the current grammar."""
    try:
        constrained.parse(prefix, start="gsm")
        return True
    except UnexpectedInput as e:
        return isinstance(e, UnexpectedEOF)

def Parser__IsComplete(prefix: str):
    """Checks if prefix is complete under the current grammar."""
    try:
        constrained.parse(prefix, start="gsm")
        return True
    except UnexpectedInput:
        return False

def Parser__AllowedNext(prefix: str):
    """Return the set of expected terminals for the current prefix."""
    try:
        constrained.parse(prefix, start="gsm")
        return []
    except UnexpectedEOF as e:
        return list(set(e.expected))
    except UnexpectedInput:
        return []

def Generator__ChooseToken(prefix: str, allowed: list[str], pointer: int, is_constrained: bool):
    """
    Generates token based on prefix and properly detokenizes it.
    """
    curr_gen = prefix[pointer:]
    if "<<" in curr_gen:
        is_constrained = True
    else:
        is_constrained = False
    # Generate tokens
    if is_constrained:
        tokens = llm_constrained.model.generate_grammar_constrained_completion(prefix, batch_size=1, stop_words=None, return_token_ids=True, debug=False)
    else:
        tokens = llm.model.generate_grammar_constrained_completion(prefix, batch_size=1, stop_words=None, return_token_ids=True, debug=False)
    # Get first token (or first allowed token if constrained)
    if allowed and token not in allowed:
        token = next(iter(allowed))
    # Use proper detokenization
    tokens = tokenizer.convert_ids_to_tokens(tokens[0][1].data.tolist())[-1 * MAX_NEW_TOKENS:]
    print(tokens)
    return tokens
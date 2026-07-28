import sys
from typing import Callable, Any, TypeVar, NamedTuple
from math import floor
from itertools import count

import module_ as module_
import _dafny as _dafny
import System_ as System_
import VerifiedDecoderAgent as VerifiedDecoderAgent

# Module: GeneratedCSD

class default__:
    def  __init__(self):
        pass

    @staticmethod
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
        generated: _dafny.Seq = _dafny.Seq({})
        insideConstrainedOut: bool = False
        currentConstrainedOut: _dafny.Seq = _dafny.Seq({})
        cost: int = int(0)
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = generatedPrefix
        insideConstrainedOut = False
        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step in plain English. After your reasoning, end with exactly one final line of the form: The answer is <<EXPR>>. EXPR must be a SINGLE Python expression that evaluates to the numeric answer. Strict rules for EXPR: (1) Use ONLY the exact variable names that appear in braces in the problem (for example y, t, d, n_1, k_2, w3, total) — do not invent new variables and do not include braces. (2) Use Python integer operators: // for integer division (never /), * for multiplication, + and -, % for remainder, and parentheses. (3) When the answer is conceptually a count of whole items, prefer // over /; if a non-integer intermediate is unavoidable and the final answer must be an integer, wrap the whole expression in int(...). (4) Do NOT put units, words, $, %, LaTeX (\\frac, \\boxed), or commas inside <<...>>. (5) Only the FINAL <<...>> span is graded — make it the complete final formula, not a partial step. (6) Re-read the problem before writing EXPR to confirm you used the right variables and the correct arithmetic operations. Example final line: The answer is <<(n1*w1 + n2*w2 + n3*w3 + n4*w4)//total>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_2_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_2_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    cost = (cost) + (1)
                    if (d_2_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                    pass
            pass
        return generated, insideConstrainedOut, currentConstrainedOut, cost


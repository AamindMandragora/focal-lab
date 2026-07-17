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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You will solve a math word problem whose variables appear in curly braces like {n}, {x}, {k_2}, {total}. Reason step by step in plain English using ordinary arithmetic words. After your reasoning, output exactly one final line of this form and nothing after it: The answer is <<EXPR>>. Rules for EXPR: (1) EXPR is a SINGLE Python expression that evaluates to the numeric answer. (2) Use the EXACT variable names from the problem WITHOUT braces (n, x, k_2, total, etc.). Do not invent new variables, do not use {} or $ or %. (3) If the answer must be a whole number, write the natural arithmetic with regular division `/` and wrap the whole expression in int(...). Prefer int(a*b/c) over (a*b)//c, because the evaluator's reference answers use the int(.../) form and these differ when division is not exact. Use // only when the problem literally describes floor/whole-batch counting and no int(...) wrapper is needed. (4) Use no LaTeX, no \\frac, no \\boxed, no commas inside numbers, no units, no words inside <<...>>. (5) Do NOT put <<...>> around intermediate calculations; only emit ONE <<...>> span, and it must hold the COMPLETE final formula, not a partial step. (6) Re-read the problem before writing the final line and double-check that every relevant quantity from the problem appears in EXPR. Worked format examples (use the same shape, not these numbers): The answer is <<int(k * y / (x * 12) * 100)>>  // percentage, integer result. The answer is <<x * k * (12 // n)>>  // whole batches per year. The answer is <<int(n - n_1 - n_2 - 2 * n_2)>>  // plain subtraction wrapped in int. End your output immediately after the final period.")))
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


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
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out3_: int
        out0_, out1_, out2_, out3_ = default__.AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken)
        generated = out0_
        insideConstrainedOut = out1_
        currentConstrainedOut = out2_
        cost = out3_
        if ((maxSteps) > (0)) and ((cost) <= (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

    @staticmethod
    def AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
        generated: _dafny.Seq = _dafny.Seq({})
        insideConstrainedOut: bool = False
        currentConstrainedOut: _dafny.Seq = _dafny.Seq({})
        cost: int = int(0)
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        (d_0_helpers_).AppendTaskGuidance(lm, ((((((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step-by-step.\n"))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CRITICAL RULE FOR << >> SPANS:\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Each << >> pair must contain ONE COMPLETE arithmetic expression — never split a single computation across multiple << >> pairs.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CORRECT: <<total - n1 - n2>>  (complete expression in ONE span)\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WRONG:   <<total>> - <<n1>> - <<n2>>  (expression split across spans — NEVER do this)\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CORRECT: <<(n + n * mult) * 7>>  (full answer in ONE span)\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WRONG:   <<(mult + 1)>> * n * 7  (part of the answer outside the span — NEVER do this)\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Variable names inside << >> must NOT have curly braces: write n not {n}, n1 not {n1}, mult not {mult}.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "The LAST << >> span in your response must be the COMPLETE final answer as one arithmetic expression.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Allowed operators inside << >>: +  -  *  /  (  )\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Do NOT use ** for exponentiation or // for floor division inside << >>."))))
        d_1_steps_: int
        d_1_steps_ = 0
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_2_remaining_: int
            d_2_remaining_ = (maxSteps) - (d_1_steps_)
            d_3_closeBudget_: int
            if (d_2_remaining_) > (8):
                d_3_closeBudget_ = 8
            elif True:
                d_3_closeBudget_ = d_2_remaining_
            d_4_cg_: _dafny.Seq
            d_5_ci_: bool
            d_6_cc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_3_closeBudget_)
            d_4_cg_ = out0_
            d_5_ci_ = out1_
            d_6_cc_ = out2_
            generated = d_4_cg_
            insideConstrainedOut = d_5_ci_
            currentConstrainedOut = d_6_cc_
            d_1_steps_ = (d_1_steps_) + (d_3_closeBudget_)
        insideConstrainedOut = False
        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_7_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_7_next_ = out3_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_7_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


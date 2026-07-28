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
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. The problem text may contain symbolic placeholders such as {x}, {n}, {frac_1}, {total}; treat each placeholder NAME as a literal variable in your arithmetic (for example use n, x, frac_1 directly — do not substitute imaginary concrete numbers and do not switch to an unrelated tree-planting scenario). Wrap every intermediate computation in << >> delimiters using the form <<expr=value>> when both sides are known or <<expr>> for a symbolic expression, for example <<n - k*x>> or <<2+3=5>>. When the final quantity must be a whole count (number of items, trips, balls, etc.) wrap the expression in int(...), e.g. <<int(n*frac_1*frac_2)>>. Finish the response with a final << ... >> answer.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_2_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_2_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                            d_3_genStr_: _dafny.Seq
                            d_3_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
                            d_4_openCount_: int
                            d_4_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_3_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                            d_5_closeCount_: int
                            d_5_closeCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_3_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                            if (d_4_openCount_) > (d_5_closeCount_):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_6_closedGenerated_: _dafny.Seq
                        d_7_closedInside_: bool
                        d_8_closedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_6_closedGenerated_ = out1_
                        d_7_closedInside_ = out2_
                        d_8_closedCurrent_ = out3_
                        generated = d_6_closedGenerated_
                        insideConstrainedOut = d_7_closedInside_
                        currentConstrainedOut = d_8_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_10_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_10_next_ = out4_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_10_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_11_appendedGenerated_: _dafny.Seq
                            d_12_appendedInside_: bool
                            d_13_appendedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                            d_11_appendedGenerated_ = out5_
                            d_12_appendedInside_ = out6_
                            d_13_appendedCurrent_ = out7_
                            generated = d_11_appendedGenerated_
                            insideConstrainedOut = d_12_appendedInside_
                            currentConstrainedOut = d_13_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


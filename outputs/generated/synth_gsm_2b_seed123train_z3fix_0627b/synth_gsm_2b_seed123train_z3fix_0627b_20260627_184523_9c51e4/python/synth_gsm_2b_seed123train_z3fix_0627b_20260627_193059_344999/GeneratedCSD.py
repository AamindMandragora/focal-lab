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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Wrap ALL symbolic expressions and the final numeric/symbolic answer inside << >> delimiters. The answer inside << >> must be a valid arithmetic expression using only variables, numbers, and operators +,-,*,/,//,%,(). Do not include LaTeX, $, {, }, or other markup inside << >>. Example: <<n1 * p1 + n2 * p2>>")))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_remainingSteps_: int
                        d_2_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_2_remainingSteps_) <= (5):
                            raise _dafny.Break("0")
                        d_3_chunkBudget_: int
                        d_3_chunkBudget_ = (d_2_remainingSteps_) - (4)
                        if (d_3_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        d_4_genOut_: _dafny.Seq
                        d_5_stoppedOnOpen_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_genOut_ = out0_
                        d_5_stoppedOnOpen_ = out1_
                        d_6_stoppedOnEos_ = out2_
                        d_7_stepsUsed_ = out3_
                        generated = d_4_genOut_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        if d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_5_stoppedOnOpen_:
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            generated = out4_
                            insideConstrainedOut = out5_
                            currentConstrainedOut = out6_
                        elif True:
                            if ((d_1_steps_) + (4)) < (maxSteps):
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                generated = out7_
                                insideConstrainedOut = out8_
                                currentConstrainedOut = out9_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                    elif True:
                        d_8_remainingSteps_: int
                        d_8_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_8_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        if ((d_8_remainingSteps_) >= (4)) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                            d_9_innerBudget_: int
                            d_9_innerBudget_ = _dafny.euclidian_division(d_8_remainingSteps_, 2)
                            if (d_9_innerBudget_) == (0):
                                d_9_innerBudget_ = 1
                            if ((d_9_innerBudget_) + (2)) >= (d_8_remainingSteps_):
                                if (d_8_remainingSteps_) >= (3):
                                    d_9_innerBudget_ = (d_8_remainingSteps_) - (2)
                                elif True:
                                    d_9_innerBudget_ = 1
                            d_10_innerSteps_: int
                            d_10_innerSteps_ = 0
                            with _dafny.label("0_1_1_0"):
                                while (((d_10_innerSteps_) < (d_9_innerBudget_)) and ((d_1_steps_) < (maxSteps))) and (insideConstrainedOut):
                                    with _dafny.c_label("0_1_1_0"):
                                        if (parser).IsCompletePrefix(currentConstrainedOut):
                                            raise _dafny.Break("0_1_1_0")
                                        d_11_stableLen_: int
                                        d_11_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
                                        d_12_constrainedPrompt_: _dafny.Seq
                                        d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:d_11_stableLen_:]))
                                        d_13_next_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                        d_13_next_ = out10_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_10_innerSteps_ = (d_10_innerSteps_) + (1)
                                        if (d_13_next_) == (eosToken):
                                            raise _dafny.Break("0_1_1_0")
                                        elif True:
                                            d_14_ag_: _dafny.Seq
                                            d_15_ai_: bool
                                            d_16_ac_: _dafny.Seq
                                            out11_: _dafny.Seq
                                            out12_: bool
                                            out13_: _dafny.Seq
                                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                            d_14_ag_ = out11_
                                            d_15_ai_ = out12_
                                            d_16_ac_ = out13_
                                            generated = d_14_ag_
                                            insideConstrainedOut = d_15_ai_
                                            currentConstrainedOut = d_16_ac_
                                        pass
                                pass
                        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                            d_17_closeBudget_: int
                            d_17_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_18_cg_: _dafny.Seq
                            d_19_ci_: bool
                            d_20_cc_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget_)
                            d_18_cg_ = out14_
                            d_19_ci_ = out15_
                            d_20_cc_ = out16_
                            generated = d_18_cg_
                            insideConstrainedOut = d_19_ci_
                            currentConstrainedOut = d_20_cc_
                            d_1_steps_ = maxSteps
                        elif insideConstrainedOut:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


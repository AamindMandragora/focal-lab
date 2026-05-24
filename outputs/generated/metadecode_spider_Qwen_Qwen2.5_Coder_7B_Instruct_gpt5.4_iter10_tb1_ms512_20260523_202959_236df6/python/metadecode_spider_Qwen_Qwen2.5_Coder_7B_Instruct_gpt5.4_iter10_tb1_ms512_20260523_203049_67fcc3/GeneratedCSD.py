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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query enclosed in << and >>. Do not add explanation, prose, or extra text outside the query.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 8
        d_3_penaltyTokens_: _dafny.Seq
        d_3_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "```")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Explanation")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "explanation")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_openedGenerated_: _dafny.Seq
                        d_5_openedInside_: bool
                        d_6_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_4_openedGenerated_ = out0_
                        d_5_openedInside_ = out1_
                        d_6_openedCurrent_ = out2_
                        generated = d_4_openedGenerated_
                        insideConstrainedOut = d_5_openedInside_
                        currentConstrainedOut = d_6_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_7_closedGenerated_: _dafny.Seq
                        d_8_closedInside_: bool
                        d_9_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_7_closedGenerated_ = out3_
                        d_8_closedInside_ = out4_
                        d_9_closedCurrent_ = out5_
                        generated = d_7_closedGenerated_
                        insideConstrainedOut = d_8_closedInside_
                        currentConstrainedOut = d_9_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_11_validCount_: int
                        out6_: int
                        out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_11_validCount_ = out6_
                        if (d_11_validCount_) <= (d_2_narrowThreshold_):
                            d_12_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, d_3_penaltyTokens_, _dafny.BigRational('4e0'), eosToken)
                            d_12_next_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_13_appendedGenerated_: _dafny.Seq
                                d_14_appendedInside_: bool
                                d_15_appendedCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                d_13_appendedGenerated_ = out8_
                                d_14_appendedInside_ = out9_
                                d_15_appendedCurrent_ = out10_
                                generated = d_13_appendedGenerated_
                                insideConstrainedOut = d_14_appendedInside_
                                currentConstrainedOut = d_15_appendedCurrent_
                        elif True:
                            d_16_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_16_next_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_16_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_17_appendedGenerated_: _dafny.Seq
                                d_18_appendedInside_: bool
                                d_19_appendedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                d_17_appendedGenerated_ = out12_
                                d_18_appendedInside_ = out13_
                                d_19_appendedCurrent_ = out14_
                                generated = d_17_appendedGenerated_
                                insideConstrainedOut = d_18_appendedInside_
                                currentConstrainedOut = d_19_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


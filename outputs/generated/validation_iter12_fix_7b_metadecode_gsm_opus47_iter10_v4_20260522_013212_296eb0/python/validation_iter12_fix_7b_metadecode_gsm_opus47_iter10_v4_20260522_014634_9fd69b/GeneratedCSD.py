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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. Every arithmetic calculation MUST be wrapped in << >> delimiters. Example: 'She has 3 apples and gets 2 more, so <<3+2=5>> apples total. The final answer is #### <<5>>.' Always end with '#### <<NUMBER>>' where NUMBER is the final answer.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_sinceLastOpen_: int
        d_2_sinceLastOpen_ = 0
        d_3_forceOpenThreshold_: int
        d_3_forceOpenThreshold_ = 80
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_2_sinceLastOpen_) >= (d_3_forceOpenThreshold_)) and (((maxSteps) - (d_1_steps_)) >= (6)):
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
                            d_2_sinceLastOpen_ = 0
                        elif True:
                            (d_0_helpers_).SafeBoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('15e-1'))
                            d_7_chunkBudget_: int
                            d_7_chunkBudget_ = (maxSteps) - (d_1_steps_)
                            if (d_7_chunkBudget_) > (20):
                                d_7_chunkBudget_ = 20
                            d_8_prevLen_: int
                            d_8_prevLen_ = len(generated)
                            d_9_chunkedG_: _dafny.Seq
                            d_10_stoppedOpen_: bool
                            d_11_stoppedEos_: bool
                            d_12_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_9_chunkedG_ = out3_
                            d_10_stoppedOpen_ = out4_
                            d_11_stoppedEos_ = out5_
                            d_12_stepsUsed_ = out6_
                            generated = d_9_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                            d_2_sinceLastOpen_ = (d_2_sinceLastOpen_) + (d_12_stepsUsed_)
                            if d_11_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_10_stoppedOpen_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_sinceLastOpen_ = 0
                            elif (d_12_stepsUsed_) == (0):
                                raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_closedGenerated_: _dafny.Seq
                        d_14_closedInside_: bool
                        d_15_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_closedGenerated_ = out7_
                        d_14_closedInside_ = out8_
                        d_15_closedCurrent_ = out9_
                        generated = d_13_closedGenerated_
                        insideConstrainedOut = d_14_closedInside_
                        currentConstrainedOut = d_15_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_sinceLastOpen_ = 0
                    elif True:
                        d_16_stablePrefix_: _dafny.Seq
                        d_16_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (d_16_stablePrefix_)
                        d_18_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 8, eosToken)
                        d_18_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_18_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_19_appendedGenerated_: _dafny.Seq
                            d_20_appendedInside_: bool
                            d_21_appendedCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                            d_19_appendedGenerated_ = out11_
                            d_20_appendedInside_ = out12_
                            d_21_appendedCurrent_ = out13_
                            generated = d_19_appendedGenerated_
                            insideConstrainedOut = d_20_appendedInside_
                            currentConstrainedOut = d_21_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


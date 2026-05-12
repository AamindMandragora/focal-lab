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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_earlyGroupLimit_: int
        d_2_earlyGroupLimit_ = 2
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_openedGenerated_: _dafny.Seq
                        d_4_openedInside_: bool
                        d_5_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_3_openedGenerated_ = out0_
                        d_4_openedInside_ = out1_
                        d_5_openedCurrent_ = out2_
                        generated = d_3_openedGenerated_
                        insideConstrainedOut = d_4_openedInside_
                        currentConstrainedOut = d_5_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_6_closedGenerated_: _dafny.Seq
                        d_7_closedInside_: bool
                        d_8_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_6_closedGenerated_ = out3_
                        d_7_closedInside_ = out4_
                        d_8_closedCurrent_ = out5_
                        generated = d_6_closedGenerated_
                        insideConstrainedOut = d_7_closedInside_
                        currentConstrainedOut = d_8_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_9_stablePrefix_: _dafny.Seq
                        d_9_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (d_9_stablePrefix_)
                        d_11_next_: _dafny.Seq
                        d_11_next_ = eosToken
                        if (len(currentConstrainedOut)) < (d_2_earlyGroupLimit_):
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_11_next_ = out6_
                        elif True:
                            d_12_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_12_validCount_ = out7_
                            if (d_12_validCount_) <= (8):
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_11_next_ = out8_
                            elif True:
                                d_13_gatedNext_: _dafny.Seq
                                d_14_wasConstrained_: bool
                                out9_: _dafny.Seq
                                out10_: bool
                                out9_, out10_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_13_gatedNext_ = out9_
                                d_14_wasConstrained_ = out10_
                                d_11_next_ = d_13_gatedNext_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_11_next_) == (eosToken):
                            d_15_repairedGenerated_: _dafny.Seq
                            d_16_repairedCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out11_, out12_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_15_repairedGenerated_ = out11_
                            d_16_repairedCurrent_ = out12_
                            generated = d_15_repairedGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_16_repairedCurrent_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_17_closedGenerated2_: _dafny.Seq
                                d_18_closedInside2_: bool
                                d_19_closedCurrent2_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_17_closedGenerated2_ = out13_
                                d_18_closedInside2_ = out14_
                                d_19_closedCurrent2_ = out15_
                                generated = d_17_closedGenerated2_
                                insideConstrainedOut = d_18_closedInside2_
                                currentConstrainedOut = d_19_closedCurrent2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_20_appendedGenerated_: _dafny.Seq
                            d_21_appendedInside_: bool
                            d_22_appendedCurrent_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                            d_20_appendedGenerated_ = out16_
                            d_21_appendedInside_ = out17_
                            d_22_appendedCurrent_ = out18_
                            generated = d_20_appendedGenerated_
                            insideConstrainedOut = d_21_appendedInside_
                            currentConstrainedOut = d_22_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
        d_2_rollbackLimit_: int
        d_2_rollbackLimit_ = 64
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 6
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
                    elif (len(currentConstrainedOut)) >= (d_2_rollbackLimit_):
                        d_10_rolledGenerated_: _dafny.Seq
                        d_11_rolledCurrent_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: _dafny.Seq
                        out6_, out7_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_10_rolledGenerated_ = out6_
                        d_11_rolledCurrent_ = out7_
                        generated = d_10_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_11_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_12_stablePrefix_: _dafny.Seq
                        d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                        d_14_validCount_: int
                        out8_: int
                        out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_14_validCount_ = out8_
                        if (d_14_validCount_) <= (d_3_narrowThreshold_):
                            d_15_nextNarrow_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_15_nextNarrow_ = out9_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_15_nextNarrow_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_16_appendedGenerated1_: _dafny.Seq
                                d_17_appendedInside1_: bool
                                d_18_appendedCurrent1_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_nextNarrow_)
                                d_16_appendedGenerated1_ = out10_
                                d_17_appendedInside1_ = out11_
                                d_18_appendedCurrent1_ = out12_
                                generated = d_16_appendedGenerated1_
                                insideConstrainedOut = d_17_appendedInside1_
                                currentConstrainedOut = d_18_appendedCurrent1_
                        elif True:
                            d_19_nextGuided_: _dafny.Seq
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_19_nextGuided_ = out13_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_19_nextGuided_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_20_appendedGenerated2_: _dafny.Seq
                                d_21_appendedInside2_: bool
                                d_22_appendedCurrent2_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_nextGuided_)
                                d_20_appendedGenerated2_ = out14_
                                d_21_appendedInside2_ = out15_
                                d_22_appendedCurrent2_ = out16_
                                generated = d_20_appendedGenerated2_
                                insideConstrainedOut = d_21_appendedInside2_
                                currentConstrainedOut = d_22_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


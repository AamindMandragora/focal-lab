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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openedSpan_: bool
        d_2_openedSpan_ = insideConstrained
        d_3_preambleDone_: bool
        d_3_preambleDone_ = (insideConstrained) or ((len(generatedPrefix)) > (0))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_4_completeNow_: bool
                        d_4_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_4_completeNow_:
                            d_5_closedGenerated0_: _dafny.Seq
                            d_6_closedInside0_: bool
                            d_7_closedCurrent0_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_5_closedGenerated0_ = out0_
                            d_6_closedInside0_ = out1_
                            d_7_closedCurrent0_ = out2_
                            generated = d_5_closedGenerated0_
                            insideConstrainedOut = d_6_closedInside0_
                            currentConstrainedOut = d_7_closedCurrent0_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_openedSpan_ = True
                            d_3_preambleDone_ = True
                        elif True:
                            d_8_remaining0_: int
                            d_8_remaining0_ = (maxSteps) - (d_1_steps_)
                            if (d_8_remaining0_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_9_stablePrefix0_: _dafny.Seq
                                d_9_stablePrefix0_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_10_constrainedPrompt0_: _dafny.Seq
                                d_10_constrainedPrompt0_ = (prompt) + (d_9_stablePrefix0_)
                                d_11_currentOut0_: _dafny.Seq
                                d_12_hitEos0_: bool
                                d_13_stepsUsed0_: int
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: int
                                out3_, out4_, out5_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_10_constrainedPrompt0_, currentConstrainedOut, 1, eosToken)
                                d_11_currentOut0_ = out3_
                                d_12_hitEos0_ = out4_
                                d_13_stepsUsed0_ = out5_
                                currentConstrainedOut = d_11_currentOut0_
                                generated = (d_9_stablePrefix0_) + (currentConstrainedOut)
                                insideConstrainedOut = True
                                d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed0_)
                                if d_12_hitEos0_:
                                    raise _dafny.Break("0")
                                elif True:
                                    d_14_completeAfterGrow0_: bool
                                    d_14_completeAfterGrow0_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if d_14_completeAfterGrow0_:
                                        if (d_1_steps_) < (maxSteps):
                                            d_15_closedGenerated1_: _dafny.Seq
                                            d_16_closedInside1_: bool
                                            d_17_closedCurrent1_: _dafny.Seq
                                            out6_: _dafny.Seq
                                            out7_: bool
                                            out8_: _dafny.Seq
                                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_15_closedGenerated1_ = out6_
                                            d_16_closedInside1_ = out7_
                                            d_17_closedCurrent1_ = out8_
                                            generated = d_15_closedGenerated1_
                                            insideConstrainedOut = d_16_closedInside1_
                                            currentConstrainedOut = d_17_closedCurrent1_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                            d_2_openedSpan_ = True
                                            d_3_preambleDone_ = True
                                        elif True:
                                            raise _dafny.Break("0")
                                    elif True:
                                        d_18_stablePrefix1_: _dafny.Seq
                                        d_18_stablePrefix1_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                        d_19_repairedGenerated_: _dafny.Seq
                                        d_20_repairedCurrent_: _dafny.Seq
                                        out9_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out9_, out10_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_18_stablePrefix1_, generated, currentConstrainedOut)
                                        d_19_repairedGenerated_ = out9_
                                        d_20_repairedCurrent_ = out10_
                                        generated = d_19_repairedGenerated_
                                        currentConstrainedOut = d_20_repairedCurrent_
                                        insideConstrainedOut = True
                                        d_21_completeAfterRepair_: bool
                                        d_21_completeAfterRepair_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                        if d_21_completeAfterRepair_:
                                            if (d_1_steps_) < (maxSteps):
                                                d_22_closedGenerated2_: _dafny.Seq
                                                d_23_closedInside2_: bool
                                                d_24_closedCurrent2_: _dafny.Seq
                                                out11_: _dafny.Seq
                                                out12_: bool
                                                out13_: _dafny.Seq
                                                out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                d_22_closedGenerated2_ = out11_
                                                d_23_closedInside2_ = out12_
                                                d_24_closedCurrent2_ = out13_
                                                generated = d_22_closedGenerated2_
                                                insideConstrainedOut = d_23_closedInside2_
                                                currentConstrainedOut = d_24_closedCurrent2_
                                                d_1_steps_ = (d_1_steps_) + (1)
                                                d_2_openedSpan_ = True
                                                d_3_preambleDone_ = True
                                            elif True:
                                                raise _dafny.Break("0")
                                        elif True:
                                            raise _dafny.Break("0")
                    elif True:
                        if not(d_3_preambleDone_):
                            d_25_remaining1_: int
                            d_25_remaining1_ = (maxSteps) - (d_1_steps_)
                            d_26_chunkBudget_: int
                            d_26_chunkBudget_ = 3
                            if (d_25_remaining1_) < (d_26_chunkBudget_):
                                d_26_chunkBudget_ = d_25_remaining1_
                            if (d_26_chunkBudget_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_27_chunkGenerated_: _dafny.Seq
                                d_28_stoppedOnOpenSpan_: bool
                                d_29_stoppedOnEos_: bool
                                d_30_stepsUsed1_: int
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: bool
                                out17_: int
                                out14_, out15_, out16_, out17_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_26_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_27_chunkGenerated_ = out14_
                                d_28_stoppedOnOpenSpan_ = out15_
                                d_29_stoppedOnEos_ = out16_
                                d_30_stepsUsed1_ = out17_
                                generated = d_27_chunkGenerated_
                                d_1_steps_ = (d_1_steps_) + (d_30_stepsUsed1_)
                                d_3_preambleDone_ = True
                                if d_29_stoppedOnEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    if d_28_stoppedOnOpenSpan_:
                                        raise _dafny.Break("0")
                        elif True:
                            if not(d_2_openedSpan_):
                                if (d_1_steps_) < (maxSteps):
                                    d_31_openedGenerated0_: _dafny.Seq
                                    d_32_openedInside0_: bool
                                    d_33_openedCurrent0_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_31_openedGenerated0_ = out18_
                                    d_32_openedInside0_ = out19_
                                    d_33_openedCurrent0_ = out20_
                                    generated = d_31_openedGenerated0_
                                    insideConstrainedOut = d_32_openedInside0_
                                    currentConstrainedOut = d_33_openedCurrent0_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_2_openedSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_34_remaining2_: int
                                d_34_remaining2_ = (maxSteps) - (d_1_steps_)
                                if (d_34_remaining2_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_35_chunkGenerated2_: _dafny.Seq
                                    d_36_stoppedOnOpenSpan2_: bool
                                    d_37_stoppedOnEos2_: bool
                                    d_38_stepsUsed2_: int
                                    out21_: _dafny.Seq
                                    out22_: bool
                                    out23_: bool
                                    out24_: int
                                    out21_, out22_, out23_, out24_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_34_remaining2_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                    d_35_chunkGenerated2_ = out21_
                                    d_36_stoppedOnOpenSpan2_ = out22_
                                    d_37_stoppedOnEos2_ = out23_
                                    d_38_stepsUsed2_ = out24_
                                    generated = d_35_chunkGenerated2_
                                    d_1_steps_ = (d_1_steps_) + (d_38_stepsUsed2_)
                                    if d_37_stoppedOnEos2_:
                                        raise _dafny.Break("0")
                                    elif True:
                                        if d_36_stoppedOnOpenSpan2_:
                                            raise _dafny.Break("0")
                                        elif True:
                                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


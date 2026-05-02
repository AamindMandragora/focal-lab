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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, eosToken):
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
        d_2_narrowCount_: int
        d_2_narrowCount_ = 4
        d_3_topK_: int
        d_3_topK_ = 6
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_chunkBudget_: int
                        d_4_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_5_chunkedGenerated_: _dafny.Seq
                        d_6_stoppedOnOpenSpan_: bool
                        d_7_stoppedOnEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_chunkedGenerated_ = out0_
                        d_6_stoppedOnOpenSpan_ = out1_
                        d_7_stoppedOnEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        generated = d_5_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        if d_7_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_6_stoppedOnOpenSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_9_completeNow_: bool
                        d_9_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_9_completeNow_:
                            d_10_closedGenerated_: _dafny.Seq
                            d_11_closedInside_: bool
                            d_12_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_closedGenerated_ = out4_
                            d_11_closedInside_ = out5_
                            d_12_closedCurrent_ = out6_
                            generated = d_10_closedGenerated_
                            insideConstrainedOut = d_11_closedInside_
                            currentConstrainedOut = d_12_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_13_deadEnd_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_13_deadEnd_ = out7_
                            if d_13_deadEnd_:
                                d_14_stablePrefixDead_: _dafny.Seq
                                d_14_stablePrefixDead_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_15_rolledGenerated_: _dafny.Seq
                                d_16_rolledCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: _dafny.Seq
                                out8_, out9_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_14_stablePrefixDead_, generated, currentConstrainedOut)
                                d_15_rolledGenerated_ = out8_
                                d_16_rolledCurrent_ = out9_
                                generated = d_15_rolledGenerated_
                                currentConstrainedOut = d_16_rolledCurrent_
                                d_17_completeAfterRollback_: bool
                                d_17_completeAfterRollback_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_17_completeAfterRollback_:
                                    if (d_1_steps_) < (maxSteps):
                                        d_18_closedGenerated2_: _dafny.Seq
                                        d_19_closedInside2_: bool
                                        d_20_closedCurrent2_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_18_closedGenerated2_ = out10_
                                        d_19_closedInside2_ = out11_
                                        d_20_closedCurrent2_ = out12_
                                        generated = d_18_closedGenerated2_
                                        insideConstrainedOut = d_19_closedInside2_
                                        currentConstrainedOut = d_20_closedCurrent2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_21_stablePrefix_: _dafny.Seq
                                d_21_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_22_constrainedPrompt_: _dafny.Seq
                                d_22_constrainedPrompt_ = (prompt) + (d_21_stablePrefix_)
                                d_23_validCount_: int
                                out13_: int
                                out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_23_validCount_ = out13_
                                if (d_23_validCount_) <= (d_2_narrowCount_):
                                    d_24_nextTight_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_24_nextTight_ = out14_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_24_nextTight_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_25_appendedGenerated1_: _dafny.Seq
                                        d_26_appendedInside1_: bool
                                        d_27_appendedCurrent1_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out17_: _dafny.Seq
                                        out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_nextTight_)
                                        d_25_appendedGenerated1_ = out15_
                                        d_26_appendedInside1_ = out16_
                                        d_27_appendedCurrent1_ = out17_
                                        generated = d_25_appendedGenerated1_
                                        insideConstrainedOut = d_26_appendedInside1_
                                        currentConstrainedOut = d_27_appendedCurrent1_
                                elif True:
                                    d_28_candidates_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out18_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, d_3_topK_, eosToken)
                                    d_28_candidates_ = out18_
                                    (lm).GenerateLogits((d_22_constrainedPrompt_) + (currentConstrainedOut))
                                    (d_0_helpers_).BoostTokenLogits(lm, d_28_candidates_, _dafny.BigRational('8e0'))
                                    d_29_nextBroad_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out19_ = (lm).ChooseNextToken()
                                    d_29_nextBroad_ = out19_
                                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_29_nextBroad_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_30_validBroad_: bool
                                        out20_: bool
                                        out20_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_29_nextBroad_)
                                        d_30_validBroad_ = out20_
                                        if d_30_validBroad_:
                                            d_31_appendedGenerated2_: _dafny.Seq
                                            d_32_appendedInside2_: bool
                                            d_33_appendedCurrent2_: _dafny.Seq
                                            out21_: _dafny.Seq
                                            out22_: bool
                                            out23_: _dafny.Seq
                                            out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_nextBroad_)
                                            d_31_appendedGenerated2_ = out21_
                                            d_32_appendedInside2_ = out22_
                                            d_33_appendedCurrent2_ = out23_
                                            generated = d_31_appendedGenerated2_
                                            insideConstrainedOut = d_32_appendedInside2_
                                            currentConstrainedOut = d_33_appendedCurrent2_
                                        elif True:
                                            d_34_nextFallback_: _dafny.Seq
                                            out24_: _dafny.Seq
                                            out24_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                                            d_34_nextFallback_ = out24_
                                            if (d_34_nextFallback_) == (eosToken):
                                                raise _dafny.Break("0")
                                            elif True:
                                                d_35_appendedGenerated3_: _dafny.Seq
                                                d_36_appendedInside3_: bool
                                                d_37_appendedCurrent3_: _dafny.Seq
                                                out25_: _dafny.Seq
                                                out26_: bool
                                                out27_: _dafny.Seq
                                                out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_34_nextFallback_)
                                                d_35_appendedGenerated3_ = out25_
                                                d_36_appendedInside3_ = out26_
                                                d_37_appendedCurrent3_ = out27_
                                                generated = d_35_appendedGenerated3_
                                                insideConstrainedOut = d_36_appendedInside3_
                                                currentConstrainedOut = d_37_appendedCurrent3_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


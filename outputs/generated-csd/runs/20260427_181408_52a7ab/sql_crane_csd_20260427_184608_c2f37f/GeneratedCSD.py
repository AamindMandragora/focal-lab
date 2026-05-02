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
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_chunkBudget_: int
                        d_3_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_4_chunkedGenerated_: _dafny.Seq
                        d_5_stoppedOnOpenSpan_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_chunkedGenerated_ = out0_
                        d_5_stoppedOnOpenSpan_ = out1_
                        d_6_stoppedOnEos_ = out2_
                        d_7_stepsUsed_ = out3_
                        generated = d_4_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        if d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_5_stoppedOnOpenSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_8_completeNow_: bool
                        d_8_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_completeNow_:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                            d_9_closeTop_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_9_closeTop_ = out4_
                            if (d_9_closeTop_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                d_10_closedGenerated_: _dafny.Seq
                                d_11_closedInside_: bool
                                d_12_closedCurrent_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_10_closedGenerated_ = out5_
                                d_11_closedInside_ = out6_
                                d_12_closedCurrent_ = out7_
                                generated = d_10_closedGenerated_
                                insideConstrainedOut = d_11_closedInside_
                                currentConstrainedOut = d_12_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_13_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (lm).ChooseNextToken()
                                d_13_next_ = out8_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_13_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_14_stillNotComplete1_: bool
                                    d_14_stillNotComplete1_ = not((parser).IsCompletePrefix(currentConstrainedOut))
                                    d_15_validNext1_: bool
                                    d_15_validNext1_ = (parser).IsValidPrefix((currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([d_13_next_])))
                                    if (d_14_stillNotComplete1_) and (d_15_validNext1_):
                                        d_16_appendedGenerated1_: _dafny.Seq
                                        d_17_appendedInside1_: bool
                                        d_18_appendedCurrent1_: _dafny.Seq
                                        out9_: _dafny.Seq
                                        out10_: bool
                                        out11_: _dafny.Seq
                                        out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                        d_16_appendedGenerated1_ = out9_
                                        d_17_appendedInside1_ = out10_
                                        d_18_appendedCurrent1_ = out11_
                                        generated = d_16_appendedGenerated1_
                                        insideConstrainedOut = d_17_appendedInside1_
                                        currentConstrainedOut = d_18_appendedCurrent1_
                                    elif True:
                                        raise _dafny.Break("0")
                        elif True:
                            d_19_constrainedPrompt_: _dafny.Seq
                            d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            (lm).GenerateLogits((d_19_constrainedPrompt_) + (currentConstrainedOut))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_20_validCount_: int
                            out12_: int
                            out12_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_20_validCount_ = out12_
                            d_21_deadEndish_: bool
                            out13_: bool
                            out13_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 3)
                            d_21_deadEndish_ = out13_
                            if d_21_deadEndish_:
                                d_22_top3_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, 3, eosToken)
                                d_22_top3_ = out14_
                                (d_0_helpers_).BoostTokenLogits(lm, d_22_top3_, _dafny.BigRational('8e0'))
                            elif (d_20_validCount_) <= (d_2_narrowThreshold_):
                                d_23_top5_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, 5, eosToken)
                                d_23_top5_ = out15_
                                (d_0_helpers_).BoostTokenLogits(lm, d_23_top5_, _dafny.BigRational('3e0'))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                            d_24_next2_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (lm).ChooseNextToken()
                            d_24_next2_ = out16_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_24_next2_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_25_stillNotComplete2_: bool
                                d_25_stillNotComplete2_ = not((parser).IsCompletePrefix(currentConstrainedOut))
                                d_26_validNext2_: bool
                                d_26_validNext2_ = (parser).IsValidPrefix((currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([d_24_next2_])))
                                if (d_25_stillNotComplete2_) and (d_26_validNext2_):
                                    d_27_appendedGenerated2_: _dafny.Seq
                                    d_28_appendedInside2_: bool
                                    d_29_appendedCurrent2_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next2_)
                                    d_27_appendedGenerated2_ = out17_
                                    d_28_appendedInside2_ = out18_
                                    d_29_appendedCurrent2_ = out19_
                                    generated = d_27_appendedGenerated2_
                                    insideConstrainedOut = d_28_appendedInside2_
                                    currentConstrainedOut = d_29_appendedCurrent2_
                                elif True:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


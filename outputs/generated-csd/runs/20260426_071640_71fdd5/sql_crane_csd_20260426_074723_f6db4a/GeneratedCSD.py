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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, eosToken):
        generated: _dafny.Seq = _dafny.Seq({})
        insideConstrainedOut: bool = False
        currentConstrainedOut: _dafny.Seq = _dafny.Seq({})
        cost: int = int(0)
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        if True:
            generated = generatedPrefix
            insideConstrainedOut = insideConstrained
            currentConstrainedOut = currentConstrained
            cost = 0
            d_1_steps_: int
            d_1_steps_ = 0
            with _dafny.label("0"):
                while (d_1_steps_) < (maxSteps):
                    with _dafny.c_label("0"):
                        if not(insideConstrainedOut):
                            d_2_openedGenerated_: _dafny.Seq
                            d_3_openedInside_: bool
                            d_4_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_2_openedGenerated_ = out0_
                            d_3_openedInside_ = out1_
                            d_4_openedCurrent_ = out2_
                            generated = d_2_openedGenerated_
                            insideConstrainedOut = d_3_openedInside_
                            currentConstrainedOut = d_4_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_5_remaining_: int
                            d_5_remaining_ = (maxSteps) - (d_1_steps_)
                            d_6_complete_: bool
                            d_6_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if ((d_5_remaining_) == (1)) and (d_6_complete_):
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
                                d_10_validCount_: int
                                out6_: int
                                out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_10_validCount_ = out6_
                                d_11_narrow_: bool
                                out7_: bool
                                out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 4)
                                d_11_narrow_ = out7_
                                d_12_constrainedPrompt_: _dafny.Seq
                                d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                (lm).GenerateLogits((d_12_constrainedPrompt_) + (currentConstrainedOut))
                                if (d_10_validCount_) <= (6):
                                    (d_0_helpers_).ScaleAllLogits(lm, _dafny.BigRational('5e-1'))
                                    d_13_cands_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out8_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 12, eosToken)
                                    d_13_cands_ = out8_
                                    (d_0_helpers_).BoostTokenLogits(lm, d_13_cands_, _dafny.BigRational('12e0'))
                                elif True:
                                    d_14_cands2_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out9_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 8, eosToken)
                                    d_14_cands2_ = out9_
                                    (d_0_helpers_).BoostTokenLogits(lm, d_14_cands2_, _dafny.BigRational('5e0'))
                                if d_11_narrow_:
                                    (d_0_helpers_).ScaleAllLogits(lm, _dafny.BigRational('4e-1'))
                                    d_15_cands3_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 15, eosToken)
                                    d_15_cands3_ = out10_
                                    (d_0_helpers_).BoostTokenLogits(lm, d_15_cands3_, _dafny.BigRational('25e0'))
                                d_16_next_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_16_next_ = out11_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_16_next_) == (eosToken):
                                    if (d_1_steps_) < (maxSteps):
                                        d_17_completeAfterEos_: bool
                                        d_17_completeAfterEos_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                        if d_17_completeAfterEos_:
                                            d_18_closedGenerated2_: _dafny.Seq
                                            d_19_closedInside2_: bool
                                            d_20_closedCurrent2_: _dafny.Seq
                                            out12_: _dafny.Seq
                                            out13_: bool
                                            out14_: _dafny.Seq
                                            out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_18_closedGenerated2_ = out12_
                                            d_19_closedInside2_ = out13_
                                            d_20_closedCurrent2_ = out14_
                                            generated = d_18_closedGenerated2_
                                            insideConstrainedOut = d_19_closedInside2_
                                            currentConstrainedOut = d_20_closedCurrent2_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                                elif True:
                                    d_21_completeNow_: bool
                                    d_21_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if d_21_completeNow_:
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_22_appendedGenerated_: _dafny.Seq
                                        d_23_appendedInside_: bool
                                        d_24_appendedCurrent_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out17_: _dafny.Seq
                                        out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                        d_22_appendedGenerated_ = out15_
                                        d_23_appendedInside_ = out16_
                                        d_24_appendedCurrent_ = out17_
                                        generated = d_22_appendedGenerated_
                                        insideConstrainedOut = d_23_appendedInside_
                                        currentConstrainedOut = d_24_appendedCurrent_
                        pass
                pass
            if ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                d_25_completeAtEnd_: bool
                d_25_completeAtEnd_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_25_completeAtEnd_:
                    d_26_closedGenerated3_: _dafny.Seq
                    d_27_closedInside3_: bool
                    d_28_closedCurrent3_: _dafny.Seq
                    out18_: _dafny.Seq
                    out19_: bool
                    out20_: _dafny.Seq
                    out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_26_closedGenerated3_ = out18_
                    d_27_closedInside3_ = out19_
                    d_28_closedCurrent3_ = out20_
                    generated = d_26_closedGenerated3_
                    insideConstrainedOut = d_27_closedInside3_
                    currentConstrainedOut = d_28_closedCurrent3_
                    d_1_steps_ = (d_1_steps_) + (1)
            cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


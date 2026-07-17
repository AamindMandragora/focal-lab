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
        d_2_phase1Budget_: int
        if (maxSteps) > (310):
            d_2_phase1Budget_ = 300
        elif True:
            if (maxSteps) > (10):
                d_2_phase1Budget_ = (maxSteps) - (10)
            elif True:
                d_2_phase1Budget_ = 0
        if ((d_2_phase1Budget_) > (0)) and (((d_1_steps_) + (d_2_phase1Budget_)) <= (maxSteps)):
            d_3_generatedOut_: _dafny.Seq
            d_4_stoppedOnOpenSpan_: bool
            d_5_stoppedOnEos_: bool
            d_6_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_phase1Budget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_3_generatedOut_ = out0_
            d_4_stoppedOnOpenSpan_ = out1_
            d_5_stoppedOnEos_ = out2_
            d_6_stepsUsed_ = out3_
            d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
            generated = d_3_generatedOut_
            if d_5_stoppedOnEos_:
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            elif d_4_stoppedOnOpenSpan_:
                d_7_g2_: _dafny.Seq
                d_8_ins2_: bool
                d_9_cur2_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_7_g2_ = out4_
                d_8_ins2_ = out5_
                d_9_cur2_ = out6_
                generated = d_7_g2_
                insideConstrainedOut = d_8_ins2_
                currentConstrainedOut = d_9_cur2_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_1_steps_) + (1)) <= (maxSteps):
                            d_10_g2_: _dafny.Seq
                            d_11_ins2_: bool
                            d_12_cur2_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_g2_ = out7_
                            d_11_ins2_ = out8_
                            d_12_cur2_ = out9_
                            generated = d_10_g2_
                            insideConstrainedOut = d_11_ins2_
                            currentConstrainedOut = d_12_cur2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        if ((d_1_steps_) + (1)) <= (maxSteps):
                            d_13_closedGenerated_: _dafny.Seq
                            d_14_closedInside_: bool
                            d_15_closedCurrent_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_13_closedGenerated_ = out10_
                            d_14_closedInside_ = out11_
                            d_15_closedCurrent_ = out12_
                            generated = d_13_closedGenerated_
                            insideConstrainedOut = d_14_closedInside_
                            currentConstrainedOut = d_15_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if ((d_1_steps_) + (5)) < (maxSteps):
                                d_16_freeChunk_: int
                                if ((maxSteps) - (d_1_steps_)) > (50):
                                    d_16_freeChunk_ = 50
                                elif True:
                                    d_16_freeChunk_ = ((maxSteps) - (d_1_steps_)) - (4)
                                if (d_16_freeChunk_) > (0):
                                    d_17_generatedOut2_: _dafny.Seq
                                    d_18_stoppedOnOpenSpan2_: bool
                                    d_19_stoppedOnEos2_: bool
                                    d_20_stepsUsed2_: int
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: bool
                                    out16_: int
                                    out13_, out14_, out15_, out16_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_16_freeChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                    d_17_generatedOut2_ = out13_
                                    d_18_stoppedOnOpenSpan2_ = out14_
                                    d_19_stoppedOnEos2_ = out15_
                                    d_20_stepsUsed2_ = out16_
                                    d_1_steps_ = (d_1_steps_) + (d_20_stepsUsed2_)
                                    generated = d_17_generatedOut2_
                                    if d_19_stoppedOnEos2_:
                                        raise _dafny.Break("0")
                                    elif d_18_stoppedOnOpenSpan2_:
                                        d_21_g3_: _dafny.Seq
                                        d_22_ins3_: bool
                                        d_23_cur3_: _dafny.Seq
                                        out17_: _dafny.Seq
                                        out18_: bool
                                        out19_: _dafny.Seq
                                        out17_, out18_, out19_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                        d_21_g3_ = out17_
                                        d_22_ins3_ = out18_
                                        d_23_cur3_ = out19_
                                        generated = d_21_g3_
                                        insideConstrainedOut = d_22_ins3_
                                        currentConstrainedOut = d_23_cur3_
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_24_next_: _dafny.Seq
                        out20_: _dafny.Seq
                        out20_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                        d_24_next_ = out20_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_24_next_) == (eosToken):
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                d_25_closedGenerated_: _dafny.Seq
                                d_26_closedInside_: bool
                                d_27_closedCurrent_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: bool
                                out23_: _dafny.Seq
                                out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_25_closedGenerated_ = out21_
                                d_26_closedInside_ = out22_
                                d_27_closedCurrent_ = out23_
                                generated = d_25_closedGenerated_
                                insideConstrainedOut = d_26_closedInside_
                                currentConstrainedOut = d_27_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_28_valid_: bool
                            out24_: bool
                            out24_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_24_next_)
                            d_28_valid_ = out24_
                            if d_28_valid_:
                                d_29_appendedGenerated_: _dafny.Seq
                                d_30_appendedInside_: bool
                                d_31_appendedCurrent_: _dafny.Seq
                                out25_: _dafny.Seq
                                out26_: bool
                                out27_: _dafny.Seq
                                out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                d_29_appendedGenerated_ = out25_
                                d_30_appendedInside_ = out26_
                                d_31_appendedCurrent_ = out27_
                                generated = d_29_appendedGenerated_
                                insideConstrainedOut = d_30_appendedInside_
                                currentConstrainedOut = d_31_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


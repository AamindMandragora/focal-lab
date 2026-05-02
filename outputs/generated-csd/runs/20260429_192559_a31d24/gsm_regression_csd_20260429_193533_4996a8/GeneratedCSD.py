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
        d_2_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_flatGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_afterOpen_: _dafny.Seq
                        out1_: _dafny.Seq
                        out1_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        d_3_afterOpen_ = out1_
                        d_4_hasSpan_: bool
                        d_4_hasSpan_ = (len(d_3_afterOpen_)) < (len(generated))
                        d_5_shouldOpen_: bool
                        d_5_shouldOpen_ = False
                        if not(d_4_hasSpan_):
                            if ((d_1_steps_) >= (3)) and (((d_1_steps_) + (1)) < (maxSteps)):
                                d_6_prevEq_: _dafny.Seq
                                d_7_foundEq_: bool
                                out2_: _dafny.Seq
                                out3_: bool
                                out2_, out3_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_6_prevEq_ = out2_
                                d_7_foundEq_ = out3_
                                if d_7_foundEq_:
                                    d_5_shouldOpen_ = True
                                elif True:
                                    d_8_prevIs_: _dafny.Seq
                                    d_9_foundIs_: bool
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out4_, out5_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "is")))
                                    d_8_prevIs_ = out4_
                                    d_9_foundIs_ = out5_
                                    if d_9_foundIs_:
                                        d_5_shouldOpen_ = True
                                    elif True:
                                        d_10_prevAre_: _dafny.Seq
                                        d_11_foundAre_: bool
                                        out6_: _dafny.Seq
                                        out7_: bool
                                        out6_, out7_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "are")))
                                        d_10_prevAre_ = out6_
                                        d_11_foundAre_ = out7_
                                        if d_11_foundAre_:
                                            d_5_shouldOpen_ = True
                                        elif True:
                                            d_12_prevTotal_: _dafny.Seq
                                            d_13_foundTotal_: bool
                                            out8_: _dafny.Seq
                                            out9_: bool
                                            out8_, out9_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")))
                                            d_12_prevTotal_ = out8_
                                            d_13_foundTotal_ = out9_
                                            if d_13_foundTotal_:
                                                d_5_shouldOpen_ = True
                                            elif True:
                                                if ((d_1_steps_) >= (6)) and (((d_1_steps_) + (1)) < (maxSteps)):
                                                    d_5_shouldOpen_ = True
                        if d_5_shouldOpen_:
                            d_14_openedGenerated_: _dafny.Seq
                            d_15_openedInside_: bool
                            d_16_openedCurrent_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_14_openedGenerated_ = out10_
                            d_15_openedInside_ = out11_
                            d_16_openedCurrent_ = out12_
                            generated = d_14_openedGenerated_
                            insideConstrainedOut = d_15_openedInside_
                            currentConstrainedOut = d_16_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_17_next_: _dafny.Seq
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_17_next_ = out13_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_17_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_17_next_]))
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_18_closedGenerated_: _dafny.Seq
                            d_19_closedInside_: bool
                            d_20_closedCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_18_closedGenerated_ = out14_
                            d_19_closedInside_ = out15_
                            d_20_closedCurrent_ = out16_
                            generated = d_18_closedGenerated_
                            insideConstrainedOut = d_19_closedInside_
                            currentConstrainedOut = d_20_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if (len(currentConstrainedOut)) >= (5):
                                d_21_rolled_: _dafny.Seq
                                out17_: _dafny.Seq
                                out17_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_21_rolled_ = out17_
                                if (len(d_21_rolled_)) < (len(currentConstrainedOut)):
                                    d_22_stablePrefix_: _dafny.Seq
                                    d_22_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_23_repairedGenerated_: _dafny.Seq
                                    d_24_repairedCurrent_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out18_, out19_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_22_stablePrefix_, generated, currentConstrainedOut)
                                    d_23_repairedGenerated_ = out18_
                                    d_24_repairedCurrent_ = out19_
                                    generated = d_23_repairedGenerated_
                                    currentConstrainedOut = d_24_repairedCurrent_
                                    insideConstrainedOut = True
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_25_dead_: bool
                                out20_: bool
                                out20_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                                d_25_dead_ = out20_
                                if d_25_dead_:
                                    d_26_rolled2_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out21_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                    d_26_rolled2_ = out21_
                                    d_27_stablePrefix2_: _dafny.Seq
                                    d_27_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_28_repairedGenerated2_: _dafny.Seq
                                    d_29_repairedCurrent2_: _dafny.Seq
                                    out22_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out22_, out23_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_27_stablePrefix2_, generated, currentConstrainedOut)
                                    d_28_repairedGenerated2_ = out22_
                                    d_29_repairedCurrent2_ = out23_
                                    generated = d_28_repairedGenerated2_
                                    currentConstrainedOut = d_29_repairedCurrent2_
                                    insideConstrainedOut = True
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_30_stablePrefix3_: _dafny.Seq
                                    d_30_stablePrefix3_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_31_constrainedPrompt_: _dafny.Seq
                                    d_31_constrainedPrompt_ = (prompt) + (d_30_stablePrefix3_)
                                    d_32_next2_: _dafny.Seq
                                    out24_: _dafny.Seq
                                    out24_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_31_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_32_next2_ = out24_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_32_next2_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_33_appendedGenerated_: _dafny.Seq
                                        d_34_appendedInside_: bool
                                        d_35_appendedCurrent_: _dafny.Seq
                                        out25_: _dafny.Seq
                                        out26_: bool
                                        out27_: _dafny.Seq
                                        out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_32_next2_)
                                        d_33_appendedGenerated_ = out25_
                                        d_34_appendedInside_ = out26_
                                        d_35_appendedCurrent_ = out27_
                                        generated = d_33_appendedGenerated_
                                        insideConstrainedOut = d_34_appendedInside_
                                        currentConstrainedOut = d_35_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


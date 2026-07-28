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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write each calculation as <<expression>> and end with #### <<answer>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanTokenCount_: int
        d_2_spanTokenCount_ = 0
        d_3_maxSpanTokens_: int
        d_3_maxSpanTokens_ = 30
        d_4_freeTokenCount_: int
        d_4_freeTokenCount_ = 0
        d_5_forcedSpanCount_: int
        d_5_forcedSpanCount_ = 0
        d_6_maxForcedSpans_: int
        d_6_maxForcedSpans_ = 8
        d_7_sawHash_: bool
        d_7_sawHash_ = False
        d_8_hashCount_: int
        d_8_hashCount_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_9_shouldForce_: bool
                        d_9_shouldForce_ = False
                        if ((d_7_sawHash_) and ((d_5_forcedSpanCount_) < (d_6_maxForcedSpans_))) and (((d_1_steps_) + (2)) < (maxSteps)):
                            d_9_shouldForce_ = True
                        if d_9_shouldForce_:
                            d_10_g2_: _dafny.Seq
                            d_11_ins2_: bool
                            d_12_cur2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_g2_ = out0_
                            d_11_ins2_ = out1_
                            d_12_cur2_ = out2_
                            generated = d_10_g2_
                            insideConstrainedOut = d_11_ins2_
                            currentConstrainedOut = d_12_cur2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanTokenCount_ = 0
                            d_7_sawHash_ = False
                            d_5_forcedSpanCount_ = (d_5_forcedSpanCount_) + (1)
                        elif True:
                            d_13_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_13_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_13_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                                d_14_g2_: _dafny.Seq
                                d_15_ins2_: bool
                                d_16_cur2_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_14_g2_ = out4_
                                d_15_ins2_ = out5_
                                d_16_cur2_ = out6_
                                generated = d_14_g2_
                                insideConstrainedOut = d_15_ins2_
                                currentConstrainedOut = d_16_cur2_
                                d_2_spanTokenCount_ = 0
                                d_5_forcedSpanCount_ = (d_5_forcedSpanCount_) + (1)
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                                d_4_freeTokenCount_ = (d_4_freeTokenCount_) + (1)
                                if (d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "#"))):
                                    d_8_hashCount_ = (d_8_hashCount_) + (1)
                                    if (d_8_hashCount_) >= (4):
                                        d_7_sawHash_ = True
                                        d_8_hashCount_ = 0
                                elif True:
                                    d_8_hashCount_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_17_closedGenerated_: _dafny.Seq
                        d_18_closedInside_: bool
                        d_19_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_closedGenerated_ = out7_
                        d_18_closedInside_ = out8_
                        d_19_closedCurrent_ = out9_
                        generated = d_17_closedGenerated_
                        insideConstrainedOut = d_18_closedInside_
                        currentConstrainedOut = d_19_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanTokenCount_ = 0
                        d_7_sawHash_ = False
                    elif (d_2_spanTokenCount_) >= (d_3_maxSpanTokens_):
                        d_20_rolledGenerated_: _dafny.Seq
                        d_21_rolledCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_20_rolledGenerated_ = out10_
                        d_21_rolledCurrent_ = out11_
                        generated = d_20_rolledGenerated_
                        currentConstrainedOut = d_21_rolledCurrent_
                        d_2_spanTokenCount_ = 0
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_22_closedGenerated_: _dafny.Seq
                            d_23_closedInside_: bool
                            d_24_closedCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_22_closedGenerated_ = out12_
                            d_23_closedInside_ = out13_
                            d_24_closedCurrent_ = out14_
                            generated = d_22_closedGenerated_
                            insideConstrainedOut = d_23_closedInside_
                            currentConstrainedOut = d_24_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_25_constrainedPrompt_: _dafny.Seq
                        d_25_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_26_next_: _dafny.Seq
                        d_27_wasConstrained_: bool
                        out15_: _dafny.Seq
                        out16_: bool
                        out15_, out16_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_25_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_26_next_ = out15_
                        d_27_wasConstrained_ = out16_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_26_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_28_valid_: bool
                            out17_: bool
                            out17_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_26_next_)
                            d_28_valid_ = out17_
                            if d_28_valid_:
                                d_29_appendedGenerated_: _dafny.Seq
                                d_30_appendedInside_: bool
                                d_31_appendedCurrent_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                                d_29_appendedGenerated_ = out18_
                                d_30_appendedInside_ = out19_
                                d_31_appendedCurrent_ = out20_
                                generated = d_29_appendedGenerated_
                                insideConstrainedOut = d_30_appendedInside_
                                currentConstrainedOut = d_31_appendedCurrent_
                                d_2_spanTokenCount_ = (d_2_spanTokenCount_) + (1)
                            elif True:
                                d_32_rolledGenerated_: _dafny.Seq
                                d_33_rolledCurrent_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: _dafny.Seq
                                out21_, out22_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_32_rolledGenerated_ = out21_
                                d_33_rolledCurrent_ = out22_
                                generated = d_32_rolledGenerated_
                                currentConstrainedOut = d_33_rolledCurrent_
                                if (parser).IsCompletePrefix(currentConstrainedOut):
                                    d_34_closedGenerated_: _dafny.Seq
                                    d_35_closedInside_: bool
                                    d_36_closedCurrent_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out24_: bool
                                    out25_: _dafny.Seq
                                    out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_34_closedGenerated_ = out23_
                                    d_35_closedInside_ = out24_
                                    d_36_closedCurrent_ = out25_
                                    generated = d_34_closedGenerated_
                                    insideConstrainedOut = d_35_closedInside_
                                    currentConstrainedOut = d_36_closedCurrent_
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_spanTokenCount_ = 0
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


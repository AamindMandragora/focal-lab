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
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 10
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
                    elif True:
                        d_6_complete_: bool
                        d_6_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_6_complete_:
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
                            d_10_stablePrefix_: _dafny.Seq
                            d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_11_constrainedPrompt_: _dafny.Seq
                            d_11_constrainedPrompt_ = (prompt) + (d_10_stablePrefix_)
                            d_12_validCount_: int
                            out6_: int
                            out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_12_validCount_ = out6_
                            if (len(currentConstrainedOut)) > (0):
                                d_13_lastTok_: _dafny.Seq
                                d_13_lastTok_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                                if (d_12_validCount_) <= (d_2_narrowThreshold_):
                                    d_14_next1_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out7_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), eosToken)
                                    d_14_next1_ = out7_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_14_next1_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_15_appendedGenerated1_: _dafny.Seq
                                        d_16_appendedInside1_: bool
                                        d_17_appendedCurrent1_: _dafny.Seq
                                        out8_: _dafny.Seq
                                        out9_: bool
                                        out10_: _dafny.Seq
                                        out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next1_)
                                        d_15_appendedGenerated1_ = out8_
                                        d_16_appendedInside1_ = out9_
                                        d_17_appendedCurrent1_ = out10_
                                        generated = d_15_appendedGenerated1_
                                        insideConstrainedOut = d_16_appendedInside1_
                                        currentConstrainedOut = d_17_appendedCurrent1_
                                elif True:
                                    if (d_13_lastTok_) in ((lm).Tokens):
                                        d_18_penalizeTokens_: _dafny.Seq
                                        d_18_penalizeTokens_ = _dafny.SeqWithoutIsStrInference([d_13_lastTok_])
                                        d_19_next2_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out11_ = (d_0_helpers_).PenalizedConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, d_18_penalizeTokens_, _dafny.BigRational('8e0'), eosToken)
                                        d_19_next2_ = out11_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        if (d_19_next2_) == (eosToken):
                                            raise _dafny.Break("0")
                                        elif True:
                                            d_20_appendedGenerated2_: _dafny.Seq
                                            d_21_appendedInside2_: bool
                                            d_22_appendedCurrent2_: _dafny.Seq
                                            out12_: _dafny.Seq
                                            out13_: bool
                                            out14_: _dafny.Seq
                                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next2_)
                                            d_20_appendedGenerated2_ = out12_
                                            d_21_appendedInside2_ = out13_
                                            d_22_appendedCurrent2_ = out14_
                                            generated = d_20_appendedGenerated2_
                                            insideConstrainedOut = d_21_appendedInside2_
                                            currentConstrainedOut = d_22_appendedCurrent2_
                                    elif True:
                                        d_23_steppedGenerated2_: _dafny.Seq
                                        d_24_steppedInside2_: bool
                                        d_25_steppedCurrent2_: _dafny.Seq
                                        d_26_hitEos2_: bool
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out17_: _dafny.Seq
                                        out18_: bool
                                        out15_, out16_, out17_, out18_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, generated, currentConstrainedOut, eosToken)
                                        d_23_steppedGenerated2_ = out15_
                                        d_24_steppedInside2_ = out16_
                                        d_25_steppedCurrent2_ = out17_
                                        d_26_hitEos2_ = out18_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        if d_26_hitEos2_:
                                            raise _dafny.Break("0")
                                        elif True:
                                            generated = d_23_steppedGenerated2_
                                            insideConstrainedOut = d_24_steppedInside2_
                                            currentConstrainedOut = d_25_steppedCurrent2_
                            elif True:
                                if (d_12_validCount_) <= (d_2_narrowThreshold_):
                                    d_27_next3_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out19_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), eosToken)
                                    d_27_next3_ = out19_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_27_next3_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_28_appendedGenerated3_: _dafny.Seq
                                        d_29_appendedInside3_: bool
                                        d_30_appendedCurrent3_: _dafny.Seq
                                        out20_: _dafny.Seq
                                        out21_: bool
                                        out22_: _dafny.Seq
                                        out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next3_)
                                        d_28_appendedGenerated3_ = out20_
                                        d_29_appendedInside3_ = out21_
                                        d_30_appendedCurrent3_ = out22_
                                        generated = d_28_appendedGenerated3_
                                        insideConstrainedOut = d_29_appendedInside3_
                                        currentConstrainedOut = d_30_appendedCurrent3_
                                elif True:
                                    d_31_steppedGenerated_: _dafny.Seq
                                    d_32_steppedInside_: bool
                                    d_33_steppedCurrent_: _dafny.Seq
                                    d_34_hitEos_: bool
                                    out23_: _dafny.Seq
                                    out24_: bool
                                    out25_: _dafny.Seq
                                    out26_: bool
                                    out23_, out24_, out25_, out26_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, generated, currentConstrainedOut, eosToken)
                                    d_31_steppedGenerated_ = out23_
                                    d_32_steppedInside_ = out24_
                                    d_33_steppedCurrent_ = out25_
                                    d_34_hitEos_ = out26_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if d_34_hitEos_:
                                        raise _dafny.Break("0")
                                    elif True:
                                        generated = d_31_steppedGenerated_
                                        insideConstrainedOut = d_32_steppedInside_
                                        currentConstrainedOut = d_33_steppedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


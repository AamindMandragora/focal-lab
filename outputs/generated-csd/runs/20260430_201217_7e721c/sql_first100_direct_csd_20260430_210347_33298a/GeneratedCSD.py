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
        d_2_longSpanThreshold_: int
        d_2_longSpanThreshold_ = 48
        d_3_finishWindow_: int
        d_3_finishWindow_ = 24
        d_4_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_4_flatGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_next_: _dafny.Seq
                        out1_: _dafny.Seq
                        out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out1_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_6_openedGenerated_: _dafny.Seq
                                d_7_openedInside_: bool
                                d_8_openedCurrent_: _dafny.Seq
                                out2_: _dafny.Seq
                                out3_: bool
                                out4_: _dafny.Seq
                                out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_6_openedGenerated_ = out2_
                                d_7_openedInside_ = out3_
                                d_8_openedCurrent_ = out4_
                                generated = d_6_openedGenerated_
                                insideConstrainedOut = d_7_openedInside_
                                currentConstrainedOut = d_8_openedCurrent_
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                    elif True:
                        d_9_completeNow_: bool
                        d_9_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_9_completeNow_:
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
                            d_13_stablePrefix_: _dafny.Seq
                            d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_14_constrainedPrompt_: _dafny.Seq
                            d_14_constrainedPrompt_ = (prompt) + (d_13_stablePrefix_)
                            d_15_remaining_: int
                            d_15_remaining_ = (maxSteps) - (d_1_steps_)
                            if (len(currentConstrainedOut)) >= (d_2_longSpanThreshold_):
                                (lm).GenerateLogits((d_14_constrainedPrompt_) + (currentConstrainedOut))
                                if (0) < (len(d_4_flatGroups_)):
                                    if (d_15_remaining_) <= (d_3_finishWindow_):
                                        (d_0_helpers_).PenalizeTokenLogits(lm, d_4_flatGroups_, _dafny.BigRational('12e0'))
                                    elif True:
                                        (d_0_helpers_).PenalizeTokenLogits(lm, d_4_flatGroups_, _dafny.BigRational('6e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_16_nextLong_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).ConstrainedSample(lm, parser, currentConstrainedOut, eosToken)
                                d_16_nextLong_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_16_nextLong_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_17_appendedGenerated1_: _dafny.Seq
                                    d_18_appendedInside1_: bool
                                    d_19_appendedCurrent1_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_nextLong_)
                                    d_17_appendedGenerated1_ = out9_
                                    d_18_appendedInside1_ = out10_
                                    d_19_appendedCurrent1_ = out11_
                                    generated = d_17_appendedGenerated1_
                                    insideConstrainedOut = d_18_appendedInside1_
                                    currentConstrainedOut = d_19_appendedCurrent1_
                            elif True:
                                if (d_15_remaining_) <= (d_3_finishWindow_):
                                    (lm).GenerateLogits((d_14_constrainedPrompt_) + (currentConstrainedOut))
                                    if (0) < (len(d_4_flatGroups_)):
                                        (d_0_helpers_).PenalizeTokenLogits(lm, d_4_flatGroups_, _dafny.BigRational('8e0'))
                                    (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                    d_20_nextFinish_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = (d_0_helpers_).ConstrainedSample(lm, parser, currentConstrainedOut, eosToken)
                                    d_20_nextFinish_ = out12_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_20_nextFinish_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_21_appendedGenerated2_: _dafny.Seq
                                        d_22_appendedInside2_: bool
                                        d_23_appendedCurrent2_: _dafny.Seq
                                        out13_: _dafny.Seq
                                        out14_: bool
                                        out15_: _dafny.Seq
                                        out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_nextFinish_)
                                        d_21_appendedGenerated2_ = out13_
                                        d_22_appendedInside2_ = out14_
                                        d_23_appendedCurrent2_ = out15_
                                        generated = d_21_appendedGenerated2_
                                        insideConstrainedOut = d_22_appendedInside2_
                                        currentConstrainedOut = d_23_appendedCurrent2_
                                elif True:
                                    d_24_nextAdaptive_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                    d_24_nextAdaptive_ = out16_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_24_nextAdaptive_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_25_appendedGenerated3_: _dafny.Seq
                                        d_26_appendedInside3_: bool
                                        d_27_appendedCurrent3_: _dafny.Seq
                                        out17_: _dafny.Seq
                                        out18_: bool
                                        out19_: _dafny.Seq
                                        out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_nextAdaptive_)
                                        d_25_appendedGenerated3_ = out17_
                                        d_26_appendedInside3_ = out18_
                                        d_27_appendedCurrent3_ = out19_
                                        generated = d_25_appendedGenerated3_
                                        insideConstrainedOut = d_26_appendedInside3_
                                        currentConstrainedOut = d_27_appendedCurrent3_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


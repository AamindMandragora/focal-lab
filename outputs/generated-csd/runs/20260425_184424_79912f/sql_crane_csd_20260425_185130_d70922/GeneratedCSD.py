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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openedAny_: bool
        d_2_openedAny_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_mustOpenSoon_: bool
                        d_3_mustOpenSoon_ = ((d_1_steps_) + (2)) >= (maxSteps)
                        if (not(d_2_openedAny_)) and (d_3_mustOpenSoon_):
                            d_4_openedGeneratedForced_: _dafny.Seq
                            d_5_openedInsideForced_: bool
                            d_6_openedCurrentForced_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openedGeneratedForced_ = out0_
                            d_5_openedInsideForced_ = out1_
                            d_6_openedCurrentForced_ = out2_
                            generated = d_4_openedGeneratedForced_
                            insideConstrainedOut = d_5_openedInsideForced_
                            currentConstrainedOut = d_6_openedCurrentForced_
                            d_2_openedAny_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            d_7_argmax_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_7_argmax_ = out3_
                            d_8_argmaxLogit_: _dafny.BigRational
                            out4_: _dafny.BigRational
                            out4_ = (d_0_helpers_).GetTokenLogit(lm, d_7_argmax_)
                            d_8_argmaxLogit_ = out4_
                            d_9_openLogit_: _dafny.BigRational
                            out5_: _dafny.BigRational
                            out5_ = (d_0_helpers_).GetTokenLogit(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                            d_9_openLogit_ = out5_
                            d_10_shouldOpen_: bool
                            d_10_shouldOpen_ = (not(d_2_openedAny_)) and (((d_9_openLogit_) >= ((d_8_argmaxLogit_) - (_dafny.BigRational('1e0')))) or (d_3_mustOpenSoon_))
                            if (d_10_shouldOpen_) and ((d_7_argmax_) != (eosToken)):
                                d_11_openedGenerated_: _dafny.Seq
                                d_12_openedInside_: bool
                                d_13_openedCurrent_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out6_, out7_, out8_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_11_openedGenerated_ = out6_
                                d_12_openedInside_ = out7_
                                d_13_openedCurrent_ = out8_
                                generated = d_11_openedGenerated_
                                insideConstrainedOut = d_12_openedInside_
                                currentConstrainedOut = d_13_openedCurrent_
                                d_2_openedAny_ = True
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_14_next_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_14_next_ = out9_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_14_next_) == (eosToken):
                                    if (not(d_2_openedAny_)) and ((d_1_steps_) < (maxSteps)):
                                        d_15_openedGeneratedLate_: _dafny.Seq
                                        d_16_openedInsideLate_: bool
                                        d_17_openedCurrentLate_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                        d_15_openedGeneratedLate_ = out10_
                                        d_16_openedInsideLate_ = out11_
                                        d_17_openedCurrentLate_ = out12_
                                        generated = d_15_openedGeneratedLate_
                                        insideConstrainedOut = d_16_openedInsideLate_
                                        currentConstrainedOut = d_17_openedCurrentLate_
                                        d_2_openedAny_ = True
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next_]))
                                    if (d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                        insideConstrainedOut = True
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                        d_2_openedAny_ = True
                    elif True:
                        d_18_complete_: bool
                        d_18_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_18_complete_:
                            d_19_closedGenerated_: _dafny.Seq
                            d_20_closedInside_: bool
                            d_21_closedCurrent_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_19_closedGenerated_ = out13_
                            d_20_closedInside_ = out14_
                            d_21_closedCurrent_ = out15_
                            generated = d_19_closedGenerated_
                            insideConstrainedOut = d_20_closedInside_
                            currentConstrainedOut = d_21_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_22_constrainedPrompt_: _dafny.Seq
                            d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            (lm).GenerateLogits((d_22_constrainedPrompt_) + (currentConstrainedOut))
                            d_23_argmaxIn_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_23_argmaxIn_ = out16_
                            d_24_argmaxValid_: bool
                            out17_: bool
                            out17_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_23_argmaxIn_)
                            d_24_argmaxValid_ = out17_
                            if ((d_23_argmaxIn_) != (eosToken)) and (d_24_argmaxValid_):
                                d_25_appendedGenerated_: _dafny.Seq
                                d_26_appendedInside_: bool
                                d_27_appendedCurrent_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_argmaxIn_)
                                d_25_appendedGenerated_ = out18_
                                d_26_appendedInside_ = out19_
                                d_27_appendedCurrent_ = out20_
                                generated = d_25_appendedGenerated_
                                insideConstrainedOut = d_26_appendedInside_
                                currentConstrainedOut = d_27_appendedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_28_nextIn_: _dafny.Seq
                                out21_: _dafny.Seq
                                out21_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_28_nextIn_ = out21_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_28_nextIn_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_29_appendedGenerated2_: _dafny.Seq
                                    d_30_appendedInside2_: bool
                                    d_31_appendedCurrent2_: _dafny.Seq
                                    out22_: _dafny.Seq
                                    out23_: bool
                                    out24_: _dafny.Seq
                                    out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_nextIn_)
                                    d_29_appendedGenerated2_ = out22_
                                    d_30_appendedInside2_ = out23_
                                    d_31_appendedCurrent2_ = out24_
                                    generated = d_29_appendedGenerated2_
                                    insideConstrainedOut = d_30_appendedInside2_
                                    currentConstrainedOut = d_31_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


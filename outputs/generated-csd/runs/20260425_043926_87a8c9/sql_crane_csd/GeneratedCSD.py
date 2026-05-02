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
        d_2_openedOnce_: bool
        d_2_openedOnce_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        (lm).GenerateLogits((prompt) + (generated))
                        if d_2_openedOnce_:
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                        d_3_argmax_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).GetHighestLogitToken(lm)
                        d_3_argmax_ = out0_
                        d_4_argmaxLogit_: _dafny.BigRational
                        out1_: _dafny.BigRational
                        out1_ = (d_0_helpers_).GetTokenLogit(lm, d_3_argmax_)
                        d_4_argmaxLogit_ = out1_
                        d_5_openLogit_: _dafny.BigRational
                        out2_: _dafny.BigRational
                        out2_ = (d_0_helpers_).GetTokenLogit(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        d_5_openLogit_ = out2_
                        d_6_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (lm).ChooseNextTokenUnconstrained()
                        d_6_next_ = out3_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_6_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif (not(d_2_openedOnce_)) and ((d_5_openLogit_) >= ((d_4_argmaxLogit_) - (_dafny.BigRational('2e0')))):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_2_openedOnce_ = True
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                            if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_openedOnce_ = True
                    elif True:
                        d_7_complete_: bool
                        d_7_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_complete_:
                            d_8_closedGenerated_: _dafny.Seq
                            d_9_closedInside_: bool
                            d_10_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_8_closedGenerated_ = out4_
                            d_9_closedInside_ = out5_
                            d_10_closedCurrent_ = out6_
                            generated = d_8_closedGenerated_
                            insideConstrainedOut = d_9_closedInside_
                            currentConstrainedOut = d_10_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_constrainedPrompt_: _dafny.Seq
                            d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            (lm).GenerateLogits((d_11_constrainedPrompt_) + (currentConstrainedOut))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                            d_12_argmaxIn_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_12_argmaxIn_ = out7_
                            d_13_argmaxValid_: bool
                            out8_: bool
                            out8_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_12_argmaxIn_)
                            d_13_argmaxValid_ = out8_
                            if (d_13_argmaxValid_) and ((d_12_argmaxIn_) != (eosToken)):
                                d_14_chosen_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (lm).ChooseNextToken()
                                d_14_chosen_ = out9_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_14_chosen_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_15_chosenValid_: bool
                                    out10_: bool
                                    out10_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_14_chosen_)
                                    d_15_chosenValid_ = out10_
                                    if d_15_chosenValid_:
                                        d_16_appendedGenerated_: _dafny.Seq
                                        d_17_appendedInside_: bool
                                        d_18_appendedCurrent_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out12_: bool
                                        out13_: _dafny.Seq
                                        out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_chosen_)
                                        d_16_appendedGenerated_ = out11_
                                        d_17_appendedInside_ = out12_
                                        d_18_appendedCurrent_ = out13_
                                        generated = d_16_appendedGenerated_
                                        insideConstrainedOut = d_17_appendedInside_
                                        currentConstrainedOut = d_18_appendedCurrent_
                                    elif True:
                                        d_19_fallback_: _dafny.Seq
                                        d_19_fallback_ = d_12_argmaxIn_
                                        d_20_appendedGenerated2_: _dafny.Seq
                                        d_21_appendedInside2_: bool
                                        d_22_appendedCurrent2_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out15_: bool
                                        out16_: _dafny.Seq
                                        out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_fallback_)
                                        d_20_appendedGenerated2_ = out14_
                                        d_21_appendedInside2_ = out15_
                                        d_22_appendedCurrent2_ = out16_
                                        generated = d_20_appendedGenerated2_
                                        insideConstrainedOut = d_21_appendedInside2_
                                        currentConstrainedOut = d_22_appendedCurrent2_
                            elif True:
                                d_23_nextConstrained_: _dafny.Seq
                                out17_: _dafny.Seq
                                out17_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_23_nextConstrained_ = out17_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_23_nextConstrained_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_24_appendedGenerated3_: _dafny.Seq
                                    d_25_appendedInside3_: bool
                                    d_26_appendedCurrent3_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_nextConstrained_)
                                    d_24_appendedGenerated3_ = out18_
                                    d_25_appendedInside3_ = out19_
                                    d_26_appendedCurrent3_ = out20_
                                    generated = d_24_appendedGenerated3_
                                    insideConstrainedOut = d_25_appendedInside3_
                                    currentConstrainedOut = d_26_appendedCurrent3_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


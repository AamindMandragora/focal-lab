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
        d_2_stop_: bool
        d_2_stop_ = False
        while ((d_1_steps_) < (maxSteps)) and (not(d_2_stop_)):
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
                d_6_completeNow_: bool
                d_6_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_6_completeNow_:
                    d_2_stop_ = True
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_7_candidates_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 8, eosToken)
                    d_7_candidates_ = out3_
                    d_8_chosen_: _dafny.Seq
                    d_8_chosen_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                    d_9_i_: int
                    d_9_i_ = 0
                    while (d_9_i_) < (len(d_7_candidates_)):
                        d_10_cand_: _dafny.Seq
                        d_10_cand_ = (d_7_candidates_)[d_9_i_]
                        if (d_8_chosen_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))):
                            if (d_10_cand_) == (eosToken):
                                pass
                            elif True:
                                if VerifiedDecoderAgent.default__.Contains(d_10_cand_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    pass
                                elif True:
                                    if VerifiedDecoderAgent.default__.Contains(d_10_cand_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<"))):
                                        pass
                                    elif True:
                                        d_8_chosen_ = d_10_cand_
                        d_9_i_ = (d_9_i_) + (1)
                    if (d_8_chosen_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))):
                        d_11_chosenValid_: bool
                        out4_: bool
                        out4_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_8_chosen_)
                        d_11_chosenValid_ = out4_
                        if d_11_chosenValid_:
                            d_12_appendedGenerated1_: _dafny.Seq
                            d_13_appendedInside1_: bool
                            d_14_appendedCurrent1_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_chosen_)
                            d_12_appendedGenerated1_ = out5_
                            d_13_appendedInside1_ = out6_
                            d_14_appendedCurrent1_ = out7_
                            generated = d_12_appendedGenerated1_
                            insideConstrainedOut = d_13_appendedInside1_
                            currentConstrainedOut = d_14_appendedCurrent1_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_15_next_: _dafny.Seq
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                        d_15_next_ = out8_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_15_next_) == (eosToken):
                            d_2_stop_ = True
                        elif True:
                            if VerifiedDecoderAgent.default__.Contains(d_15_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_2_stop_ = True
                            elif True:
                                if VerifiedDecoderAgent.default__.Contains(d_15_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<"))):
                                    d_2_stop_ = True
                                elif True:
                                    d_16_appendedGenerated2_: _dafny.Seq
                                    d_17_appendedInside2_: bool
                                    d_18_appendedCurrent2_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                    d_16_appendedGenerated2_ = out9_
                                    d_17_appendedInside2_ = out10_
                                    d_18_appendedCurrent2_ = out11_
                                    generated = d_16_appendedGenerated2_
                                    insideConstrainedOut = d_17_appendedInside2_
                                    currentConstrainedOut = d_18_appendedCurrent2_
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_19_completeAtEnd_: bool
            d_19_completeAtEnd_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_19_completeAtEnd_:
                d_20_closedGenerated_: _dafny.Seq
                d_21_closedInside_: bool
                d_22_closedCurrent_: _dafny.Seq
                out12_: _dafny.Seq
                out13_: bool
                out14_: _dafny.Seq
                out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_20_closedGenerated_ = out12_
                d_21_closedInside_ = out13_
                d_22_closedCurrent_ = out14_
                generated = d_20_closedGenerated_
                insideConstrainedOut = d_21_closedInside_
                currentConstrainedOut = d_22_closedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


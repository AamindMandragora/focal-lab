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
                        if not(d_2_openedOnce_):
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
                            d_2_openedOnce_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                            d_6_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (lm).ChooseNextToken()
                            d_6_next_ = out3_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
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
                            d_11_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_11_validCount_ = out7_
                            d_12_narrow_: bool
                            out8_: bool
                            out8_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 3)
                            d_12_narrow_ = out8_
                            if (d_12_narrow_) or ((d_11_validCount_) <= (8)):
                                (lm).GenerateLogits((prompt) + (generated))
                                d_13_candidates_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).TopValidCandidates(lm, parser, (prompt) + (generated), currentConstrainedOut, 6, eosToken)
                                d_13_candidates_ = out9_
                                (d_0_helpers_).BoostTokenLogits(lm, d_13_candidates_, _dafny.BigRational('8e0'))
                                (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                                d_14_next_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (lm).ChooseNextToken()
                                d_14_next_ = out10_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_14_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_15_validNext_: bool
                                    out11_: bool
                                    out11_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_14_next_)
                                    d_15_validNext_ = out11_
                                    if d_15_validNext_:
                                        d_16_appendedGenerated_: _dafny.Seq
                                        d_17_appendedInside_: bool
                                        d_18_appendedCurrent_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out13_: bool
                                        out14_: _dafny.Seq
                                        out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                        d_16_appendedGenerated_ = out12_
                                        d_17_appendedInside_ = out13_
                                        d_18_appendedCurrent_ = out14_
                                        generated = d_16_appendedGenerated_
                                        insideConstrainedOut = d_17_appendedInside_
                                        currentConstrainedOut = d_18_appendedCurrent_
                                    elif True:
                                        raise _dafny.Break("0")
                            elif True:
                                d_19_next_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (generated), currentConstrainedOut, eosToken)
                                d_19_next_ = out15_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_19_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_20_appendedGenerated2_: _dafny.Seq
                                    d_21_appendedInside2_: bool
                                    d_22_appendedCurrent2_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                    d_20_appendedGenerated2_ = out16_
                                    d_21_appendedInside2_ = out17_
                                    d_22_appendedCurrent2_ = out18_
                                    generated = d_20_appendedGenerated2_
                                    insideConstrainedOut = d_21_appendedInside2_
                                    currentConstrainedOut = d_22_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
        d_2_helperBase_: int
        d_2_helperBase_ = d_0_helpers_.cost
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        (lm).GenerateLogits((prompt) + (generated))
                        (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                        (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (lm).ChooseNextTokenUnconstrained()
                        d_3_next_ = out0_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_4_complete_: bool
                        d_4_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_4_complete_:
                            d_5_closedGenerated_: _dafny.Seq
                            d_6_closedInside_: bool
                            d_7_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_5_closedGenerated_ = out1_
                            d_6_closedInside_ = out2_
                            d_7_closedCurrent_ = out3_
                            generated = d_5_closedGenerated_
                            insideConstrainedOut = d_6_closedInside_
                            currentConstrainedOut = d_7_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_constrainedPrompt_: _dafny.Seq
                            d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_9_narrow_: bool
                            out4_: bool
                            out4_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_9_narrow_ = out4_
                            if d_9_narrow_:
                                d_10_next_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_10_next_ = out5_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_10_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_11_appendedGenerated_: _dafny.Seq
                                    d_12_appendedInside_: bool
                                    d_13_appendedCurrent_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                                    d_11_appendedGenerated_ = out6_
                                    d_12_appendedInside_ = out7_
                                    d_13_appendedCurrent_ = out8_
                                    generated = d_11_appendedGenerated_
                                    insideConstrainedOut = d_12_appendedInside_
                                    currentConstrainedOut = d_13_appendedCurrent_
                            elif True:
                                d_14_candidates_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                                d_14_candidates_ = out9_
                                (lm).GenerateLogits((d_8_constrainedPrompt_) + (currentConstrainedOut))
                                (d_0_helpers_).BoostTokenLogits(lm, d_14_candidates_, _dafny.BigRational('8e0'))
                                (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                                d_15_next_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (lm).ChooseNextTokenUnconstrained()
                                d_15_next_ = out10_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_15_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_16_valid_: bool
                                    out11_: bool
                                    out11_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_15_next_)
                                    d_16_valid_ = out11_
                                    if d_16_valid_:
                                        d_17_appendedGenerated_: _dafny.Seq
                                        d_18_appendedInside_: bool
                                        d_19_appendedCurrent_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out13_: bool
                                        out14_: _dafny.Seq
                                        out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                        d_17_appendedGenerated_ = out12_
                                        d_18_appendedInside_ = out13_
                                        d_19_appendedCurrent_ = out14_
                                        generated = d_17_appendedGenerated_
                                        insideConstrainedOut = d_18_appendedInside_
                                        currentConstrainedOut = d_19_appendedCurrent_
                                    elif True:
                                        d_20_fallback_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, eosToken)
                                        d_20_fallback_ = out15_
                                        if (d_20_fallback_) == (eosToken):
                                            raise _dafny.Break("0")
                                        elif True:
                                            d_21_appendedGenerated2_: _dafny.Seq
                                            d_22_appendedInside2_: bool
                                            d_23_appendedCurrent2_: _dafny.Seq
                                            out16_: _dafny.Seq
                                            out17_: bool
                                            out18_: _dafny.Seq
                                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_fallback_)
                                            d_21_appendedGenerated2_ = out16_
                                            d_22_appendedInside2_ = out17_
                                            d_23_appendedCurrent2_ = out18_
                                            generated = d_21_appendedGenerated2_
                                            insideConstrainedOut = d_22_appendedInside2_
                                            currentConstrainedOut = d_23_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


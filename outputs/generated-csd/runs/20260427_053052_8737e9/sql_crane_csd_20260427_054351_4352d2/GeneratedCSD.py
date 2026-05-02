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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, eosToken):
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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_2_next_ = out0_
                        if (d_2_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                            d_1_steps_ = (d_1_steps_) + (1)
                            if VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_3_isComplete_: bool
                        d_3_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_isComplete_:
                            d_4_closedGenerated_: _dafny.Seq
                            d_5_closedInside_: bool
                            d_6_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_4_closedGenerated_ = out1_
                            d_5_closedInside_ = out2_
                            d_6_closedCurrent_ = out3_
                            generated = d_4_closedGenerated_
                            insideConstrainedOut = d_5_closedInside_
                            currentConstrainedOut = d_6_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_candidates_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 12, eosToken)
                            d_7_candidates_ = out4_
                            if (len(d_7_candidates_)) == (0):
                                d_8_stablePrefix_: _dafny.Seq
                                d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_9_rolledGenerated_: _dafny.Seq
                                d_10_rolledCurrent_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: _dafny.Seq
                                out5_, out6_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_8_stablePrefix_, generated, currentConstrainedOut)
                                d_9_rolledGenerated_ = out5_
                                d_10_rolledCurrent_ = out6_
                                generated = d_9_rolledGenerated_
                                currentConstrainedOut = d_10_rolledCurrent_
                                raise _dafny.Break("0")
                            elif True:
                                d_11_stablePrefix_: _dafny.Seq
                                d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                (lm).GenerateLogits(((prompt) + (d_11_stablePrefix_)) + (currentConstrainedOut))
                                (d_0_helpers_).BoostTokenLogits(lm, d_7_candidates_, _dafny.BigRational('1e2'))
                                (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                                d_12_next_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = (lm).ChooseNextToken()
                                d_12_next_ = out7_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                if (d_12_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_13_validNext_: bool
                                    out8_: bool
                                    out8_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_12_next_)
                                    d_13_validNext_ = out8_
                                    if d_13_validNext_:
                                        d_14_appendedGenerated_: _dafny.Seq
                                        d_15_appendedInside_: bool
                                        d_16_appendedCurrent_: _dafny.Seq
                                        out9_: _dafny.Seq
                                        out10_: bool
                                        out11_: _dafny.Seq
                                        out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                        d_14_appendedGenerated_ = out9_
                                        d_15_appendedInside_ = out10_
                                        d_16_appendedCurrent_ = out11_
                                        generated = d_14_appendedGenerated_
                                        insideConstrainedOut = d_15_appendedInside_
                                        currentConstrainedOut = d_16_appendedCurrent_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        (lm).GenerateLogits(((prompt) + (d_11_stablePrefix_)) + (currentConstrainedOut))
                                        (d_0_helpers_).BoostTokenLogits(lm, d_7_candidates_, _dafny.BigRational('1e2'))
                                        (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                                        d_17_fallback_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out12_ = (d_0_helpers_).GetHighestLogitToken(lm)
                                        d_17_fallback_ = out12_
                                        d_18_fallbackValid_: bool
                                        out13_: bool
                                        out13_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_17_fallback_)
                                        d_18_fallbackValid_ = out13_
                                        if d_18_fallbackValid_:
                                            d_19_appendedGenerated2_: _dafny.Seq
                                            d_20_appendedInside2_: bool
                                            d_21_appendedCurrent2_: _dafny.Seq
                                            out14_: _dafny.Seq
                                            out15_: bool
                                            out16_: _dafny.Seq
                                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_fallback_)
                                            d_19_appendedGenerated2_ = out14_
                                            d_20_appendedInside2_ = out15_
                                            d_21_appendedCurrent2_ = out16_
                                            generated = d_19_appendedGenerated2_
                                            insideConstrainedOut = d_20_appendedInside2_
                                            currentConstrainedOut = d_21_appendedCurrent2_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                        elif True:
                                            d_22_rolledGenerated2_: _dafny.Seq
                                            d_23_rolledCurrent2_: _dafny.Seq
                                            out17_: _dafny.Seq
                                            out18_: _dafny.Seq
                                            out17_, out18_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_11_stablePrefix_, generated, currentConstrainedOut)
                                            d_22_rolledGenerated2_ = out17_
                                            d_23_rolledCurrent2_ = out18_
                                            generated = d_22_rolledGenerated2_
                                            currentConstrainedOut = d_23_rolledCurrent2_
                                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


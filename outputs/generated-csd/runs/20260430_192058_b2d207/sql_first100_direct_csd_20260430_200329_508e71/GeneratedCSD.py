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
        d_2_lateStopWindow_: int
        d_2_lateStopWindow_ = 24
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens):
                            d_3_g0_: _dafny.Seq
                            d_4_i0_: bool
                            d_5_c0_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_3_g0_ = out0_
                            d_4_i0_ = out1_
                            d_5_c0_ = out2_
                            generated = d_3_g0_
                            insideConstrainedOut = d_4_i0_
                            currentConstrainedOut = d_5_c0_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_6_completeNow_: bool
                        d_6_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if (((maxSteps) - (d_1_steps_)) <= (d_2_lateStopWindow_)) and (d_6_completeNow_):
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))) in ((lm).Tokens):
                                d_7_g1_: _dafny.Seq
                                d_8_i1_: bool
                                d_9_c1_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_7_g1_ = out3_
                                d_8_i1_ = out4_
                                d_9_c1_ = out5_
                                generated = d_7_g1_
                                insideConstrainedOut = d_8_i1_
                                currentConstrainedOut = d_9_c1_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_10_stablePrefix_: _dafny.Seq
                            d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_11_constrainedPrompt_: _dafny.Seq
                            d_11_constrainedPrompt_ = (prompt) + (d_10_stablePrefix_)
                            if d_6_completeNow_:
                                d_12_next0_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_12_next0_ = out6_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_12_next0_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_13_g2_: _dafny.Seq
                                    d_14_i2_: bool
                                    d_15_c2_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next0_)
                                    d_13_g2_ = out7_
                                    d_14_i2_ = out8_
                                    d_15_c2_ = out9_
                                    generated = d_13_g2_
                                    insideConstrainedOut = d_14_i2_
                                    currentConstrainedOut = d_15_c2_
                            elif True:
                                if ((maxSteps) - (d_1_steps_)) <= (d_2_lateStopWindow_):
                                    d_16_next1_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_16_next1_ = out10_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_16_next1_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_17_g3_: _dafny.Seq
                                        d_18_i3_: bool
                                        d_19_c3_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out12_: bool
                                        out13_: _dafny.Seq
                                        out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next1_)
                                        d_17_g3_ = out11_
                                        d_18_i3_ = out12_
                                        d_19_c3_ = out13_
                                        generated = d_17_g3_
                                        insideConstrainedOut = d_18_i3_
                                        currentConstrainedOut = d_19_c3_
                                elif True:
                                    d_20_next2_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                    d_20_next2_ = out14_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_20_next2_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_21_g4_: _dafny.Seq
                                        d_22_i4_: bool
                                        d_23_c4_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out17_: _dafny.Seq
                                        out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next2_)
                                        d_21_g4_ = out15_
                                        d_22_i4_ = out16_
                                        d_23_c4_ = out17_
                                        generated = d_21_g4_
                                        insideConstrainedOut = d_22_i4_
                                        currentConstrainedOut = d_23_c4_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


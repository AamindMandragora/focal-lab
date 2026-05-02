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
        d_2_eosBiasStart_: int
        d_2_eosBiasStart_ = 18
        d_3_longGuard_: int
        d_3_longGuard_ = 60
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens):
                            d_4_g0_: _dafny.Seq
                            d_5_i0_: bool
                            d_6_c0_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_g0_ = out0_
                            d_5_i0_ = out1_
                            d_6_c0_ = out2_
                            generated = d_4_g0_
                            insideConstrainedOut = d_5_i0_
                            currentConstrainedOut = d_6_c0_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_7_completeNow_: bool
                        d_7_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_completeNow_:
                            if (len(currentConstrainedOut)) >= (d_2_eosBiasStart_):
                                d_8_stablePrefix0_: _dafny.Seq
                                d_8_stablePrefix0_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                (lm).GenerateLogits(((prompt) + (d_8_stablePrefix0_)) + (currentConstrainedOut))
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_9_next0_: _dafny.Seq
                                out3_: _dafny.Seq
                                out3_ = (d_0_helpers_).ConstrainedSample(lm, parser, currentConstrainedOut, eosToken)
                                d_9_next0_ = out3_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_9_next0_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))) in ((lm).Tokens):
                                        d_10_g1_: _dafny.Seq
                                        d_11_i1_: bool
                                        d_12_c1_: _dafny.Seq
                                        out4_: _dafny.Seq
                                        out5_: bool
                                        out6_: _dafny.Seq
                                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_10_g1_ = out4_
                                        d_11_i1_ = out5_
                                        d_12_c1_ = out6_
                                        generated = d_10_g1_
                                        insideConstrainedOut = d_11_i1_
                                        currentConstrainedOut = d_12_c1_
                                    elif True:
                                        raise _dafny.Break("0")
                            elif True:
                                if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))) in ((lm).Tokens):
                                    d_13_g2_: _dafny.Seq
                                    d_14_i2_: bool
                                    d_15_c2_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_13_g2_ = out7_
                                    d_14_i2_ = out8_
                                    d_15_c2_ = out9_
                                    generated = d_13_g2_
                                    insideConstrainedOut = d_14_i2_
                                    currentConstrainedOut = d_15_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                        elif True:
                            d_16_stablePrefix_: _dafny.Seq
                            d_16_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_17_constrainedPrompt_: _dafny.Seq
                            d_17_constrainedPrompt_ = (prompt) + (d_16_stablePrefix_)
                            if (len(currentConstrainedOut)) >= (d_3_longGuard_):
                                (lm).GenerateLogits((d_17_constrainedPrompt_) + (currentConstrainedOut))
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_18_next1_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).ConstrainedSample(lm, parser, currentConstrainedOut, eosToken)
                                d_18_next1_ = out10_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_18_next1_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_19_g3_: _dafny.Seq
                                    d_20_i3_: bool
                                    d_21_c3_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next1_)
                                    d_19_g3_ = out11_
                                    d_20_i3_ = out12_
                                    d_21_c3_ = out13_
                                    generated = d_19_g3_
                                    insideConstrainedOut = d_20_i3_
                                    currentConstrainedOut = d_21_c3_
                            elif True:
                                d_22_next2_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_22_next2_ = out14_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_22_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_23_g4_: _dafny.Seq
                                    d_24_i4_: bool
                                    d_25_c4_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next2_)
                                    d_23_g4_ = out15_
                                    d_24_i4_ = out16_
                                    d_25_c4_ = out17_
                                    generated = d_23_g4_
                                    insideConstrainedOut = d_24_i4_
                                    currentConstrainedOut = d_25_c4_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Put every intermediate computation and the final numeric answer inside << >> delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((maxSteps) > (10)):
            d_2_reasonBudget_: int
            if ((maxSteps) - (d_1_steps_)) > (50):
                d_2_reasonBudget_ = 50
            elif True:
                d_2_reasonBudget_ = (maxSteps) - (d_1_steps_)
            if (d_2_reasonBudget_) > (0):
                d_3_g1_: _dafny.Seq
                d_4_stoppedOpen1_: bool
                d_5_stoppedEos1_: bool
                d_6_used1_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_reasonBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_3_g1_ = out0_
                d_4_stoppedOpen1_ = out1_
                d_5_stoppedEos1_ = out2_
                d_6_used1_ = out3_
                generated = d_3_g1_
                d_1_steps_ = (d_1_steps_) + (d_6_used1_)
                if d_4_stoppedOpen1_:
                    d_7_g2_: _dafny.Seq
                    d_8_i2_: bool
                    d_9_c2_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_7_g2_ = out4_
                    d_8_i2_ = out5_
                    d_9_c2_ = out6_
                    generated = d_7_g2_
                    insideConstrainedOut = d_8_i2_
                    currentConstrainedOut = d_9_c2_
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_10_g3_: _dafny.Seq
            d_11_i3_: bool
            d_12_c3_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_10_g3_ = out7_
            d_11_i3_ = out8_
            d_12_c3_ = out9_
            generated = d_10_g3_
            insideConstrainedOut = d_11_i3_
            currentConstrainedOut = d_12_c3_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_g4_: _dafny.Seq
                        d_14_i4_: bool
                        d_15_c4_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_g4_ = out10_
                        d_14_i4_ = out11_
                        d_15_c4_ = out12_
                        generated = d_13_g4_
                        insideConstrainedOut = d_14_i4_
                        currentConstrainedOut = d_15_c4_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_16_constrainedPrompt_: _dafny.Seq
                        d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_17_next_: _dafny.Seq
                        d_18_wasConstrained_: bool
                        out13_: _dafny.Seq
                        out14_: bool
                        out13_, out14_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_17_next_ = out13_
                        d_18_wasConstrained_ = out14_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_17_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_19_g5_: _dafny.Seq
                            d_20_i5_: bool
                            d_21_c5_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                            d_19_g5_ = out15_
                            d_20_i5_ = out16_
                            d_21_c5_ = out17_
                            generated = d_19_g5_
                            insideConstrainedOut = d_20_i5_
                            currentConstrainedOut = d_21_c5_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


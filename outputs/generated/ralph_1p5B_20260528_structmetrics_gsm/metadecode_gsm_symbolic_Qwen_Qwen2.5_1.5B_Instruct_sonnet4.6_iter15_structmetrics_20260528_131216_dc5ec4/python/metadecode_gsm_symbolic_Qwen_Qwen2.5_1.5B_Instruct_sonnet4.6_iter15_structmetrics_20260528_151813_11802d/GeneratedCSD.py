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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final numeric answer inside << >> delimiters, e.g. <<42>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if ((d_1_steps_) + (3)) < (maxSteps):
            d_2_chunkBudget_: int
            d_2_chunkBudget_ = (maxSteps) - (3)
            d_3_genOut_: _dafny.Seq
            d_4_stoppedOnOpen_: bool
            d_5_stoppedOnEos_: bool
            d_6_usedSteps_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_3_genOut_ = out0_
            d_4_stoppedOnOpen_ = out1_
            d_5_stoppedOnEos_ = out2_
            d_6_usedSteps_ = out3_
            generated = d_3_genOut_
            d_1_steps_ = (d_1_steps_) + (d_6_usedSteps_)
            if d_4_stoppedOnOpen_:
                d_7_g1_: _dafny.Seq
                d_8_i1_: bool
                d_9_c1_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_7_g1_ = out4_
                d_8_i1_ = out5_
                d_9_c1_ = out6_
                generated = d_7_g1_
                insideConstrainedOut = d_8_i1_
                currentConstrainedOut = d_9_c1_
        if (not(insideConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
            d_10_g2_: _dafny.Seq
            d_11_i2_: bool
            d_12_c2_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_10_g2_ = out7_
            d_11_i2_ = out8_
            d_12_c2_ = out9_
            generated = d_10_g2_
            insideConstrainedOut = d_11_i2_
            currentConstrainedOut = d_12_c2_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_13_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_13_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_13_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                            if (d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_14_g3_: _dafny.Seq
                        d_15_i3_: bool
                        d_16_c3_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_g3_ = out11_
                        d_15_i3_ = out12_
                        d_16_c3_ = out13_
                        generated = d_14_g3_
                        insideConstrainedOut = d_15_i3_
                        currentConstrainedOut = d_16_c3_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_17_next_: _dafny.Seq
                        out14_: _dafny.Seq
                        out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                        d_17_next_ = out14_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_17_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_18_g4_: _dafny.Seq
                            d_19_i4_: bool
                            d_20_c4_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                            d_18_g4_ = out15_
                            d_19_i4_ = out16_
                            d_20_c4_ = out17_
                            generated = d_18_g4_
                            insideConstrainedOut = d_19_i4_
                            currentConstrainedOut = d_20_c4_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


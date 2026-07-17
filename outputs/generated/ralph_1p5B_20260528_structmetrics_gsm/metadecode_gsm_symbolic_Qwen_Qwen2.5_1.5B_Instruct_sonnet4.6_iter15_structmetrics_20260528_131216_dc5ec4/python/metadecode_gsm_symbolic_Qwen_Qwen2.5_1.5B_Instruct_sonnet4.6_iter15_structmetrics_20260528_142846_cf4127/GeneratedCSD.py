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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_1_steps_) + (2)) < (maxSteps):
                            d_2_chunkMax_: int
                            d_2_chunkMax_ = 60
                            if (((maxSteps) - (d_1_steps_)) - (1)) < (d_2_chunkMax_):
                                d_2_chunkMax_ = ((maxSteps) - (d_1_steps_)) - (1)
                            d_3_g2_: _dafny.Seq
                            d_4_stoppedOpen_: bool
                            d_5_stoppedEos_: bool
                            d_6_used_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkMax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_3_g2_ = out0_
                            d_4_stoppedOpen_ = out1_
                            d_5_stoppedEos_ = out2_
                            d_6_used_ = out3_
                            d_1_steps_ = (d_1_steps_) + (d_6_used_)
                            generated = d_3_g2_
                            if d_4_stoppedOpen_:
                                d_7_g3_: _dafny.Seq
                                d_8_i3_: bool
                                d_9_c3_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_7_g3_ = out4_
                                d_8_i3_ = out5_
                                d_9_c3_ = out6_
                                generated = d_7_g3_
                                insideConstrainedOut = d_8_i3_
                                currentConstrainedOut = d_9_c3_
                            elif d_5_stoppedEos_:
                                if ((d_1_steps_) + (1)) < (maxSteps):
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
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                if (d_1_steps_) < (maxSteps):
                                    d_13_g3_: _dafny.Seq
                                    d_14_i3_: bool
                                    d_15_c3_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_13_g3_ = out10_
                                    d_14_i3_ = out11_
                                    d_15_c3_ = out12_
                                    generated = d_13_g3_
                                    insideConstrainedOut = d_14_i3_
                                    currentConstrainedOut = d_15_c3_
                                    d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if ((d_1_steps_) + (1)) <= (maxSteps):
                                d_16_g3_: _dafny.Seq
                                d_17_i3_: bool
                                d_18_c3_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_16_g3_ = out13_
                                d_17_i3_ = out14_
                                d_18_c3_ = out15_
                                generated = d_16_g3_
                                insideConstrainedOut = d_17_i3_
                                currentConstrainedOut = d_18_c3_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_19_g2_: _dafny.Seq
                        d_20_i2_: bool
                        d_21_c2_: _dafny.Seq
                        out16_: _dafny.Seq
                        out17_: bool
                        out18_: _dafny.Seq
                        out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_19_g2_ = out16_
                        d_20_i2_ = out17_
                        d_21_c2_ = out18_
                        generated = d_19_g2_
                        insideConstrainedOut = d_20_i2_
                        currentConstrainedOut = d_21_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_22_next_: _dafny.Seq
                        out19_: _dafny.Seq
                        out19_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                        d_22_next_ = out19_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_22_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_23_g2_: _dafny.Seq
                            d_24_i2_: bool
                            d_25_c2_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                            d_23_g2_ = out20_
                            d_24_i2_ = out21_
                            d_25_c2_ = out22_
                            generated = d_23_g2_
                            insideConstrainedOut = d_24_i2_
                            currentConstrainedOut = d_25_c2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Place every arithmetic operation and the final answer inside << >> delimiters, e.g. <<3 * 4 = 12>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_completedSpans_: int
        d_2_completedSpans_ = 0
        d_3_freeInPhase_: int
        d_3_freeInPhase_ = 0
        d_4_maxFreePerPhase_: int
        d_4_maxFreePerPhase_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (((d_2_completedSpans_) == (0)) or ((d_3_freeInPhase_) >= (d_4_maxFreePerPhase_))) and (((d_1_steps_) + (2)) <= (maxSteps)):
                            d_5_g2_: _dafny.Seq
                            d_6_i2_: bool
                            d_7_c2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_g2_ = out0_
                            d_6_i2_ = out1_
                            d_7_c2_ = out2_
                            generated = d_5_g2_
                            insideConstrainedOut = d_6_i2_
                            currentConstrainedOut = d_7_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_freeInPhase_ = 0
                        elif ((d_1_steps_) + (1)) <= (maxSteps):
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                if ((d_2_completedSpans_) == (0)) and (((d_1_steps_) + (2)) <= (maxSteps)):
                                    d_9_g2_: _dafny.Seq
                                    d_10_i2_: bool
                                    d_11_c2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_9_g2_ = out4_
                                    d_10_i2_ = out5_
                                    d_11_c2_ = out6_
                                    generated = d_9_g2_
                                    insideConstrainedOut = d_10_i2_
                                    currentConstrainedOut = d_11_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_freeInPhase_ = 0
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                d_3_freeInPhase_ = (d_3_freeInPhase_) + (1)
                                if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_3_freeInPhase_ = 0
                        elif True:
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_g2_: _dafny.Seq
                        d_13_i2_: bool
                        d_14_c2_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_g2_ = out7_
                        d_13_i2_ = out8_
                        d_14_c2_ = out9_
                        generated = d_12_g2_
                        insideConstrainedOut = d_13_i2_
                        currentConstrainedOut = d_14_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_completedSpans_ = (d_2_completedSpans_) + (1)
                    elif True:
                        d_15_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                        d_15_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_15_next_) == (eosToken):
                            if (d_2_completedSpans_) == (0):
                                pass
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_16_g2_: _dafny.Seq
                            d_17_i2_: bool
                            d_18_c2_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                            d_16_g2_ = out11_
                            d_17_i2_ = out12_
                            d_18_c2_ = out13_
                            generated = d_16_g2_
                            insideConstrainedOut = d_17_i2_
                            currentConstrainedOut = d_18_c2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


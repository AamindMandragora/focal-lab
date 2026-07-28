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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Show all your reasoning. At the end, write the final numeric answer inside << >>. For example: <<42>>. Use exactly one << >> span containing only the number."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_reservedSteps_: int
        d_3_reservedSteps_ = 20
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_2_steps_) + (d_3_reservedSteps_)) >= (maxSteps):
                            d_4_ng_: _dafny.Seq
                            d_5_ni_: bool
                            d_6_nc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_ng_ = out0_
                            d_5_ni_ = out1_
                            d_6_nc_ = out2_
                            generated = d_4_ng_
                            insideConstrainedOut = d_5_ni_
                            currentConstrainedOut = d_6_nc_
                            d_2_steps_ = (d_2_steps_) + (1)
                        elif True:
                            d_7_chunkBudget_: int
                            d_7_chunkBudget_ = ((maxSteps) - (d_2_steps_)) - (d_3_reservedSteps_)
                            if (d_7_chunkBudget_) == (0):
                                d_8_ng_: _dafny.Seq
                                d_9_ni_: bool
                                d_10_nc_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_8_ng_ = out3_
                                d_9_ni_ = out4_
                                d_10_nc_ = out5_
                                generated = d_8_ng_
                                insideConstrainedOut = d_9_ni_
                                currentConstrainedOut = d_10_nc_
                                d_2_steps_ = (d_2_steps_) + (1)
                            elif True:
                                d_11_next_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_11_next_ = out6_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_11_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                    if (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                        d_12_ng_: _dafny.Seq
                                        d_13_ni_: bool
                                        d_14_nc_: _dafny.Seq
                                        out7_: _dafny.Seq
                                        out8_: bool
                                        out9_: _dafny.Seq
                                        out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                        d_12_ng_ = out7_
                                        d_13_ni_ = out8_
                                        d_14_nc_ = out9_
                                        generated = d_12_ng_
                                        insideConstrainedOut = d_13_ni_
                                        currentConstrainedOut = d_14_nc_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_closedGenerated_: _dafny.Seq
                        d_16_closedInside_: bool
                        d_17_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_closedGenerated_ = out10_
                        d_16_closedInside_ = out11_
                        d_17_closedCurrent_ = out12_
                        generated = d_15_closedGenerated_
                        insideConstrainedOut = d_16_closedInside_
                        currentConstrainedOut = d_17_closedCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_next_: _dafny.Seq
                        out13_: _dafny.Seq
                        out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_19_next_ = out13_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_19_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_20_appendedGenerated_: _dafny.Seq
                            d_21_appendedInside_: bool
                            d_22_appendedCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                            d_20_appendedGenerated_ = out14_
                            d_21_appendedInside_ = out15_
                            d_22_appendedCurrent_ = out16_
                            generated = d_20_appendedGenerated_
                            insideConstrainedOut = d_21_appendedInside_
                            currentConstrainedOut = d_22_appendedCurrent_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


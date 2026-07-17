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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Show all intermediate calculations in your reasoning. At the very end, write the final numeric answer inside << >> delimiters. Use exactly one << >> span containing only the final number, like <<42>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_2_steps_) + (10)) >= (maxSteps):
                            d_3_ng_: _dafny.Seq
                            d_4_ni_: bool
                            d_5_nc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_3_ng_ = out0_
                            d_4_ni_ = out1_
                            d_5_nc_ = out2_
                            generated = d_3_ng_
                            insideConstrainedOut = d_4_ni_
                            currentConstrainedOut = d_5_nc_
                            d_2_steps_ = (d_2_steps_) + (1)
                        elif True:
                            d_6_chunkBudget_: int
                            d_6_chunkBudget_ = ((maxSteps) - (d_2_steps_)) - (10)
                            if (d_6_chunkBudget_) == (0):
                                d_6_chunkBudget_ = 1
                            if (d_6_chunkBudget_) > (50):
                                d_6_chunkBudget_ = 50
                            d_7_generatedOut_: _dafny.Seq
                            d_8_stoppedOnOpenSpan_: bool
                            d_9_stoppedOnEos_: bool
                            d_10_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_7_generatedOut_ = out3_
                            d_8_stoppedOnOpenSpan_ = out4_
                            d_9_stoppedOnEos_ = out5_
                            d_10_stepsUsed_ = out6_
                            generated = d_7_generatedOut_
                            d_2_steps_ = (d_2_steps_) + (d_10_stepsUsed_)
                            if d_8_stoppedOnOpenSpan_:
                                d_11_ng_: _dafny.Seq
                                d_12_ni_: bool
                                d_13_nc_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_11_ng_ = out7_
                                d_12_ni_ = out8_
                                d_13_nc_ = out9_
                                generated = d_11_ng_
                                insideConstrainedOut = d_12_ni_
                                currentConstrainedOut = d_13_nc_
                            elif d_9_stoppedOnEos_:
                                if ((d_2_steps_) + (8)) <= (maxSteps):
                                    d_14_ng_: _dafny.Seq
                                    d_15_ni_: bool
                                    d_16_nc_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_14_ng_ = out10_
                                    d_15_ni_ = out11_
                                    d_16_nc_ = out12_
                                    generated = d_14_ng_
                                    insideConstrainedOut = d_15_ni_
                                    currentConstrainedOut = d_16_nc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_17_closedGenerated_: _dafny.Seq
                        d_18_closedInside_: bool
                        d_19_closedCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_closedGenerated_ = out13_
                        d_18_closedInside_ = out14_
                        d_19_closedCurrent_ = out15_
                        generated = d_17_closedGenerated_
                        insideConstrainedOut = d_18_closedInside_
                        currentConstrainedOut = d_19_closedCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_21_next_: _dafny.Seq
                        out16_: _dafny.Seq
                        out16_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_21_next_ = out16_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_21_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_22_appendedGenerated_: _dafny.Seq
                            d_23_appendedInside_: bool
                            d_24_appendedCurrent_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: _dafny.Seq
                            out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                            d_22_appendedGenerated_ = out17_
                            d_23_appendedInside_ = out18_
                            d_24_appendedCurrent_ = out19_
                            generated = d_22_appendedGenerated_
                            insideConstrainedOut = d_23_appendedInside_
                            currentConstrainedOut = d_24_appendedCurrent_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


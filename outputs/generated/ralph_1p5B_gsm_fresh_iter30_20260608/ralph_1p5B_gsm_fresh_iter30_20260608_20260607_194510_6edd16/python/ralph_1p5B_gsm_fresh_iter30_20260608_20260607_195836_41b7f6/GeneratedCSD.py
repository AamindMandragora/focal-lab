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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Show all calculations. At the end, write the final numeric answer inside << >> like this: <<42>>. Use exactly one << >> span with only the number inside, no text."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_reservedSteps_: int
        d_3_reservedSteps_ = 60
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
                            d_7_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                if ((d_2_steps_) + (1)) < (maxSteps):
                                    d_8_ng_: _dafny.Seq
                                    d_9_ni_: bool
                                    d_10_nc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_8_ng_ = out4_
                                    d_9_ni_ = out5_
                                    d_10_nc_ = out6_
                                    generated = d_8_ng_
                                    insideConstrainedOut = d_9_ni_
                                    currentConstrainedOut = d_10_nc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
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
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_14_closedGenerated_: _dafny.Seq
                        d_15_closedInside_: bool
                        d_16_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_closedGenerated_ = out10_
                        d_15_closedInside_ = out11_
                        d_16_closedCurrent_ = out12_
                        generated = d_14_closedGenerated_
                        insideConstrainedOut = d_15_closedInside_
                        currentConstrainedOut = d_16_closedCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_18_next_: _dafny.Seq
                        out13_: _dafny.Seq
                        out13_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
                        d_18_next_ = out13_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_18_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_19_appendedGenerated_: _dafny.Seq
                            d_20_appendedInside_: bool
                            d_21_appendedCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                            d_19_appendedGenerated_ = out14_
                            d_20_appendedInside_ = out15_
                            d_21_appendedCurrent_ = out16_
                            generated = d_19_appendedGenerated_
                            insideConstrainedOut = d_20_appendedInside_
                            currentConstrainedOut = d_21_appendedCurrent_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


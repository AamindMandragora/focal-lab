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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Ensure the response contains visible << >> spans. Put intermediate symbolic expressions and the final answer inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openedSpans_: int
        if insideConstrained:
            d_2_openedSpans_ = 1
        elif True:
            d_2_openedSpans_ = 0
        d_3_maxForcedSpans_: int
        d_3_maxForcedSpans_ = 2
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_sinceClose_: int
                        d_4_sinceClose_ = 0
                        out0_: int
                        out0_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                        d_4_sinceClose_ = out0_
                        if (d_2_openedSpans_) == (0):
                            d_5_openedGenerated_: _dafny.Seq
                            d_6_openedInside_: bool
                            d_7_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedGenerated_ = out1_
                            d_6_openedInside_ = out2_
                            d_7_openedCurrent_ = out3_
                            generated = d_5_openedGenerated_
                            insideConstrainedOut = d_6_openedInside_
                            currentConstrainedOut = d_7_openedCurrent_
                            d_2_openedSpans_ = (d_2_openedSpans_) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif ((d_2_openedSpans_) < (d_3_maxForcedSpans_)) and ((d_4_sinceClose_) >= (8)):
                            d_8_openedGenerated2_: _dafny.Seq
                            d_9_openedInside2_: bool
                            d_10_openedCurrent2_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_openedGenerated2_ = out4_
                            d_9_openedInside2_ = out5_
                            d_10_openedCurrent2_ = out6_
                            generated = d_8_openedGenerated2_
                            insideConstrainedOut = d_9_openedInside2_
                            currentConstrainedOut = d_10_openedCurrent2_
                            d_2_openedSpans_ = (d_2_openedSpans_) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_next_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                if (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_12_enteredGenerated_: _dafny.Seq
                                    d_13_enteredInside_: bool
                                    d_14_enteredCurrent_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_12_enteredGenerated_ = out8_
                                    d_13_enteredInside_ = out9_
                                    d_14_enteredCurrent_ = out10_
                                    generated = d_12_enteredGenerated_
                                    insideConstrainedOut = d_13_enteredInside_
                                    currentConstrainedOut = d_14_enteredCurrent_
                                    if (d_2_openedSpans_) < (d_3_maxForcedSpans_):
                                        d_2_openedSpans_ = (d_2_openedSpans_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_closedGenerated_: _dafny.Seq
                        d_16_closedInside_: bool
                        d_17_closedCurrent_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_closedGenerated_ = out11_
                        d_16_closedInside_ = out12_
                        d_17_closedCurrent_ = out13_
                        generated = d_15_closedGenerated_
                        insideConstrainedOut = d_16_closedInside_
                        currentConstrainedOut = d_17_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_18_stablePrefix_: _dafny.Seq
                        d_18_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (d_18_stablePrefix_)
                        d_20_nextIn_: _dafny.Seq
                        out14_: _dafny.Seq
                        out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_20_nextIn_ = out14_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_20_nextIn_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_21_appendedGenerated_: _dafny.Seq
                            d_22_appendedInside_: bool
                            d_23_appendedCurrent_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_nextIn_)
                            d_21_appendedGenerated_ = out15_
                            d_22_appendedInside_ = out16_
                            d_23_appendedCurrent_ = out17_
                            generated = d_21_appendedGenerated_
                            insideConstrainedOut = d_22_appendedInside_
                            currentConstrainedOut = d_23_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


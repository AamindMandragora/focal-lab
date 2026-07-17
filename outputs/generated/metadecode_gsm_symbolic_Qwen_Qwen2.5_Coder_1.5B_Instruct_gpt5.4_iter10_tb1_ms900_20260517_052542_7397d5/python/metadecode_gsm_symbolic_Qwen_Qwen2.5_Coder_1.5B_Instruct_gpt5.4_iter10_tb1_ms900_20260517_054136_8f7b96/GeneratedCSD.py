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
        if (maxSteps) == (0):
            pass
        elif True:
            (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. Write each arithmetic computation inside visible << >> delimiters.")))
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_openedAnySpan_: bool
            d_2_openedAnySpan_ = insideConstrained
            d_3_delayedOpenThreshold_: int
            d_3_delayedOpenThreshold_ = 2
            with _dafny.label("1_0"):
                while (d_1_steps_) < (maxSteps):
                    with _dafny.c_label("1_0"):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_1_steps_)
                        if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_4_remaining_) == (1)):
                            d_5_closedGenerated0_: _dafny.Seq
                            d_6_closedInside0_: bool
                            d_7_closedCurrent0_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_5_closedGenerated0_ = out0_
                            d_6_closedInside0_ = out1_
                            d_7_closedCurrent0_ = out2_
                            generated = d_5_closedGenerated0_
                            insideConstrainedOut = d_6_closedInside0_
                            currentConstrainedOut = d_7_closedCurrent0_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (insideConstrainedOut) and ((d_4_remaining_) == (1)):
                            d_8_nextFinal_: _dafny.Seq
                            d_9_wasConstrainedFinal_: bool
                            out3_: _dafny.Seq
                            out4_: bool
                            out3_, out4_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])), currentConstrainedOut, eosToken)
                            d_8_nextFinal_ = out3_
                            d_9_wasConstrainedFinal_ = out4_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_nextFinal_) != (eosToken):
                                d_10_appendedGeneratedFinal_: _dafny.Seq
                                d_11_appendedInsideFinal_: bool
                                d_12_appendedCurrentFinal_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_nextFinal_)
                                d_10_appendedGeneratedFinal_ = out5_
                                d_11_appendedInsideFinal_ = out6_
                                d_12_appendedCurrentFinal_ = out7_
                                generated = d_10_appendedGeneratedFinal_
                                insideConstrainedOut = d_11_appendedInsideFinal_
                                currentConstrainedOut = d_12_appendedCurrentFinal_
                            raise _dafny.Break("1_0")
                        elif not(insideConstrainedOut):
                            if (not(d_2_openedAnySpan_)) and ((d_1_steps_) >= (d_3_delayedOpenThreshold_)):
                                d_13_openedGenerated_: _dafny.Seq
                                d_14_openedInside_: bool
                                d_15_openedCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_13_openedGenerated_ = out8_
                                d_14_openedInside_ = out9_
                                d_15_openedCurrent_ = out10_
                                generated = d_13_openedGenerated_
                                insideConstrainedOut = d_14_openedInside_
                                currentConstrainedOut = d_15_openedCurrent_
                                d_2_openedAnySpan_ = True
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_16_next_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_16_next_ = out11_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_16_next_) == (eosToken):
                                    raise _dafny.Break("1_0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_16_next_]))
                                    if (d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                        d_17_observedGenerated_: _dafny.Seq
                                        d_18_observedInside_: bool
                                        d_19_observedCurrent_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out13_: bool
                                        out14_: _dafny.Seq
                                        out12_, out13_, out14_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                        d_17_observedGenerated_ = out12_
                                        d_18_observedInside_ = out13_
                                        d_19_observedCurrent_ = out14_
                                        generated = d_17_observedGenerated_
                                        insideConstrainedOut = d_18_observedInside_
                                        currentConstrainedOut = d_19_observedCurrent_
                                        d_2_openedAnySpan_ = True
                        elif (parser).IsCompletePrefix(currentConstrainedOut):
                            d_20_closedGenerated_: _dafny.Seq
                            d_21_closedInside_: bool
                            d_22_closedCurrent_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_20_closedGenerated_ = out15_
                            d_21_closedInside_ = out16_
                            d_22_closedCurrent_ = out17_
                            generated = d_20_closedGenerated_
                            insideConstrainedOut = d_21_closedInside_
                            currentConstrainedOut = d_22_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_23_constrainedPrompt_: _dafny.Seq
                            d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_24_next_: _dafny.Seq
                            d_25_wasConstrained_: bool
                            out18_: _dafny.Seq
                            out19_: bool
                            out18_, out19_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_24_next_ = out18_
                            d_25_wasConstrained_ = out19_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_24_next_) == (eosToken):
                                raise _dafny.Break("1_0")
                            elif True:
                                d_26_appendedGenerated_: _dafny.Seq
                                d_27_appendedInside_: bool
                                d_28_appendedCurrent_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                d_26_appendedGenerated_ = out20_
                                d_27_appendedInside_ = out21_
                                d_28_appendedCurrent_ = out22_
                                generated = d_26_appendedGenerated_
                                insideConstrainedOut = d_27_appendedInside_
                                currentConstrainedOut = d_28_appendedCurrent_
                        pass
                pass
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


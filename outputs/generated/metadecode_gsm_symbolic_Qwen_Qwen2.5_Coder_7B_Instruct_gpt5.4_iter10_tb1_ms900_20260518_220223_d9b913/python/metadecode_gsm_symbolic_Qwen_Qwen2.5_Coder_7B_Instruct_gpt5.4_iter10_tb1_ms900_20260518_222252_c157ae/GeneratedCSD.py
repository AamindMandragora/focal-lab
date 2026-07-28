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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Write every arithmetic computation inside visible << >> delimiters, and keep each such span to just the computation.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openCount_: int
        d_2_openCount_ = 0
        if not(insideConstrained):
            out0_: int
            out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generatedPrefix, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
            d_2_openCount_ = out0_
        d_3_sawAnyOpen_: bool
        d_3_sawAnyOpen_ = insideConstrained
        if not(d_3_sawAnyOpen_):
            d_3_sawAnyOpen_ = (d_2_openCount_) > (0)
        d_4_forcedOpenUsed_: bool
        d_4_forcedOpenUsed_ = False
        d_5_minOutsideBeforeFallback_: int
        d_5_minOutsideBeforeFallback_ = 24
        d_6_fallbackWindow_: int
        d_6_fallbackWindow_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (((not(d_3_sawAnyOpen_)) and (not(d_4_forcedOpenUsed_))) and (((d_1_steps_) + (d_6_fallbackWindow_)) >= (maxSteps))) and ((len(generated)) >= ((len(generatedPrefix)) + (d_5_minOutsideBeforeFallback_))):
                            d_7_openedGenerated_: _dafny.Seq
                            d_8_openedInside_: bool
                            d_9_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_openedGenerated_ = out1_
                            d_8_openedInside_ = out2_
                            d_9_openedCurrent_ = out3_
                            generated = d_7_openedGenerated_
                            insideConstrainedOut = d_8_openedInside_
                            currentConstrainedOut = d_9_openedCurrent_
                            d_4_forcedOpenUsed_ = True
                            d_3_sawAnyOpen_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_next_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_next_ = out4_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                                if (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_11_enteredGenerated_: _dafny.Seq
                                    d_12_enteredInside_: bool
                                    d_13_enteredCurrent_: _dafny.Seq
                                    out5_: _dafny.Seq
                                    out6_: bool
                                    out7_: _dafny.Seq
                                    out5_, out6_, out7_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_11_enteredGenerated_ = out5_
                                    d_12_enteredInside_ = out6_
                                    d_13_enteredCurrent_ = out7_
                                    generated = d_11_enteredGenerated_
                                    insideConstrainedOut = d_12_enteredInside_
                                    currentConstrainedOut = d_13_enteredCurrent_
                                    d_3_sawAnyOpen_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_14_closedGenerated_: _dafny.Seq
                        d_15_closedInside_: bool
                        d_16_closedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_closedGenerated_ = out8_
                        d_15_closedInside_ = out9_
                        d_16_closedCurrent_ = out10_
                        generated = d_14_closedGenerated_
                        insideConstrainedOut = d_15_closedInside_
                        currentConstrainedOut = d_16_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_17_stablePrefix_: _dafny.Seq
                        d_17_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (d_17_stablePrefix_)
                        d_19_nextIn_: _dafny.Seq
                        d_20_wasConstrained_: bool
                        out11_: _dafny.Seq
                        out12_: bool
                        out11_, out12_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_19_nextIn_ = out11_
                        d_20_wasConstrained_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_19_nextIn_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_21_appendedGenerated_: _dafny.Seq
                            d_22_appendedInside_: bool
                            d_23_appendedCurrent_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_nextIn_)
                            d_21_appendedGenerated_ = out13_
                            d_22_appendedInside_ = out14_
                            d_23_appendedCurrent_ = out15_
                            generated = d_21_appendedGenerated_
                            insideConstrainedOut = d_22_appendedInside_
                            currentConstrainedOut = d_23_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


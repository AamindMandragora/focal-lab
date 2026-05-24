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
        if True:
            generated = generatedPrefix
            insideConstrainedOut = insideConstrained
            currentConstrainedOut = currentConstrained
            cost = 0
            if (maxSteps) == (0):
                cost = 0
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Put each arithmetic computation inside visible << >> delimiters.")))
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_didPrelude_: bool
            d_2_didPrelude_ = insideConstrainedOut
            with _dafny.label("0"):
                while (d_1_steps_) < (maxSteps):
                    with _dafny.c_label("0"):
                        if not(insideConstrainedOut):
                            if not(d_2_didPrelude_):
                                d_3_remainingPrelude_: int
                                d_3_remainingPrelude_ = (maxSteps) - (d_1_steps_)
                                d_4_preludeBudget_: int
                                if (d_3_remainingPrelude_) <= (3):
                                    d_4_preludeBudget_ = d_3_remainingPrelude_
                                elif True:
                                    d_4_preludeBudget_ = 3
                                d_5_chunkedG_: _dafny.Seq
                                d_6_stoppedOpen_: bool
                                d_7_stoppedEos_: bool
                                d_8_stepsUsed_: int
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: bool
                                out3_: int
                                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_preludeBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_5_chunkedG_ = out0_
                                d_6_stoppedOpen_ = out1_
                                d_7_stoppedEos_ = out2_
                                d_8_stepsUsed_ = out3_
                                generated = d_5_chunkedG_
                                d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                                d_2_didPrelude_ = True
                                if d_7_stoppedEos_:
                                    raise _dafny.Break("0")
                                elif d_6_stoppedOpen_:
                                    d_9_enteredGenerated_: _dafny.Seq
                                    d_10_enteredInside_: bool
                                    d_11_enteredCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_9_enteredGenerated_ = out4_
                                    d_10_enteredInside_ = out5_
                                    d_11_enteredCurrent_ = out6_
                                    generated = d_9_enteredGenerated_
                                    insideConstrainedOut = d_10_enteredInside_
                                    currentConstrainedOut = d_11_enteredCurrent_
                            elif True:
                                d_12_openedGenerated_: _dafny.Seq
                                d_13_openedInside_: bool
                                d_14_openedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_12_openedGenerated_ = out7_
                                d_13_openedInside_ = out8_
                                d_14_openedCurrent_ = out9_
                                generated = d_12_openedGenerated_
                                insideConstrainedOut = d_13_openedInside_
                                currentConstrainedOut = d_14_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
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
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_18_constrainedPrompt_: _dafny.Seq
                            d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_19_next_: _dafny.Seq
                            d_19_next_ = eosToken
                            if (len(currentConstrainedOut)) < (2):
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_19_next_ = out13_
                            elif True:
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_19_next_ = out14_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_19_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_20_appendedGenerated_: _dafny.Seq
                                d_21_appendedInside_: bool
                                d_22_appendedCurrent_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                d_20_appendedGenerated_ = out15_
                                d_21_appendedInside_ = out16_
                                d_22_appendedCurrent_ = out17_
                                generated = d_20_appendedGenerated_
                                insideConstrainedOut = d_21_appendedInside_
                                currentConstrainedOut = d_22_appendedCurrent_
                        pass
                pass
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


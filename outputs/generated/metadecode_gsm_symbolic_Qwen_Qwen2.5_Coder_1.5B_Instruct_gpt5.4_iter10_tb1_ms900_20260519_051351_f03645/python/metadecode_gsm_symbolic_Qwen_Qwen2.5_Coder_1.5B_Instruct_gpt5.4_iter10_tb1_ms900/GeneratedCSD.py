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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step, and put only the final numeric answer inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openedAnswerSpan_: bool
        d_2_openedAnswerSpan_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remaining_: int
                        d_3_remaining_ = (maxSteps) - (d_1_steps_)
                        if (not(d_2_openedAnswerSpan_)) and ((d_3_remaining_) <= (2)):
                            d_4_openedGenerated_: _dafny.Seq
                            d_5_openedInside_: bool
                            d_6_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openedGenerated_ = out0_
                            d_5_openedInside_ = out1_
                            d_6_openedCurrent_ = out2_
                            generated = d_4_openedGenerated_
                            insideConstrainedOut = d_5_openedInside_
                            currentConstrainedOut = d_6_openedCurrent_
                            d_2_openedAnswerSpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_8_enteredGenerated_: _dafny.Seq
                                    d_9_enteredInside_: bool
                                    d_10_enteredCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_8_enteredGenerated_ = out4_
                                    d_9_enteredInside_ = out5_
                                    d_10_enteredCurrent_ = out6_
                                    generated = d_8_enteredGenerated_
                                    insideConstrainedOut = d_9_enteredInside_
                                    currentConstrainedOut = d_10_enteredCurrent_
                                    d_2_openedAnswerSpan_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_11_closedGenerated_: _dafny.Seq
                        d_12_closedInside_: bool
                        d_13_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_11_closedGenerated_ = out7_
                        d_12_closedInside_ = out8_
                        d_13_closedCurrent_ = out9_
                        generated = d_11_closedGenerated_
                        insideConstrainedOut = d_12_closedInside_
                        currentConstrainedOut = d_13_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_14_stablePrefix_: _dafny.Seq
                        d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                        d_16_remainingInside_: int
                        d_16_remainingInside_ = (maxSteps) - (d_1_steps_)
                        d_17_validCount_: int
                        out10_: int
                        out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_17_validCount_ = out10_
                        if (((d_17_validCount_) <= (12)) or ((stepTokenBudget) <= (1))) or ((d_16_remainingInside_) <= (1)):
                            d_18_nextIn_: _dafny.Seq
                            d_18_nextIn_ = eosToken
                            d_19_narrow_: bool
                            out11_: bool
                            out11_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 3)
                            d_19_narrow_ = out11_
                            if d_19_narrow_:
                                d_20_nextSoft_: _dafny.Seq
                                d_21_usedFallback_: bool
                                out12_: _dafny.Seq
                                out13_: bool
                                out12_, out13_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                                d_20_nextSoft_ = out12_
                                d_21_usedFallback_ = out13_
                                d_18_nextIn_ = d_20_nextSoft_
                            elif (len(validTokenGroups)) > (0):
                                d_22_nextAdaptive_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_22_nextAdaptive_ = out14_
                                d_18_nextIn_ = d_22_nextAdaptive_
                            elif True:
                                d_23_nextTemp_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e-1'), eosToken)
                                d_23_nextTemp_ = out15_
                                d_18_nextIn_ = d_23_nextTemp_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_18_nextIn_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_24_appendedGenerated_: _dafny.Seq
                                d_25_appendedInside_: bool
                                d_26_appendedCurrent_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_nextIn_)
                                d_24_appendedGenerated_ = out16_
                                d_25_appendedInside_ = out17_
                                d_26_appendedCurrent_ = out18_
                                generated = d_24_appendedGenerated_
                                insideConstrainedOut = d_25_appendedInside_
                                currentConstrainedOut = d_26_appendedCurrent_
                        elif True:
                            d_27_symbolBudget_: int
                            if (stepTokenBudget) > (d_16_remainingInside_):
                                d_27_symbolBudget_ = d_16_remainingInside_
                            elif True:
                                d_27_symbolBudget_ = stepTokenBudget
                            d_28_symbolGenerated_: _dafny.Seq
                            d_29_symbolCurrent_: _dafny.Seq
                            d_30_hitEos_: bool
                            d_31_stepsUsed_: int
                            out19_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: int
                            out19_, out20_, out21_, out22_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_15_constrainedPrompt_, generated, currentConstrainedOut, d_27_symbolBudget_, eosToken)
                            d_28_symbolGenerated_ = out19_
                            d_29_symbolCurrent_ = out20_
                            d_30_hitEos_ = out21_
                            d_31_stepsUsed_ = out22_
                            generated = d_28_symbolGenerated_
                            currentConstrainedOut = d_29_symbolCurrent_
                            d_1_steps_ = (d_1_steps_) + (d_31_stepsUsed_)
                            if d_30_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


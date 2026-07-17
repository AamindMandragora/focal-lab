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
        d_3_freePrefixCap_: int
        d_3_freePrefixCap_ = 48
        d_4_narrowThreshold_: int
        d_4_narrowThreshold_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_2_openedAnswerSpan_)) and ((d_1_steps_) >= (d_3_freePrefixCap_)):
                            d_5_openedGenerated_: _dafny.Seq
                            d_6_openedInside_: bool
                            d_7_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedGenerated_ = out0_
                            d_6_openedInside_ = out1_
                            d_7_openedCurrent_ = out2_
                            generated = d_5_openedGenerated_
                            insideConstrainedOut = d_6_openedInside_
                            currentConstrainedOut = d_7_openedCurrent_
                            d_2_openedAnswerSpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                if (not(d_2_openedAnswerSpan_)) and ((d_1_steps_) < (maxSteps)):
                                    d_9_openedGenerated2_: _dafny.Seq
                                    d_10_openedInside2_: bool
                                    d_11_openedCurrent2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_9_openedGenerated2_ = out4_
                                    d_10_openedInside2_ = out5_
                                    d_11_openedCurrent2_ = out6_
                                    generated = d_9_openedGenerated2_
                                    insideConstrainedOut = d_10_openedInside2_
                                    currentConstrainedOut = d_11_openedCurrent2_
                                    d_2_openedAnswerSpan_ = True
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_12_enteredGenerated_: _dafny.Seq
                                    d_13_enteredInside_: bool
                                    d_14_enteredCurrent_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_12_enteredGenerated_ = out7_
                                    d_13_enteredInside_ = out8_
                                    d_14_enteredCurrent_ = out9_
                                    generated = d_12_enteredGenerated_
                                    insideConstrainedOut = d_13_enteredInside_
                                    currentConstrainedOut = d_14_enteredCurrent_
                                    d_2_openedAnswerSpan_ = True
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
                        raise _dafny.Break("0")
                    elif True:
                        d_18_stablePrefix_: _dafny.Seq
                        d_18_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (d_18_stablePrefix_)
                        d_20_remainingInside_: int
                        d_20_remainingInside_ = (maxSteps) - (d_1_steps_)
                        d_21_validCount_: int
                        out13_: int
                        out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_21_validCount_ = out13_
                        if (((d_21_validCount_) <= (d_4_narrowThreshold_)) or ((stepTokenBudget) <= (1))) or ((d_20_remainingInside_) <= (1)):
                            d_22_nextIn_: _dafny.Seq
                            d_22_nextIn_ = eosToken
                            d_23_narrow_: bool
                            out14_: bool
                            out14_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 3)
                            d_23_narrow_ = out14_
                            if d_23_narrow_:
                                d_24_nextSoft_: _dafny.Seq
                                d_25_usedFallback_: bool
                                out15_: _dafny.Seq
                                out16_: bool
                                out15_, out16_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                                d_24_nextSoft_ = out15_
                                d_25_usedFallback_ = out16_
                                d_22_nextIn_ = d_24_nextSoft_
                            elif (len(validTokenGroups)) > (0):
                                d_26_nextAdaptive_: _dafny.Seq
                                out17_: _dafny.Seq
                                out17_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_26_nextAdaptive_ = out17_
                                d_22_nextIn_ = d_26_nextAdaptive_
                            elif True:
                                d_27_nextTemp_: _dafny.Seq
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e-1'), eosToken)
                                d_27_nextTemp_ = out18_
                                d_22_nextIn_ = d_27_nextTemp_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_nextIn_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_28_appendedGenerated_: _dafny.Seq
                                d_29_appendedInside_: bool
                                d_30_appendedCurrent_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_nextIn_)
                                d_28_appendedGenerated_ = out19_
                                d_29_appendedInside_ = out20_
                                d_30_appendedCurrent_ = out21_
                                generated = d_28_appendedGenerated_
                                insideConstrainedOut = d_29_appendedInside_
                                currentConstrainedOut = d_30_appendedCurrent_
                        elif True:
                            d_31_symbolBudget_: int
                            if (stepTokenBudget) > (d_20_remainingInside_):
                                d_31_symbolBudget_ = d_20_remainingInside_
                            elif True:
                                d_31_symbolBudget_ = stepTokenBudget
                            d_32_symbolGenerated_: _dafny.Seq
                            d_33_symbolCurrent_: _dafny.Seq
                            d_34_hitEos_: bool
                            d_35_stepsUsed_: int
                            out22_: _dafny.Seq
                            out23_: _dafny.Seq
                            out24_: bool
                            out25_: int
                            out22_, out23_, out24_, out25_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_19_constrainedPrompt_, generated, currentConstrainedOut, d_31_symbolBudget_, eosToken)
                            d_32_symbolGenerated_ = out22_
                            d_33_symbolCurrent_ = out23_
                            d_34_hitEos_ = out24_
                            d_35_stepsUsed_ = out25_
                            generated = d_32_symbolGenerated_
                            currentConstrainedOut = d_33_symbolCurrent_
                            d_1_steps_ = (d_1_steps_) + (d_35_stepsUsed_)
                            if d_34_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


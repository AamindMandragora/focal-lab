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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_rollbackLimit_: int
        d_2_rollbackLimit_ = 32
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 10
        d_4_outsideSinceSpan_: int
        d_4_outsideSinceSpan_ = 0
        d_5_forceOpenAfter_: int
        d_5_forceOpenAfter_ = 4
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_4_outsideSinceSpan_) >= (d_5_forceOpenAfter_):
                            d_6_openedGenerated_: _dafny.Seq
                            d_7_openedInside_: bool
                            d_8_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openedGenerated_ = out0_
                            d_7_openedInside_ = out1_
                            d_8_openedCurrent_ = out2_
                            generated = d_6_openedGenerated_
                            insideConstrainedOut = d_7_openedInside_
                            currentConstrainedOut = d_8_openedCurrent_
                            d_4_outsideSinceSpan_ = 0
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_nextOutside_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_nextOutside_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_nextOutside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_nextOutside_]))
                                if (d_9_nextOutside_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_10_enteredGenerated_: _dafny.Seq
                                    d_11_enteredInside_: bool
                                    d_12_enteredCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_10_enteredGenerated_ = out4_
                                    d_11_enteredInside_ = out5_
                                    d_12_enteredCurrent_ = out6_
                                    generated = d_10_enteredGenerated_
                                    insideConstrainedOut = d_11_enteredInside_
                                    currentConstrainedOut = d_12_enteredCurrent_
                                    d_4_outsideSinceSpan_ = 0
                                elif True:
                                    d_4_outsideSinceSpan_ = (d_4_outsideSinceSpan_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_closedGenerated_: _dafny.Seq
                        d_14_closedInside_: bool
                        d_15_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_closedGenerated_ = out7_
                        d_14_closedInside_ = out8_
                        d_15_closedCurrent_ = out9_
                        generated = d_13_closedGenerated_
                        insideConstrainedOut = d_14_closedInside_
                        currentConstrainedOut = d_15_closedCurrent_
                        d_4_outsideSinceSpan_ = 0
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_2_rollbackLimit_):
                        d_16_rolledGenerated_: _dafny.Seq
                        d_17_rolledCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_16_rolledGenerated_ = out10_
                        d_17_rolledCurrent_ = out11_
                        generated = d_16_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_17_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_18_stablePrefix_: _dafny.Seq
                        d_18_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (d_18_stablePrefix_)
                        d_20_validCount_: int
                        out12_: int
                        out12_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_20_validCount_ = out12_
                        if ((stepTokenBudget) > (1)) and ((d_20_validCount_) > (d_3_narrowThreshold_)):
                            d_21_remaining_: int
                            d_21_remaining_ = (maxSteps) - (d_1_steps_)
                            d_22_symbolBudget_: int
                            if (stepTokenBudget) > (d_21_remaining_):
                                d_22_symbolBudget_ = d_21_remaining_
                            elif True:
                                d_22_symbolBudget_ = stepTokenBudget
                            d_23_symbolGenerated_: _dafny.Seq
                            d_24_symbolCurrent_: _dafny.Seq
                            d_25_hitEos_: bool
                            d_26_stepsUsed_: int
                            out13_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: int
                            out13_, out14_, out15_, out16_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_19_constrainedPrompt_, generated, currentConstrainedOut, d_22_symbolBudget_, eosToken)
                            d_23_symbolGenerated_ = out13_
                            d_24_symbolCurrent_ = out14_
                            d_25_hitEos_ = out15_
                            d_26_stepsUsed_ = out16_
                            generated = d_23_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_24_symbolCurrent_
                            d_1_steps_ = (d_1_steps_) + (d_26_stepsUsed_)
                            if d_25_hitEos_:
                                raise _dafny.Break("0")
                        elif True:
                            d_27_next_: _dafny.Seq
                            d_27_next_ = eosToken
                            d_28_isDeadEnd_: bool
                            out17_: bool
                            out17_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_28_isDeadEnd_ = out17_
                            if d_28_isDeadEnd_:
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_27_next_ = out18_
                            elif (len(currentConstrainedOut)) < (2):
                                out19_: _dafny.Seq
                                out19_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('6e0'), eosToken)
                                d_27_next_ = out19_
                            elif (d_20_validCount_) <= (d_3_narrowThreshold_):
                                d_29_gatedNext_: _dafny.Seq
                                d_30_wasConstrained_: bool
                                out20_: _dafny.Seq
                                out21_: bool
                                out20_, out21_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_29_gatedNext_ = out20_
                                d_30_wasConstrained_ = out21_
                                d_27_next_ = d_29_gatedNext_
                            elif True:
                                out22_: _dafny.Seq
                                out22_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_27_next_ = out22_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_27_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_31_appendedGenerated_: _dafny.Seq
                                d_32_appendedInside_: bool
                                d_33_appendedCurrent_: _dafny.Seq
                                out23_: _dafny.Seq
                                out24_: bool
                                out25_: _dafny.Seq
                                out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                                d_31_appendedGenerated_ = out23_
                                d_32_appendedInside_ = out24_
                                d_33_appendedCurrent_ = out25_
                                generated = d_31_appendedGenerated_
                                insideConstrainedOut = d_32_appendedInside_
                                currentConstrainedOut = d_33_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


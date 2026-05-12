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
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((len(generated)) >= (2)) and (((generated)[(len(generated)) - (2)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                            d_3_enteredGenerated_: _dafny.Seq
                            d_4_enteredInside_: bool
                            d_5_enteredCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_3_enteredGenerated_ = out0_
                            d_4_enteredInside_ = out1_
                            d_5_enteredCurrent_ = out2_
                            generated = d_3_enteredGenerated_
                            insideConstrainedOut = d_4_enteredInside_
                            currentConstrainedOut = d_5_enteredCurrent_
                        elif True:
                            d_6_openedGenerated_: _dafny.Seq
                            d_7_openedInside_: bool
                            d_8_openedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openedGenerated_ = out3_
                            d_7_openedInside_ = out4_
                            d_8_openedCurrent_ = out5_
                            generated = d_6_openedGenerated_
                            insideConstrainedOut = d_7_openedInside_
                            currentConstrainedOut = d_8_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_9_closedGenerated_: _dafny.Seq
                        d_10_closedInside_: bool
                        d_11_closedCurrent_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_9_closedGenerated_ = out6_
                        d_10_closedInside_ = out7_
                        d_11_closedCurrent_ = out8_
                        generated = d_9_closedGenerated_
                        insideConstrainedOut = d_10_closedInside_
                        currentConstrainedOut = d_11_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_12_stablePrefix_: _dafny.Seq
                        d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                        d_14_validCount_: int
                        out9_: int
                        out9_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_14_validCount_ = out9_
                        if ((stepTokenBudget) > (1)) and ((d_14_validCount_) > (d_2_narrowThreshold_)):
                            d_15_remaining_: int
                            d_15_remaining_ = (maxSteps) - (d_1_steps_)
                            d_16_symbolBudget_: int
                            if (stepTokenBudget) > (d_15_remaining_):
                                d_16_symbolBudget_ = d_15_remaining_
                            elif True:
                                d_16_symbolBudget_ = stepTokenBudget
                            d_17_symbolGenerated_: _dafny.Seq
                            d_18_symbolOut_: _dafny.Seq
                            d_19_hitEos_: bool
                            d_20_stepsUsed_: int
                            out10_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: int
                            out10_, out11_, out12_, out13_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_13_constrainedPrompt_, generated, currentConstrainedOut, d_16_symbolBudget_, eosToken)
                            d_17_symbolGenerated_ = out10_
                            d_18_symbolOut_ = out11_
                            d_19_hitEos_ = out12_
                            d_20_stepsUsed_ = out13_
                            generated = d_17_symbolGenerated_
                            currentConstrainedOut = d_18_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_20_stepsUsed_)
                            if d_19_hitEos_:
                                raise _dafny.Break("0")
                        elif True:
                            d_21_next_: _dafny.Seq
                            d_21_next_ = eosToken
                            if (d_14_validCount_) <= (2):
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), generated, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                                d_21_next_ = out14_
                            elif (len(currentConstrainedOut)) == (0):
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                                d_21_next_ = out15_
                            elif (len(currentConstrainedOut)) <= (4):
                                d_22_gatedNext_: _dafny.Seq
                                d_23_wasConstrained_: bool
                                out16_: _dafny.Seq
                                out17_: bool
                                out16_, out17_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_22_gatedNext_ = out16_
                                d_23_wasConstrained_ = out17_
                                d_21_next_ = d_22_gatedNext_
                            elif True:
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_21_next_ = out18_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_21_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_24_appendedGenerated_: _dafny.Seq
                                d_25_appendedInside_: bool
                                d_26_appendedCurrent_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                d_24_appendedGenerated_ = out19_
                                d_25_appendedInside_ = out20_
                                d_26_appendedCurrent_ = out21_
                                generated = d_24_appendedGenerated_
                                insideConstrainedOut = d_25_appendedInside_
                                currentConstrainedOut = d_26_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


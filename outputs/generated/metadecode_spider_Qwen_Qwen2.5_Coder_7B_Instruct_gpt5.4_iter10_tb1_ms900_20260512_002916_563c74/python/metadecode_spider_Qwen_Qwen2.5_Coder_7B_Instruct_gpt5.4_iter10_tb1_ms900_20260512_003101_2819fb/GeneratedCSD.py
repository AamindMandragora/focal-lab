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
        d_2_narrowThreshold_ = 8
        d_3_broadThreshold_: int
        d_3_broadThreshold_ = 24
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((len(generated)) > (0)) and (((generated)[(len(generated)) - (1)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                            d_4_enteredGenerated_: _dafny.Seq
                            d_5_enteredInside_: bool
                            d_6_enteredCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_4_enteredGenerated_ = out0_
                            d_5_enteredInside_ = out1_
                            d_6_enteredCurrent_ = out2_
                            generated = d_4_enteredGenerated_
                            insideConstrainedOut = d_5_enteredInside_
                            currentConstrainedOut = d_6_enteredCurrent_
                        elif True:
                            d_7_openedGenerated_: _dafny.Seq
                            d_8_openedInside_: bool
                            d_9_openedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_openedGenerated_ = out3_
                            d_8_openedInside_ = out4_
                            d_9_openedCurrent_ = out5_
                            generated = d_7_openedGenerated_
                            insideConstrainedOut = d_8_openedInside_
                            currentConstrainedOut = d_9_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_10_closedGenerated_: _dafny.Seq
                        d_11_closedInside_: bool
                        d_12_closedCurrent_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_10_closedGenerated_ = out6_
                        d_11_closedInside_ = out7_
                        d_12_closedCurrent_ = out8_
                        generated = d_10_closedGenerated_
                        insideConstrainedOut = d_11_closedInside_
                        currentConstrainedOut = d_12_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_13_stablePrefix_: _dafny.Seq
                        d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_14_constrainedPrompt_: _dafny.Seq
                        d_14_constrainedPrompt_ = (prompt) + (d_13_stablePrefix_)
                        d_15_validCount_: int
                        out9_: int
                        out9_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_15_validCount_ = out9_
                        if ((stepTokenBudget) > (1)) and ((d_15_validCount_) > (d_3_broadThreshold_)):
                            d_16_remaining_: int
                            d_16_remaining_ = (maxSteps) - (d_1_steps_)
                            d_17_symbolBudget_: int
                            if (stepTokenBudget) > (d_16_remaining_):
                                d_17_symbolBudget_ = d_16_remaining_
                            elif True:
                                d_17_symbolBudget_ = stepTokenBudget
                            d_18_symbolGenerated_: _dafny.Seq
                            d_19_symbolOut_: _dafny.Seq
                            d_20_hitEos_: bool
                            d_21_stepsUsed_: int
                            out10_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: int
                            out10_, out11_, out12_, out13_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_14_constrainedPrompt_, generated, currentConstrainedOut, d_17_symbolBudget_, eosToken)
                            d_18_symbolGenerated_ = out10_
                            d_19_symbolOut_ = out11_
                            d_20_hitEos_ = out12_
                            d_21_stepsUsed_ = out13_
                            generated = d_18_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_19_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_21_stepsUsed_)
                            if d_20_hitEos_:
                                raise _dafny.Break("0")
                        elif True:
                            d_22_next_: _dafny.Seq
                            d_22_next_ = eosToken
                            if (len(currentConstrainedOut)) == (0):
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                                d_22_next_ = out14_
                            elif (d_15_validCount_) <= (2):
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_22_next_ = out15_
                            elif (d_15_validCount_) <= (d_2_narrowThreshold_):
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_22_next_ = out16_
                            elif True:
                                d_23_gatedNext_: _dafny.Seq
                                d_24_wasConstrained_: bool
                                out17_: _dafny.Seq
                                out18_: bool
                                out17_, out18_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_23_gatedNext_ = out17_
                                d_24_wasConstrained_ = out18_
                                d_22_next_ = d_23_gatedNext_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_25_appendedGenerated_: _dafny.Seq
                                d_26_appendedInside_: bool
                                d_27_appendedCurrent_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                d_25_appendedGenerated_ = out19_
                                d_26_appendedInside_ = out20_
                                d_27_appendedCurrent_ = out21_
                                generated = d_25_appendedGenerated_
                                insideConstrainedOut = d_26_appendedInside_
                                currentConstrainedOut = d_27_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


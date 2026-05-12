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
        d_2_rollbackLimit_ = 48
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 8
        d_4_mediumThreshold_: int
        d_4_mediumThreshold_ = 24
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
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
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_8_closedGenerated_: _dafny.Seq
                        d_9_closedInside_: bool
                        d_10_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_8_closedGenerated_ = out3_
                        d_9_closedInside_ = out4_
                        d_10_closedCurrent_ = out5_
                        generated = d_8_closedGenerated_
                        insideConstrainedOut = d_9_closedInside_
                        currentConstrainedOut = d_10_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_2_rollbackLimit_):
                        d_11_rolledGenerated_: _dafny.Seq
                        d_12_rolledCurrent_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: _dafny.Seq
                        out6_, out7_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_11_rolledGenerated_ = out6_
                        d_12_rolledCurrent_ = out7_
                        generated = d_11_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_12_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_13_stablePrefix_: _dafny.Seq
                        d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_14_constrainedPrompt_: _dafny.Seq
                        d_14_constrainedPrompt_ = (prompt) + (d_13_stablePrefix_)
                        d_15_validCount_: int
                        out8_: int
                        out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_15_validCount_ = out8_
                        if ((stepTokenBudget) > (1)) and ((d_15_validCount_) > (d_4_mediumThreshold_)):
                            d_16_remaining_: int
                            d_16_remaining_ = (maxSteps) - (d_1_steps_)
                            d_17_symbolBudget_: int
                            if (stepTokenBudget) > (d_16_remaining_):
                                d_17_symbolBudget_ = d_16_remaining_
                            elif True:
                                d_17_symbolBudget_ = stepTokenBudget
                            d_18_symbolGenerated_: _dafny.Seq
                            d_19_symbolCurrent_: _dafny.Seq
                            d_20_hitEos_: bool
                            d_21_stepsUsed_: int
                            out9_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: int
                            out9_, out10_, out11_, out12_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_14_constrainedPrompt_, generated, currentConstrainedOut, d_17_symbolBudget_, eosToken)
                            d_18_symbolGenerated_ = out9_
                            d_19_symbolCurrent_ = out10_
                            d_20_hitEos_ = out11_
                            d_21_stepsUsed_ = out12_
                            generated = d_18_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_19_symbolCurrent_
                            d_1_steps_ = (d_1_steps_) + (d_21_stepsUsed_)
                            if d_20_hitEos_:
                                raise _dafny.Break("0")
                        elif True:
                            d_22_next_: _dafny.Seq
                            d_22_next_ = eosToken
                            d_23_isDeadEnd_: bool
                            out13_: bool
                            out13_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_23_isDeadEnd_ = out13_
                            if d_23_isDeadEnd_:
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_22_next_ = out14_
                            elif (len(currentConstrainedOut)) < (2):
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('6e0'), eosToken)
                                d_22_next_ = out15_
                            elif (d_15_validCount_) <= (d_3_narrowThreshold_):
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_22_next_ = out16_
                            elif True:
                                d_24_gatedNext_: _dafny.Seq
                                d_25_wasConstrained_: bool
                                out17_: _dafny.Seq
                                out18_: bool
                                out17_, out18_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_24_gatedNext_ = out17_
                                d_25_wasConstrained_ = out18_
                                d_22_next_ = d_24_gatedNext_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_26_appendedGenerated_: _dafny.Seq
                                d_27_appendedInside_: bool
                                d_28_appendedCurrent_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                d_26_appendedGenerated_ = out19_
                                d_27_appendedInside_ = out20_
                                d_28_appendedCurrent_ = out21_
                                generated = d_26_appendedGenerated_
                                insideConstrainedOut = d_27_appendedInside_
                                currentConstrainedOut = d_28_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


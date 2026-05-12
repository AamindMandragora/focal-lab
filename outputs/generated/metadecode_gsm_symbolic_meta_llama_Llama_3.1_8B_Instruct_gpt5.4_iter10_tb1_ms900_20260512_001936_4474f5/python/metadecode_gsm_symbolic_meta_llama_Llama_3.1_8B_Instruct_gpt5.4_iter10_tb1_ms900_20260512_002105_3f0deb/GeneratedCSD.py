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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
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
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_7_closedGenerated_: _dafny.Seq
                        d_8_closedInside_: bool
                        d_9_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_7_closedGenerated_ = out3_
                        d_8_closedInside_ = out4_
                        d_9_closedCurrent_ = out5_
                        generated = d_7_closedGenerated_
                        insideConstrainedOut = d_8_closedInside_
                        currentConstrainedOut = d_9_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_2_rollbackLimit_):
                        d_10_rolledGenerated_: _dafny.Seq
                        d_11_rolledCurrent_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: _dafny.Seq
                        out6_, out7_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_10_rolledGenerated_ = out6_
                        d_11_rolledCurrent_ = out7_
                        generated = d_10_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_11_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_12_stablePrefix_: _dafny.Seq
                        d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                        d_14_validCount_: int
                        out8_: int
                        out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_14_validCount_ = out8_
                        if ((stepTokenBudget) > (1)) and ((d_14_validCount_) > (d_3_narrowThreshold_)):
                            d_15_remaining_: int
                            d_15_remaining_ = (maxSteps) - (d_1_steps_)
                            d_16_symbolBudget_: int
                            if (stepTokenBudget) > (d_15_remaining_):
                                d_16_symbolBudget_ = d_15_remaining_
                            elif True:
                                d_16_symbolBudget_ = stepTokenBudget
                            d_17_symbolGenerated_: _dafny.Seq
                            d_18_symbolCurrent_: _dafny.Seq
                            d_19_hitEos_: bool
                            d_20_stepsUsed_: int
                            out9_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: int
                            out9_, out10_, out11_, out12_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_13_constrainedPrompt_, generated, currentConstrainedOut, d_16_symbolBudget_, eosToken)
                            d_17_symbolGenerated_ = out9_
                            d_18_symbolCurrent_ = out10_
                            d_19_hitEos_ = out11_
                            d_20_stepsUsed_ = out12_
                            generated = d_17_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_18_symbolCurrent_
                            d_1_steps_ = (d_1_steps_) + (d_20_stepsUsed_)
                            if d_19_hitEos_:
                                raise _dafny.Break("0")
                        elif True:
                            d_21_next_: _dafny.Seq
                            d_21_next_ = eosToken
                            d_22_isDeadEnd_: bool
                            out13_: bool
                            out13_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_22_isDeadEnd_ = out13_
                            if d_22_isDeadEnd_:
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_21_next_ = out14_
                            elif (len(currentConstrainedOut)) < (2):
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('6e0'), eosToken)
                                d_21_next_ = out15_
                            elif (d_14_validCount_) <= (d_3_narrowThreshold_):
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_21_next_ = out16_
                            elif True:
                                out17_: _dafny.Seq
                                out17_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_21_next_ = out17_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_21_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_23_appendedGenerated_: _dafny.Seq
                                d_24_appendedInside_: bool
                                d_25_appendedCurrent_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                d_23_appendedGenerated_ = out18_
                                d_24_appendedInside_ = out19_
                                d_25_appendedCurrent_ = out20_
                                generated = d_23_appendedGenerated_
                                insideConstrainedOut = d_24_appendedInside_
                                currentConstrainedOut = d_25_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


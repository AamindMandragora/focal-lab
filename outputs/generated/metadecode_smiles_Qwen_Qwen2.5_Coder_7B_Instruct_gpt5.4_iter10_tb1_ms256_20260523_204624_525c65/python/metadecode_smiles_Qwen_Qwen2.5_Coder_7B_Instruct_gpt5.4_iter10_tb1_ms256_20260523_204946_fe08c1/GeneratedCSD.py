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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output the answer as a valid SMILES string inside a constrained span. Start the SMILES immediately when the span opens, avoid prose, and prefer a complete chemically plausible molecule matching the requested class.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_openedGenerated_: _dafny.Seq
                        d_4_openedInside_: bool
                        d_5_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_3_openedGenerated_ = out0_
                        d_4_openedInside_ = out1_
                        d_5_openedCurrent_ = out2_
                        generated = d_3_openedGenerated_
                        insideConstrainedOut = d_4_openedInside_
                        currentConstrainedOut = d_5_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_6_closedGenerated_: _dafny.Seq
                        d_7_closedInside_: bool
                        d_8_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_6_closedGenerated_ = out3_
                        d_7_closedInside_ = out4_
                        d_8_closedCurrent_ = out5_
                        generated = d_6_closedGenerated_
                        insideConstrainedOut = d_7_closedInside_
                        currentConstrainedOut = d_8_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_9_stablePrefix_: _dafny.Seq
                        d_9_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (d_9_stablePrefix_)
                        d_11_remaining_: int
                        d_11_remaining_ = (maxSteps) - (d_1_steps_)
                        d_12_symbolBudget_: int
                        if (stepTokenBudget) == (0):
                            d_12_symbolBudget_ = 1
                        elif (stepTokenBudget) > (d_11_remaining_):
                            d_12_symbolBudget_ = d_11_remaining_
                        elif True:
                            d_12_symbolBudget_ = stepTokenBudget
                        d_13_validCount_: int
                        out6_: int
                        out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_13_validCount_ = out6_
                        if (len(currentConstrainedOut)) < (2):
                            d_14_nextEarly_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('8e0'), eosToken)
                            d_14_nextEarly_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_14_nextEarly_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_15_appendedGeneratedEarly_: _dafny.Seq
                                d_16_appendedInsideEarly_: bool
                                d_17_appendedCurrentEarly_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_nextEarly_)
                                d_15_appendedGeneratedEarly_ = out8_
                                d_16_appendedInsideEarly_ = out9_
                                d_17_appendedCurrentEarly_ = out10_
                                generated = d_15_appendedGeneratedEarly_
                                insideConstrainedOut = d_16_appendedInsideEarly_
                                currentConstrainedOut = d_17_appendedCurrentEarly_
                        elif ((d_13_validCount_) <= (d_2_narrowThreshold_)) or ((d_12_symbolBudget_) == (1)):
                            d_18_next_: _dafny.Seq
                            d_18_next_ = eosToken
                            if (d_13_validCount_) <= (d_2_narrowThreshold_):
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                                d_18_next_ = out11_
                            elif True:
                                out12_: _dafny.Seq
                                out12_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), eosToken)
                                d_18_next_ = out12_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_18_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_19_appendedGenerated_: _dafny.Seq
                                d_20_appendedInside_: bool
                                d_21_appendedCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                d_19_appendedGenerated_ = out13_
                                d_20_appendedInside_ = out14_
                                d_21_appendedCurrent_ = out15_
                                generated = d_19_appendedGenerated_
                                insideConstrainedOut = d_20_appendedInside_
                                currentConstrainedOut = d_21_appendedCurrent_
                        elif True:
                            d_22_symbolGenerated_: _dafny.Seq
                            d_23_symbolOut_: _dafny.Seq
                            d_24_hitEos_: bool
                            d_25_stepsUsed_: int
                            out16_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: int
                            out16_, out17_, out18_, out19_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_10_constrainedPrompt_, generated, currentConstrainedOut, d_12_symbolBudget_, eosToken)
                            d_22_symbolGenerated_ = out16_
                            d_23_symbolOut_ = out17_
                            d_24_hitEos_ = out18_
                            d_25_stepsUsed_ = out19_
                            generated = d_22_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_23_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_25_stepsUsed_)
                            if d_24_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output a single valid SMILES string for the requested molecular class. Prefer starting the molecule early, avoid prose, and when a constrained span is used keep the span content as a complete valid SMILES.")))
        (d_0_helpers_).SetNonDeterministic(lm, False)
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_openAfter_: int
        d_3_openAfter_ = 16
        d_4_rollbackLimit_: int
        d_4_rollbackLimit_ = 48
        d_5_armedOpen_: bool
        d_5_armedOpen_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_5_armedOpen_:
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
                            d_5_armedOpen_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_5_armedOpen_ = False
                                elif (len(generated)) >= ((len(generatedPrefix)) + (d_3_openAfter_)):
                                    d_5_armedOpen_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_10_closedGenerated_: _dafny.Seq
                        d_11_closedInside_: bool
                        d_12_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_10_closedGenerated_ = out4_
                        d_11_closedInside_ = out5_
                        d_12_closedCurrent_ = out6_
                        generated = d_10_closedGenerated_
                        insideConstrainedOut = d_11_closedInside_
                        currentConstrainedOut = d_12_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_4_rollbackLimit_):
                        d_13_rolledGenerated_: _dafny.Seq
                        d_14_rolledCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: _dafny.Seq
                        out7_, out8_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_13_rolledGenerated_ = out7_
                        d_14_rolledCurrent_ = out8_
                        generated = d_13_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_14_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_15_stablePrefix_: _dafny.Seq
                        d_15_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_16_constrainedPrompt_: _dafny.Seq
                        d_16_constrainedPrompt_ = (prompt) + (d_15_stablePrefix_)
                        d_17_validCount_: int
                        out9_: int
                        out9_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_17_validCount_ = out9_
                        if (len(currentConstrainedOut)) < (2):
                            d_18_nextEarly_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_18_nextEarly_ = out10_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_18_nextEarly_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_19_appendedGenerated_: _dafny.Seq
                                d_20_appendedInside_: bool
                                d_21_appendedCurrent_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_nextEarly_)
                                d_19_appendedGenerated_ = out11_
                                d_20_appendedInside_ = out12_
                                d_21_appendedCurrent_ = out13_
                                generated = d_19_appendedGenerated_
                                insideConstrainedOut = d_20_appendedInside_
                                currentConstrainedOut = d_21_appendedCurrent_
                        elif (d_17_validCount_) <= (d_2_narrowThreshold_):
                            d_22_nextNarrow_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                            d_22_nextNarrow_ = out14_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_nextNarrow_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_23_appendedGenerated2_: _dafny.Seq
                                d_24_appendedInside2_: bool
                                d_25_appendedCurrent2_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_nextNarrow_)
                                d_23_appendedGenerated2_ = out15_
                                d_24_appendedInside2_ = out16_
                                d_25_appendedCurrent2_ = out17_
                                generated = d_23_appendedGenerated2_
                                insideConstrainedOut = d_24_appendedInside2_
                                currentConstrainedOut = d_25_appendedCurrent2_
                        elif True:
                            d_26_remaining_: int
                            d_26_remaining_ = (maxSteps) - (d_1_steps_)
                            d_27_symbolBudget_: int
                            if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_26_remaining_)):
                                d_27_symbolBudget_ = d_26_remaining_
                            elif True:
                                d_27_symbolBudget_ = stepTokenBudget
                            d_28_symbolGenerated_: _dafny.Seq
                            d_29_symbolOut_: _dafny.Seq
                            d_30_hitEos_: bool
                            d_31_stepsUsed_: int
                            out18_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: int
                            out18_, out19_, out20_, out21_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_16_constrainedPrompt_, generated, currentConstrainedOut, d_27_symbolBudget_, eosToken)
                            d_28_symbolGenerated_ = out18_
                            d_29_symbolOut_ = out19_
                            d_30_hitEos_ = out20_
                            d_31_stepsUsed_ = out21_
                            generated = d_28_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_29_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_31_stepsUsed_)
                            if d_30_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


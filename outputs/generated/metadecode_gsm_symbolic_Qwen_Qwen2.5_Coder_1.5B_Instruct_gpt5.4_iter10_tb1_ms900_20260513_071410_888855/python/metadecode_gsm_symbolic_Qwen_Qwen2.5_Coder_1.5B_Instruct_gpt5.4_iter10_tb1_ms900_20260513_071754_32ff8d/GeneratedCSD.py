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
        d_2_openedAnySpan_: bool
        d_2_openedAnySpan_ = insideConstrained
        d_3_rollbackLimit_: int
        d_3_rollbackLimit_ = 48
        d_4_narrowThreshold_: int
        d_4_narrowThreshold_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_2_openedAnySpan_):
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
                            d_2_openedAnySpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_chunkBudget_: int
                            d_8_chunkBudget_ = (maxSteps) - (d_1_steps_)
                            d_9_freeGenerated_: _dafny.Seq
                            d_10_stoppedOpen_: bool
                            d_11_stoppedEos_: bool
                            d_12_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_9_freeGenerated_ = out3_
                            d_10_stoppedOpen_ = out4_
                            d_11_stoppedEos_ = out5_
                            d_12_stepsUsed_ = out6_
                            generated = d_9_freeGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                            if d_11_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_10_stoppedOpen_:
                                d_13_enteredGenerated_: _dafny.Seq
                                d_14_enteredInside_: bool
                                d_15_enteredCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_13_enteredGenerated_ = out7_
                                d_14_enteredInside_ = out8_
                                d_15_enteredCurrent_ = out9_
                                generated = d_13_enteredGenerated_
                                insideConstrainedOut = d_14_enteredInside_
                                currentConstrainedOut = d_15_enteredCurrent_
                                d_2_openedAnySpan_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_closedGenerated_: _dafny.Seq
                        d_17_closedInside_: bool
                        d_18_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_closedGenerated_ = out10_
                        d_17_closedInside_ = out11_
                        d_18_closedCurrent_ = out12_
                        generated = d_16_closedGenerated_
                        insideConstrainedOut = d_17_closedInside_
                        currentConstrainedOut = d_18_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_3_rollbackLimit_):
                        d_19_rolledGenerated_: _dafny.Seq
                        d_20_rolledCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_19_rolledGenerated_ = out13_
                        d_20_rolledCurrent_ = out14_
                        generated = d_19_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_20_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_21_stablePrefix_: _dafny.Seq
                        d_21_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_22_constrainedPrompt_: _dafny.Seq
                        d_22_constrainedPrompt_ = (prompt) + (d_21_stablePrefix_)
                        d_23_validCount_: int
                        out15_: int
                        out15_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_23_validCount_ = out15_
                        if (d_23_validCount_) <= (d_4_narrowThreshold_):
                            d_24_next_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.BigRational('3e0'), eosToken)
                            d_24_next_ = out16_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_24_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_25_appendedGenerated_: _dafny.Seq
                                d_26_appendedInside_: bool
                                d_27_appendedCurrent_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                d_25_appendedGenerated_ = out17_
                                d_26_appendedInside_ = out18_
                                d_27_appendedCurrent_ = out19_
                                generated = d_25_appendedGenerated_
                                insideConstrainedOut = d_26_appendedInside_
                                currentConstrainedOut = d_27_appendedCurrent_
                        elif True:
                            d_28_remaining_: int
                            d_28_remaining_ = (maxSteps) - (d_1_steps_)
                            d_29_symbolBudget_: int
                            if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_28_remaining_)):
                                d_29_symbolBudget_ = d_28_remaining_
                            elif True:
                                d_29_symbolBudget_ = stepTokenBudget
                            d_30_symbolGenerated_: _dafny.Seq
                            d_31_symbolOut_: _dafny.Seq
                            d_32_hitEos_: bool
                            d_33_stepsUsed_: int
                            out20_: _dafny.Seq
                            out21_: _dafny.Seq
                            out22_: bool
                            out23_: int
                            out20_, out21_, out22_, out23_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_22_constrainedPrompt_, generated, currentConstrainedOut, d_29_symbolBudget_, eosToken)
                            d_30_symbolGenerated_ = out20_
                            d_31_symbolOut_ = out21_
                            d_32_hitEos_ = out22_
                            d_33_stepsUsed_ = out23_
                            generated = d_30_symbolGenerated_
                            currentConstrainedOut = d_31_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_33_stepsUsed_)
                            if d_32_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
        d_2_arithmeticArmed_: bool
        d_2_arithmeticArmed_ = False
        d_3_cueTokens_: _dafny.Seq
        d_3_cueTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "difference")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "product")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "quotient")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "compute")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "calculate"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_arithmeticArmed_:
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
                            d_2_arithmeticArmed_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_remainingOutside_: int
                            d_7_remainingOutside_ = (maxSteps) - (d_1_steps_)
                            d_8_chunkBudget_: int
                            if (d_7_remainingOutside_) > (3):
                                d_8_chunkBudget_ = 3
                            elif True:
                                d_8_chunkBudget_ = d_7_remainingOutside_
                            d_9_chunkedG_: _dafny.Seq
                            d_10_stoppedOpen_: bool
                            d_11_stoppedEos_: bool
                            d_12_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_9_chunkedG_ = out3_
                            d_10_stoppedOpen_ = out4_
                            d_11_stoppedEos_ = out5_
                            d_12_stepsUsed_ = out6_
                            generated = d_9_chunkedG_
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
                                d_2_arithmeticArmed_ = False
                            elif True:
                                d_16_prevEq_: _dafny.Seq
                                d_17_foundEq_: bool
                                out10_: _dafny.Seq
                                out11_: bool
                                out10_, out11_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_16_prevEq_ = out10_
                                d_17_foundEq_ = out11_
                                if ((d_17_foundEq_) and ((d_16_prevEq_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))))) and ((d_16_prevEq_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
                                    d_2_arithmeticArmed_ = True
                                elif True:
                                    d_18_plusSince_: int
                                    out12_: int
                                    out12_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")))
                                    d_18_plusSince_ = out12_
                                    d_19_minusSince_: int
                                    out13_: int
                                    out13_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")))
                                    d_19_minusSince_ = out13_
                                    d_20_timesSince_: int
                                    out14_: int
                                    out14_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")))
                                    d_20_timesSince_ = out14_
                                    d_21_divSince_: int
                                    out15_: int
                                    out15_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")))
                                    d_21_divSince_ = out15_
                                    if ((((d_18_plusSince_) <= (2)) or ((d_19_minusSince_) <= (2))) or ((d_20_timesSince_) <= (2))) or ((d_21_divSince_) <= (2)):
                                        d_2_arithmeticArmed_ = True
                                    elif True:
                                        d_22_totalCount_: int
                                        out16_: int
                                        out16_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")))
                                        d_22_totalCount_ = out16_
                                        d_23_sumCount_: int
                                        out17_: int
                                        out17_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sum")))
                                        d_23_sumCount_ = out17_
                                        if ((d_22_totalCount_) > (0)) or ((d_23_sumCount_) > (0)):
                                            d_2_arithmeticArmed_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_24_closedGenerated_: _dafny.Seq
                        d_25_closedInside_: bool
                        d_26_closedCurrent_: _dafny.Seq
                        out18_: _dafny.Seq
                        out19_: bool
                        out20_: _dafny.Seq
                        out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_24_closedGenerated_ = out18_
                        d_25_closedInside_ = out19_
                        d_26_closedCurrent_ = out20_
                        generated = d_24_closedGenerated_
                        insideConstrainedOut = d_25_closedInside_
                        currentConstrainedOut = d_26_closedCurrent_
                        d_2_arithmeticArmed_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_27_isDeadEnd_: bool
                        out21_: bool
                        out21_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_27_isDeadEnd_ = out21_
                        if d_27_isDeadEnd_:
                            d_28_rolledGenerated_: _dafny.Seq
                            d_29_rolledCurrent_: _dafny.Seq
                            out22_: _dafny.Seq
                            out23_: _dafny.Seq
                            out22_, out23_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_28_rolledGenerated_ = out22_
                            d_29_rolledCurrent_ = out23_
                            generated = d_28_rolledGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_29_rolledCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_30_stablePrefix_: _dafny.Seq
                            d_30_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_31_constrainedPrompt_: _dafny.Seq
                            d_31_constrainedPrompt_ = (prompt) + (d_30_stablePrefix_)
                            d_32_validCount_: int
                            out24_: int
                            out24_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_32_validCount_ = out24_
                            if ((d_32_validCount_) <= (10)) or ((stepTokenBudget) <= (1)):
                                d_33_next_: _dafny.Seq
                                out25_: _dafny.Seq
                                out25_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_31_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_cueTokens_, _dafny.BigRational('15e-1'), 12, eosToken)
                                d_33_next_ = out25_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_33_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_34_appendedGenerated_: _dafny.Seq
                                    d_35_appendedInside_: bool
                                    d_36_appendedCurrent_: _dafny.Seq
                                    out26_: _dafny.Seq
                                    out27_: bool
                                    out28_: _dafny.Seq
                                    out26_, out27_, out28_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_33_next_)
                                    d_34_appendedGenerated_ = out26_
                                    d_35_appendedInside_ = out27_
                                    d_36_appendedCurrent_ = out28_
                                    generated = d_34_appendedGenerated_
                                    insideConstrainedOut = d_35_appendedInside_
                                    currentConstrainedOut = d_36_appendedCurrent_
                            elif True:
                                d_37_remainingInside_: int
                                d_37_remainingInside_ = (maxSteps) - (d_1_steps_)
                                d_38_symbolBudget_: int
                                if (stepTokenBudget) == (0):
                                    d_38_symbolBudget_ = 1
                                elif (stepTokenBudget) > (d_37_remainingInside_):
                                    d_38_symbolBudget_ = d_37_remainingInside_
                                elif True:
                                    d_38_symbolBudget_ = stepTokenBudget
                                d_39_symbolGenerated_: _dafny.Seq
                                d_40_symbolOut_: _dafny.Seq
                                d_41_hitEos_: bool
                                d_42_stepsUsed2_: int
                                out29_: _dafny.Seq
                                out30_: _dafny.Seq
                                out31_: bool
                                out32_: int
                                out29_, out30_, out31_, out32_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_31_constrainedPrompt_, generated, currentConstrainedOut, d_38_symbolBudget_, eosToken)
                                d_39_symbolGenerated_ = out29_
                                d_40_symbolOut_ = out30_
                                d_41_hitEos_ = out31_
                                d_42_stepsUsed2_ = out32_
                                generated = d_39_symbolGenerated_
                                insideConstrainedOut = True
                                currentConstrainedOut = d_40_symbolOut_
                                d_1_steps_ = (d_1_steps_) + (d_42_stepsUsed2_)
                                if d_41_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


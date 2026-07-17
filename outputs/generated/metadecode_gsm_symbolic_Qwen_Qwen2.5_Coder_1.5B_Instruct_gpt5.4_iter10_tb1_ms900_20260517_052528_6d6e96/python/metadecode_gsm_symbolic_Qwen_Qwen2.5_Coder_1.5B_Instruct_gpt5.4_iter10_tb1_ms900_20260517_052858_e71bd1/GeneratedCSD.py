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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Every arithmetic computation must appear inside visible << and >> delimiters. Keep each delimited computation short and close >> immediately after the computation.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openArmed_: bool
        d_2_openArmed_ = False
        d_3_chunkLimit_: int
        d_3_chunkLimit_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_openArmed_:
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
                            d_2_openArmed_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_remainingChunk_: int
                            d_7_remainingChunk_ = (maxSteps) - (d_1_steps_)
                            d_8_chunkBudget_: int
                            if (d_3_chunkLimit_) < (d_7_remainingChunk_):
                                d_8_chunkBudget_ = d_3_chunkLimit_
                            elif True:
                                d_8_chunkBudget_ = d_7_remainingChunk_
                            d_9_chunkedGenerated_: _dafny.Seq
                            d_10_stoppedOpen_: bool
                            d_11_stoppedEos_: bool
                            d_12_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_9_chunkedGenerated_ = out3_
                            d_10_stoppedOpen_ = out4_
                            d_11_stoppedEos_ = out5_
                            d_12_stepsUsed_ = out6_
                            generated = d_9_chunkedGenerated_
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
                                d_2_openArmed_ = False
                            elif True:
                                d_16_eqCount_: int
                                out10_: int
                                out10_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_16_eqCount_ = out10_
                                d_17_plusCount_: int
                                out11_: int
                                out11_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")))
                                d_17_plusCount_ = out11_
                                d_18_minusCount_: int
                                out12_: int
                                out12_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")))
                                d_18_minusCount_ = out12_
                                d_19_timesCount_: int
                                out13_: int
                                out13_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")))
                                d_19_timesCount_ = out13_
                                d_20_divCount_: int
                                out14_: int
                                out14_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")))
                                d_20_divCount_ = out14_
                                d_21_colonCount_: int
                                out15_: int
                                out15_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))
                                d_21_colonCount_ = out15_
                                d_22_beforeEq_: int
                                d_22_beforeEq_ = VerifiedDecoderAgent.CSDHelpers.OccurrencesInRange(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), (len(generated)) - (1))
                                d_23_beforePlus_: int
                                d_23_beforePlus_ = VerifiedDecoderAgent.CSDHelpers.OccurrencesInRange(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), (len(generated)) - (1))
                                d_24_beforeMinus_: int
                                d_24_beforeMinus_ = VerifiedDecoderAgent.CSDHelpers.OccurrencesInRange(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), (len(generated)) - (1))
                                d_25_beforeTimes_: int
                                d_25_beforeTimes_ = VerifiedDecoderAgent.CSDHelpers.OccurrencesInRange(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), (len(generated)) - (1))
                                d_26_beforeDiv_: int
                                d_26_beforeDiv_ = VerifiedDecoderAgent.CSDHelpers.OccurrencesInRange(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), (len(generated)) - (1))
                                d_27_beforeColon_: int
                                d_27_beforeColon_ = VerifiedDecoderAgent.CSDHelpers.OccurrencesInRange(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")), (len(generated)) - (1))
                                if ((((((d_16_eqCount_) > (d_22_beforeEq_)) or ((d_17_plusCount_) > (d_23_beforePlus_))) or ((d_18_minusCount_) > (d_24_beforeMinus_))) or ((d_19_timesCount_) > (d_25_beforeTimes_))) or ((d_20_divCount_) > (d_26_beforeDiv_))) or ((d_21_colonCount_) > (d_27_beforeColon_)):
                                    d_2_openArmed_ = True
                                elif True:
                                    d_2_openArmed_ = False
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_28_closedGenerated_: _dafny.Seq
                        d_29_closedInside_: bool
                        d_30_closedCurrent_: _dafny.Seq
                        out16_: _dafny.Seq
                        out17_: bool
                        out18_: _dafny.Seq
                        out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_28_closedGenerated_ = out16_
                        d_29_closedInside_ = out17_
                        d_30_closedCurrent_ = out18_
                        generated = d_28_closedGenerated_
                        insideConstrainedOut = d_29_closedInside_
                        currentConstrainedOut = d_30_closedCurrent_
                        d_2_openArmed_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_31_stablePrefix_: _dafny.Seq
                        d_31_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_32_constrainedPrompt_: _dafny.Seq
                        d_32_constrainedPrompt_ = (prompt) + (d_31_stablePrefix_)
                        d_33_remaining_: int
                        d_33_remaining_ = (maxSteps) - (d_1_steps_)
                        d_34_symbolBudget_: int
                        if (d_33_remaining_) == (0):
                            d_34_symbolBudget_ = 0
                        elif (stepTokenBudget) == (0):
                            d_34_symbolBudget_ = 1
                        elif (stepTokenBudget) < (d_33_remaining_):
                            d_34_symbolBudget_ = stepTokenBudget
                        elif True:
                            d_34_symbolBudget_ = d_33_remaining_
                        d_35_symbolGenerated_: _dafny.Seq
                        d_36_symbolOut_: _dafny.Seq
                        d_37_hitEos_: bool
                        d_38_stepsUsed_: int
                        out19_: _dafny.Seq
                        out20_: _dafny.Seq
                        out21_: bool
                        out22_: int
                        out19_, out20_, out21_, out22_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_32_constrainedPrompt_, generated, currentConstrainedOut, d_34_symbolBudget_, eosToken)
                        d_35_symbolGenerated_ = out19_
                        d_36_symbolOut_ = out20_
                        d_37_hitEos_ = out21_
                        d_38_stepsUsed_ = out22_
                        generated = d_35_symbolGenerated_
                        currentConstrainedOut = d_36_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_38_stepsUsed_)
                        if d_37_hitEos_:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


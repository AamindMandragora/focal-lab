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
        d_2_arithmeticCueSeen_: bool
        d_2_arithmeticCueSeen_ = False
        d_3_cueTokens_: _dafny.Seq
        d_3_cueTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "x")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "compute")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sum"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_arithmeticCueSeen_:
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
                            d_2_arithmeticCueSeen_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_remainingChunk_: int
                            d_7_remainingChunk_ = (maxSteps) - (d_1_steps_)
                            d_8_chunkBudget_: int
                            if (d_7_remainingChunk_) > (3):
                                d_8_chunkBudget_ = 3
                            elif True:
                                d_8_chunkBudget_ = d_7_remainingChunk_
                            d_9_chunkedGenerated_: _dafny.Seq
                            d_10_stoppedOnOpenSpan_: bool
                            d_11_stoppedOnEos_: bool
                            d_12_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_9_chunkedGenerated_ = out3_
                            d_10_stoppedOnOpenSpan_ = out4_
                            d_11_stoppedOnEos_ = out5_
                            d_12_stepsUsed_ = out6_
                            generated = d_9_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                            if d_11_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_10_stoppedOnOpenSpan_:
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
                                d_2_arithmeticCueSeen_ = False
                            elif True:
                                d_16_prevEq_: _dafny.Seq
                                d_17_foundEq_: bool
                                out10_: _dafny.Seq
                                out11_: bool
                                out10_, out11_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                                d_16_prevEq_ = out10_
                                d_17_foundEq_ = out11_
                                if d_17_foundEq_:
                                    if (((((d_16_prevEq_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))) or ((d_16_prevEq_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+"))))) or ((d_16_prevEq_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-"))))) or ((d_16_prevEq_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*"))))) or ((d_16_prevEq_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")))):
                                        d_2_arithmeticCueSeen_ = True
                                    elif True:
                                        d_2_arithmeticCueSeen_ = False
                                elif True:
                                    d_2_arithmeticCueSeen_ = False
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_18_closedGenerated_: _dafny.Seq
                        d_19_closedInside_: bool
                        d_20_closedCurrent_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_18_closedGenerated_ = out12_
                        d_19_closedInside_ = out13_
                        d_20_closedCurrent_ = out14_
                        generated = d_18_closedGenerated_
                        insideConstrainedOut = d_19_closedInside_
                        currentConstrainedOut = d_20_closedCurrent_
                        d_2_arithmeticCueSeen_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_21_stablePrefix_: _dafny.Seq
                        d_21_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_22_constrainedPrompt_: _dafny.Seq
                        d_22_constrainedPrompt_ = (prompt) + (d_21_stablePrefix_)
                        d_23_next_: _dafny.Seq
                        d_23_next_ = eosToken
                        d_24_eqCount_: int
                        out15_: int
                        out15_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                        d_24_eqCount_ = out15_
                        d_25_narrow_: bool
                        out16_: bool
                        out16_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 12)
                        d_25_narrow_ = out16_
                        if ((d_24_eqCount_) == (0)) and (not(d_25_narrow_)):
                            out17_: _dafny.Seq
                            out17_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_23_next_ = out17_
                        elif True:
                            out18_: _dafny.Seq
                            out18_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('4e0'), eosToken)
                            d_23_next_ = out18_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_23_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_26_appendedGenerated_: _dafny.Seq
                            d_27_appendedInside_: bool
                            d_28_appendedCurrent_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
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


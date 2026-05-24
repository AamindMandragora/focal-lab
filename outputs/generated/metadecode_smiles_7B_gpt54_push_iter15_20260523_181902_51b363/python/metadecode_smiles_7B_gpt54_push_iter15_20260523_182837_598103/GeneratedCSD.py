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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Produce the requested answer as a valid SMILES molecule. Use any outer continuation only to reach the molecule span naturally; once in the constrained molecule span, keep every prefix parser-valid and close the span as soon as a complete valid molecule is formed.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_openedFallback_: bool
        d_3_openedFallback_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_3_openedFallback_)) and (((len(generated)) > (len(generatedPrefix))) or ((len(generatedPrefix)) > (0))):
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
                            d_3_openedFallback_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_chunkBudget_: int
                            d_7_chunkBudget_ = (maxSteps) - (d_1_steps_)
                            d_8_chunkSize_: int
                            if (d_7_chunkBudget_) == (0):
                                d_8_chunkSize_ = 0
                            elif True:
                                d_8_chunkSize_ = 1
                            if (d_7_chunkBudget_) > (1):
                                d_8_chunkSize_ = 2
                            d_9_chunkedGenerated_: _dafny.Seq
                            d_10_stoppedOpen_: bool
                            d_11_stoppedEos_: bool
                            d_12_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
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
                        raise _dafny.Break("0")
                    elif True:
                        d_19_stablePrefix_: _dafny.Seq
                        d_19_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (d_19_stablePrefix_)
                        d_21_validCount_: int
                        out13_: int
                        out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_21_validCount_ = out13_
                        d_22_openParenCount_: int
                        out14_: int
                        out14_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")))
                        d_22_openParenCount_ = out14_
                        d_23_bondCount_: int
                        out15_: int
                        out15_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                        d_23_bondCount_ = out15_
                        d_24_next_: _dafny.Seq
                        d_24_next_ = eosToken
                        if (d_21_validCount_) <= (d_2_narrowThreshold_):
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                            d_24_next_ = out16_
                        elif ((d_22_openParenCount_) > (1)) or ((d_23_bondCount_) > (1)):
                            out17_: _dafny.Seq
                            out17_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_24_next_ = out17_
                        elif True:
                            d_25_nextSoft_: _dafny.Seq
                            d_26_usedFallback_: bool
                            out18_: _dafny.Seq
                            out19_: bool
                            out18_, out19_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                            d_25_nextSoft_ = out18_
                            d_26_usedFallback_ = out19_
                            d_24_next_ = d_25_nextSoft_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_24_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_27_appendedGenerated_: _dafny.Seq
                            d_28_appendedInside_: bool
                            d_29_appendedCurrent_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                            d_27_appendedGenerated_ = out20_
                            d_28_appendedInside_ = out21_
                            d_29_appendedCurrent_ = out22_
                            generated = d_27_appendedGenerated_
                            insideConstrainedOut = d_28_appendedInside_
                            currentConstrainedOut = d_29_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


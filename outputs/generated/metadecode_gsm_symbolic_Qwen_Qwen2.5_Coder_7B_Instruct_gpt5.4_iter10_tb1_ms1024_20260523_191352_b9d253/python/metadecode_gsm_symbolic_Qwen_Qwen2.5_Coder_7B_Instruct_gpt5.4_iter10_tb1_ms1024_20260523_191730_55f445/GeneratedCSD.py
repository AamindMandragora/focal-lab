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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write arithmetic computations inside visible << >> delimiters, but only when you reach an actual computation.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_eqToken_: _dafny.Seq
        d_2_eqToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))
        d_3_colonToken_: _dafny.Seq
        d_3_colonToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":"))
        d_4_isToken_: _dafny.Seq
        d_4_isToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "is"))
        d_5_areToken_: _dafny.Seq
        d_5_areToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "are"))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_remainingOutside_: int
                        d_6_remainingOutside_ = (maxSteps) - (d_1_steps_)
                        d_7_chunkBudget_: int
                        if (d_6_remainingOutside_) > (4):
                            d_7_chunkBudget_ = 4
                        elif True:
                            d_7_chunkBudget_ = d_6_remainingOutside_
                        d_8_chunkedGenerated_: _dafny.Seq
                        d_9_stoppedOpen_: bool
                        d_10_stoppedEos_: bool
                        d_11_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_8_chunkedGenerated_ = out0_
                        d_9_stoppedOpen_ = out1_
                        d_10_stoppedEos_ = out2_
                        d_11_stepsUsed_ = out3_
                        generated = d_8_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_11_stepsUsed_)
                        if d_10_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_9_stoppedOpen_:
                            d_12_enteredGenerated_: _dafny.Seq
                            d_13_enteredInside_: bool
                            d_14_enteredCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_12_enteredGenerated_ = out4_
                            d_13_enteredInside_ = out5_
                            d_14_enteredCurrent_ = out6_
                            generated = d_12_enteredGenerated_
                            insideConstrainedOut = d_13_enteredInside_
                            currentConstrainedOut = d_14_enteredCurrent_
                        elif (d_1_steps_) < (maxSteps):
                            d_15_openCount_: int
                            out7_: int
                            out7_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                            d_15_openCount_ = out7_
                            d_16_sinceEq_: int
                            out8_: int
                            out8_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_2_eqToken_)
                            d_16_sinceEq_ = out8_
                            d_17_sinceColon_: int
                            out9_: int
                            out9_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_3_colonToken_)
                            d_17_sinceColon_ = out9_
                            d_18_sinceIs_: int
                            out10_: int
                            out10_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_4_isToken_)
                            d_18_sinceIs_ = out10_
                            d_19_sinceAre_: int
                            out11_: int
                            out11_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_5_areToken_)
                            d_19_sinceAre_ = out11_
                            if (((d_15_openCount_) == (0)) and ((len(generated)) >= ((len(generatedPrefix)) + (4)))) and (((((d_16_sinceEq_) <= (2)) or ((d_17_sinceColon_) <= (2))) or ((d_18_sinceIs_) <= (2))) or ((d_19_sinceAre_) <= (2))):
                                d_20_openedGenerated_: _dafny.Seq
                                d_21_openedInside_: bool
                                d_22_openedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_20_openedGenerated_ = out12_
                                d_21_openedInside_ = out13_
                                d_22_openedCurrent_ = out14_
                                generated = d_20_openedGenerated_
                                insideConstrainedOut = d_21_openedInside_
                                currentConstrainedOut = d_22_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_23_closedGenerated_: _dafny.Seq
                        d_24_closedInside_: bool
                        d_25_closedCurrent_: _dafny.Seq
                        out15_: _dafny.Seq
                        out16_: bool
                        out17_: _dafny.Seq
                        out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_23_closedGenerated_ = out15_
                        d_24_closedInside_ = out16_
                        d_25_closedCurrent_ = out17_
                        generated = d_23_closedGenerated_
                        insideConstrainedOut = d_24_closedInside_
                        currentConstrainedOut = d_25_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_26_constrainedPrompt_: _dafny.Seq
                        d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_27_next_: _dafny.Seq
                        d_27_next_ = eosToken
                        if (len(currentConstrainedOut)) < (2):
                            out18_: _dafny.Seq
                            out18_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('6e0'), eosToken)
                            d_27_next_ = out18_
                        elif True:
                            out19_: _dafny.Seq
                            out19_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_27_next_ = out19_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_27_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_28_appendedGenerated_: _dafny.Seq
                            d_29_appendedInside_: bool
                            d_30_appendedCurrent_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                            d_28_appendedGenerated_ = out20_
                            d_29_appendedInside_ = out21_
                            d_30_appendedCurrent_ = out22_
                            generated = d_28_appendedGenerated_
                            insideConstrainedOut = d_29_appendedInside_
                            currentConstrainedOut = d_30_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Every arithmetic computation must appear inside visible delimiters << and >>. Open << only when writing a computation, and close >> immediately after that computation.")))
        if (maxSteps) == (0):
            if not(insideConstrainedOut):
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        elif True:
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_needOpenSoon_: bool
            d_2_needOpenSoon_ = False
            with _dafny.label("1_0"):
                while (d_1_steps_) < (maxSteps):
                    with _dafny.c_label("1_0"):
                        if (not(insideConstrainedOut)) and (d_2_needOpenSoon_):
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
                            d_2_needOpenSoon_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif not(insideConstrainedOut):
                            d_6_remainingOutside_: int
                            d_6_remainingOutside_ = (maxSteps) - (d_1_steps_)
                            d_7_chunkBudget_: int
                            if (d_6_remainingOutside_) <= (2):
                                d_7_chunkBudget_ = d_6_remainingOutside_
                            elif True:
                                d_7_chunkBudget_ = 2
                            if (d_7_chunkBudget_) == (0):
                                raise _dafny.Break("1_0")
                            d_8_chunkedG_: _dafny.Seq
                            d_9_stoppedOpen_: bool
                            d_10_stoppedEos_: bool
                            d_11_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_8_chunkedG_ = out3_
                            d_9_stoppedOpen_ = out4_
                            d_10_stoppedEos_ = out5_
                            d_11_stepsUsed_ = out6_
                            generated = d_8_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_11_stepsUsed_)
                            if d_10_stoppedEos_:
                                raise _dafny.Break("1_0")
                            elif d_9_stoppedOpen_:
                                d_12_enteredGenerated_: _dafny.Seq
                                d_13_enteredInside_: bool
                                d_14_enteredCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_12_enteredGenerated_ = out7_
                                d_13_enteredInside_ = out8_
                                d_14_enteredCurrent_ = out9_
                                generated = d_12_enteredGenerated_
                                insideConstrainedOut = d_13_enteredInside_
                                currentConstrainedOut = d_14_enteredCurrent_
                                d_2_needOpenSoon_ = False
                            elif True:
                                d_15_sinceEq_: int
                                out10_: int
                                out10_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_15_sinceEq_ = out10_
                                d_16_sincePlus_: int
                                out11_: int
                                out11_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")))
                                d_16_sincePlus_ = out11_
                                d_17_sinceMinus_: int
                                out12_: int
                                out12_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")))
                                d_17_sinceMinus_ = out12_
                                d_18_sinceTimes_: int
                                out13_: int
                                out13_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")))
                                d_18_sinceTimes_ = out13_
                                d_19_sinceDiv_: int
                                out14_: int
                                out14_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")))
                                d_19_sinceDiv_ = out14_
                                if (((((d_15_sinceEq_) <= (2)) or ((d_16_sincePlus_) <= (2))) or ((d_17_sinceMinus_) <= (2))) or ((d_18_sinceTimes_) <= (2))) or ((d_19_sinceDiv_) <= (2)):
                                    d_2_needOpenSoon_ = True
                                elif True:
                                    d_2_needOpenSoon_ = False
                        elif (parser).IsCompletePrefix(currentConstrainedOut):
                            d_20_closedGenerated_: _dafny.Seq
                            d_21_closedInside_: bool
                            d_22_closedCurrent_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_20_closedGenerated_ = out15_
                            d_21_closedInside_ = out16_
                            d_22_closedCurrent_ = out17_
                            generated = d_20_closedGenerated_
                            insideConstrainedOut = d_21_closedInside_
                            currentConstrainedOut = d_22_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_23_stablePrefix_: _dafny.Seq
                            d_23_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_24_constrainedPrompt_: _dafny.Seq
                            d_24_constrainedPrompt_ = (prompt) + (d_23_stablePrefix_)
                            d_25_next_: _dafny.Seq
                            d_25_next_ = eosToken
                            if (len(currentConstrainedOut)) < (2):
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('6e0'), eosToken)
                                d_25_next_ = out18_
                            elif True:
                                out19_: _dafny.Seq
                                out19_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('2e0'), 12, eosToken)
                                d_25_next_ = out19_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_25_next_) == (eosToken):
                                raise _dafny.Break("1_0")
                            elif True:
                                d_26_appendedGenerated_: _dafny.Seq
                                d_27_appendedInside_: bool
                                d_28_appendedCurrent_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                                d_26_appendedGenerated_ = out20_
                                d_27_appendedInside_ = out21_
                                d_28_appendedCurrent_ = out22_
                                generated = d_26_appendedGenerated_
                                insideConstrainedOut = d_27_appendedInside_
                                currentConstrainedOut = d_28_appendedCurrent_
                        pass
                pass
            if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
                d_29_closedGenerated2_: _dafny.Seq
                d_30_closedInside2_: bool
                d_31_closedCurrent2_: _dafny.Seq
                out23_: _dafny.Seq
                out24_: bool
                out25_: _dafny.Seq
                out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_29_closedGenerated2_ = out23_
                d_30_closedInside2_ = out24_
                d_31_closedCurrent2_ = out25_
                generated = d_29_closedGenerated2_
                insideConstrainedOut = d_30_closedInside2_
                currentConstrainedOut = d_31_closedCurrent2_
                d_1_steps_ = (d_1_steps_) + (1)
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


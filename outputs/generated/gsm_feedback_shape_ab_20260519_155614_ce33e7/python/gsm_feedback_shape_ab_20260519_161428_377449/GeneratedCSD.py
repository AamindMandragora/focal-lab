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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Write each arithmetic computation inside visible delimiters << and >>. Do not open << until you are writing an actual arithmetic expression, and close >> immediately after that computation.")))
        if (maxSteps) == (0):
            if not(insideConstrainedOut):
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        elif True:
            d_1_steps_: int
            d_1_steps_ = 0
            with _dafny.label("1_0"):
                while (d_1_steps_) < (maxSteps):
                    with _dafny.c_label("1_0"):
                        if not(insideConstrainedOut):
                            d_2_remainingOutside_: int
                            d_2_remainingOutside_ = (maxSteps) - (d_1_steps_)
                            d_3_chunkBudget_: int
                            if (d_2_remainingOutside_) == (0):
                                d_3_chunkBudget_ = 0
                            elif (d_2_remainingOutside_) == (1):
                                d_3_chunkBudget_ = 1
                            elif True:
                                d_3_chunkBudget_ = 2
                            if (d_3_chunkBudget_) == (0):
                                raise _dafny.Break("1_0")
                            d_4_chunkedG_: _dafny.Seq
                            d_5_stoppedOpen_: bool
                            d_6_stoppedEos_: bool
                            d_7_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_4_chunkedG_ = out0_
                            d_5_stoppedOpen_ = out1_
                            d_6_stoppedEos_ = out2_
                            d_7_stepsUsed_ = out3_
                            generated = d_4_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                            if d_6_stoppedEos_:
                                raise _dafny.Break("1_0")
                            elif d_5_stoppedOpen_:
                                d_8_enteredGenerated_: _dafny.Seq
                                d_9_enteredInside_: bool
                                d_10_enteredCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_8_enteredGenerated_ = out4_
                                d_9_enteredInside_ = out5_
                                d_10_enteredCurrent_ = out6_
                                generated = d_8_enteredGenerated_
                                insideConstrainedOut = d_9_enteredInside_
                                currentConstrainedOut = d_10_enteredCurrent_
                        elif (parser).IsCompletePrefix(currentConstrainedOut):
                            d_11_closedGenerated_: _dafny.Seq
                            d_12_closedInside_: bool
                            d_13_closedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_closedGenerated_ = out7_
                            d_12_closedInside_ = out8_
                            d_13_closedCurrent_ = out9_
                            generated = d_11_closedGenerated_
                            insideConstrainedOut = d_12_closedInside_
                            currentConstrainedOut = d_13_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_14_stablePrefix_: _dafny.Seq
                            d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_15_constrainedPrompt_: _dafny.Seq
                            d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                            d_16_next_: _dafny.Seq
                            d_16_next_ = eosToken
                            if (len(currentConstrainedOut)) < (2):
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('6e0'), eosToken)
                                d_16_next_ = out10_
                            elif True:
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('2e0'), 12, eosToken)
                                d_16_next_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_16_next_) == (eosToken):
                                raise _dafny.Break("1_0")
                            elif True:
                                d_17_appendedGenerated_: _dafny.Seq
                                d_18_appendedInside_: bool
                                d_19_appendedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                d_17_appendedGenerated_ = out12_
                                d_18_appendedInside_ = out13_
                                d_19_appendedCurrent_ = out14_
                                generated = d_17_appendedGenerated_
                                insideConstrainedOut = d_18_appendedInside_
                                currentConstrainedOut = d_19_appendedCurrent_
                        pass
                pass
            if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
                d_20_closedGenerated2_: _dafny.Seq
                d_21_closedInside2_: bool
                d_22_closedCurrent2_: _dafny.Seq
                out15_: _dafny.Seq
                out16_: bool
                out17_: _dafny.Seq
                out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_20_closedGenerated2_ = out15_
                d_21_closedInside2_ = out16_
                d_22_closedCurrent2_ = out17_
                generated = d_20_closedGenerated2_
                insideConstrainedOut = d_21_closedInside2_
                currentConstrainedOut = d_22_closedCurrent2_
                d_1_steps_ = (d_1_steps_) + (1)
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


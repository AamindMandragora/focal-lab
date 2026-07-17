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
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out3_: int
        out0_, out1_, out2_, out3_ = default__.AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken)
        generated = out0_
        insideConstrainedOut = out1_
        currentConstrainedOut = out2_
        cost = out3_
        if ((maxSteps) > (0)) and ((cost) <= (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

    @staticmethod
    def AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. At the very end, write ONLY the final symbolic answer expression inside << >> delimiters. The expression must be the COMPLETE formula using ALL relevant variable names and numbers with +, -, *, /, //, %, (, ) only. No LaTeX, no curly braces, no text inside << >>. Write a complete multi-term expression, not just one variable. Example: <<n * price + extra>> or <<n0 * (r + 1) // d>> or <<total - n1 + n2>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_freeCapSteps_: int
        d_3_freeCapSteps_ = 400
        with _dafny.label("0"):
            while (((d_2_steps_) < (maxSteps)) and ((d_2_steps_) < (d_3_freeCapSteps_))) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_4_remaining_: int
                    d_4_remaining_ = (maxSteps) - (d_2_steps_)
                    d_5_chunkSize_: int
                    d_5_chunkSize_ = 15
                    if (d_4_remaining_) < (d_5_chunkSize_):
                        d_5_chunkSize_ = d_4_remaining_
                    if (d_5_chunkSize_) == (0):
                        raise _dafny.Break("0")
                    d_6_generatedOut_: _dafny.Seq
                    d_7_stoppedOnOpenSpan_: bool
                    d_8_stoppedOnEos_: bool
                    d_9_chunkUsed_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_6_generatedOut_ = out0_
                    d_7_stoppedOnOpenSpan_ = out1_
                    d_8_stoppedOnEos_ = out2_
                    d_9_chunkUsed_ = out3_
                    generated = d_6_generatedOut_
                    d_2_steps_ = (d_2_steps_) + (d_9_chunkUsed_)
                    if d_7_stoppedOnOpenSpan_:
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        generated = out4_
                        insideConstrainedOut = out5_
                        currentConstrainedOut = out6_
                    elif d_8_stoppedOnEos_:
                        raise _dafny.Break("0")
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            generated = out7_
            insideConstrainedOut = out8_
            currentConstrainedOut = out9_
            d_2_steps_ = (d_2_steps_) + (1)
        d_10_minConstrainedSteps_: int
        d_10_minConstrainedSteps_ = 15
        d_11_constrainedStepsTaken_: int
        d_11_constrainedStepsTaken_ = 0
        with _dafny.label("1"):
            while ((insideConstrainedOut) and ((d_2_steps_) < (maxSteps))) and ((d_11_constrainedStepsTaken_) < (d_10_minConstrainedSteps_)):
                with _dafny.c_label("1"):
                    d_12_constrainedPrompt_: _dafny.Seq
                    d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_13_next_: _dafny.Seq
                    out10_: _dafny.Seq
                    out10_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('4e0'), eosToken)
                    d_13_next_ = out10_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_11_constrainedStepsTaken_ = (d_11_constrainedStepsTaken_) + (1)
                    if (d_13_next_) == (eosToken):
                        raise _dafny.Break("1")
                    elif True:
                        d_14_isComplete_: bool
                        d_14_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if not(d_14_isComplete_):
                            d_15_appendedGenerated_: _dafny.Seq
                            d_16_appendedInside_: bool
                            d_17_appendedCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                            d_15_appendedGenerated_ = out11_
                            d_16_appendedInside_ = out12_
                            d_17_appendedCurrent_ = out13_
                            generated = d_15_appendedGenerated_
                            insideConstrainedOut = d_16_appendedInside_
                            currentConstrainedOut = d_17_appendedCurrent_
                    pass
            pass
        d_18_constrainedStepCap_: int
        d_18_constrainedStepCap_ = 75
        with _dafny.label("2"):
            while ((insideConstrainedOut) and ((d_2_steps_) < (maxSteps))) and ((d_11_constrainedStepsTaken_) < (d_18_constrainedStepCap_)):
                with _dafny.c_label("2"):
                    d_19_cg_: _dafny.Seq
                    d_20_ci_: bool
                    d_21_cc_: _dafny.Seq
                    d_22_closed_: bool
                    out14_: _dafny.Seq
                    out15_: bool
                    out16_: _dafny.Seq
                    out17_: bool
                    out14_, out15_, out16_, out17_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_19_cg_ = out14_
                    d_20_ci_ = out15_
                    d_21_cc_ = out16_
                    d_22_closed_ = out17_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_11_constrainedStepsTaken_ = (d_11_constrainedStepsTaken_) + (1)
                    if d_22_closed_:
                        generated = d_19_cg_
                        insideConstrainedOut = d_20_ci_
                        currentConstrainedOut = d_21_cc_
                    elif True:
                        d_23_constrainedPrompt_: _dafny.Seq
                        d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_24_next_: _dafny.Seq
                        out18_: _dafny.Seq
                        out18_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('4e0'), eosToken)
                        d_24_next_ = out18_
                        if (d_24_next_) == (eosToken):
                            raise _dafny.Break("2")
                        elif True:
                            d_25_isComplete_: bool
                            d_25_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if not(d_25_isComplete_):
                                d_26_appendedGenerated_: _dafny.Seq
                                d_27_appendedInside_: bool
                                d_28_appendedCurrent_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                d_26_appendedGenerated_ = out19_
                                d_27_appendedInside_ = out20_
                                d_28_appendedCurrent_ = out21_
                                generated = d_26_appendedGenerated_
                                insideConstrainedOut = d_27_appendedInside_
                                currentConstrainedOut = d_28_appendedCurrent_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_29_closeBudget_: int
            d_29_closeBudget_ = (maxSteps) - (d_2_steps_)
            out22_: _dafny.Seq
            out23_: bool
            out24_: _dafny.Seq
            out22_, out23_, out24_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_29_closeBudget_)
            generated = out22_
            insideConstrainedOut = out23_
            currentConstrainedOut = out24_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


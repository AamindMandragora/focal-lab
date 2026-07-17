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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step, then write ONLY the final answer expression inside << >> at the very end. The expression inside << >> must use only variable names, numbers, +, -, *, /, //, %, (, ) - no LaTeX, no curly braces, no text. Example: <<n * price + extra>>. Keep the expression concise and correct."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_freeCapSteps_: int
        d_3_freeCapSteps_ = 300
        with _dafny.label("0"):
            while (((d_2_steps_) < (maxSteps)) and ((d_2_steps_) < (d_3_freeCapSteps_))) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_4_remaining_: int
                    d_4_remaining_ = (maxSteps) - (d_2_steps_)
                    d_5_remFree_: int
                    d_5_remFree_ = (d_3_freeCapSteps_) - (d_2_steps_)
                    d_6_chunkSize_: int
                    d_6_chunkSize_ = 30
                    if (d_4_remaining_) < (d_6_chunkSize_):
                        d_6_chunkSize_ = d_4_remaining_
                    if (d_5_remFree_) < (d_6_chunkSize_):
                        d_6_chunkSize_ = d_5_remFree_
                    if (d_6_chunkSize_) == (0):
                        raise _dafny.Break("0")
                    d_7_generatedOut_: _dafny.Seq
                    d_8_stoppedOnOpenSpan_: bool
                    d_9_stoppedOnEos_: bool
                    d_10_chunkUsed_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_7_generatedOut_ = out0_
                    d_8_stoppedOnOpenSpan_ = out1_
                    d_9_stoppedOnEos_ = out2_
                    d_10_chunkUsed_ = out3_
                    generated = d_7_generatedOut_
                    d_2_steps_ = (d_2_steps_) + (d_10_chunkUsed_)
                    if d_8_stoppedOnOpenSpan_:
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        generated = out4_
                        insideConstrainedOut = out5_
                        currentConstrainedOut = out6_
                    elif d_9_stoppedOnEos_:
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
        d_11_constrainedStepCap_: int
        d_11_constrainedStepCap_ = 80
        d_12_constrainedStepsTaken_: int
        d_12_constrainedStepsTaken_ = 0
        d_13_lastToken_: _dafny.Seq
        d_13_lastToken_ = eosToken
        d_14_repeatCount_: int
        d_14_repeatCount_ = 0
        with _dafny.label("1"):
            while ((insideConstrainedOut) and ((d_2_steps_) < (maxSteps))) and ((d_12_constrainedStepsTaken_) < (d_11_constrainedStepCap_)):
                with _dafny.c_label("1"):
                    d_15_cg_: _dafny.Seq
                    d_16_ci_: bool
                    d_17_cc_: _dafny.Seq
                    d_18_closed_: bool
                    out10_: _dafny.Seq
                    out11_: bool
                    out12_: _dafny.Seq
                    out13_: bool
                    out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_15_cg_ = out10_
                    d_16_ci_ = out11_
                    d_17_cc_ = out12_
                    d_18_closed_ = out13_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_12_constrainedStepsTaken_ = (d_12_constrainedStepsTaken_) + (1)
                    if d_18_closed_:
                        generated = d_15_cg_
                        insideConstrainedOut = d_16_ci_
                        currentConstrainedOut = d_17_cc_
                    elif True:
                        if ((d_14_repeatCount_) >= (3)) and ((len(currentConstrainedOut)) > (0)):
                            d_19_rolledGen_: _dafny.Seq
                            d_20_rolledCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: _dafny.Seq
                            out14_, out15_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_19_rolledGen_ = out14_
                            d_20_rolledCurrent_ = out15_
                            generated = d_19_rolledGen_
                            currentConstrainedOut = d_20_rolledCurrent_
                            d_13_lastToken_ = eosToken
                            d_14_repeatCount_ = 0
                        elif True:
                            d_21_constrainedPrompt_: _dafny.Seq
                            d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_22_next_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('6e0'), eosToken)
                            d_22_next_ = out16_
                            if (d_22_next_) == (eosToken):
                                raise _dafny.Break("1")
                            elif True:
                                if (d_22_next_) == (d_13_lastToken_):
                                    d_14_repeatCount_ = (d_14_repeatCount_) + (1)
                                elif True:
                                    d_13_lastToken_ = d_22_next_
                                    d_14_repeatCount_ = 0
                                d_23_appendedGenerated_: _dafny.Seq
                                d_24_appendedInside_: bool
                                d_25_appendedCurrent_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                d_23_appendedGenerated_ = out17_
                                d_24_appendedInside_ = out18_
                                d_25_appendedCurrent_ = out19_
                                generated = d_23_appendedGenerated_
                                insideConstrainedOut = d_24_appendedInside_
                                currentConstrainedOut = d_25_appendedCurrent_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_26_closeBudget_: int
            d_26_closeBudget_ = (maxSteps) - (d_2_steps_)
            out20_: _dafny.Seq
            out21_: bool
            out22_: _dafny.Seq
            out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_26_closeBudget_)
            generated = out20_
            insideConstrainedOut = out21_
            currentConstrainedOut = out22_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


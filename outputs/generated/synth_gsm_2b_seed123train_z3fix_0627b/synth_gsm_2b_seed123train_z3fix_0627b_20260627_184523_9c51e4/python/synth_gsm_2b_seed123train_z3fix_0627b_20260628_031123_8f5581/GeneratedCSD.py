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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. At the very end, place ONLY the final arithmetic expression inside << >> with no repetition. Write exactly one complete expression like <<n * price + extra>> and then stop.")))
        d_2_phase1Cap_: int
        d_2_phase1Cap_ = 500
        if (d_2_phase1Cap_) > (maxSteps):
            d_2_phase1Cap_ = maxSteps
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_2_phase1Cap_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_3_remaining_: int
                    d_3_remaining_ = (d_2_phase1Cap_) - (d_1_steps_)
                    d_4_chunkSize_: int
                    d_4_chunkSize_ = d_3_remaining_
                    if (d_4_chunkSize_) > (60):
                        d_4_chunkSize_ = 60
                    if (d_4_chunkSize_) == (0):
                        raise _dafny.Break("0")
                    d_5_genOut_: _dafny.Seq
                    d_6_stoppedOnOpen_: bool
                    d_7_stoppedOnEos_: bool
                    d_8_stepsUsed_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_5_genOut_ = out0_
                    d_6_stoppedOnOpen_ = out1_
                    d_7_stoppedOnEos_ = out2_
                    d_8_stepsUsed_ = out3_
                    generated = d_5_genOut_
                    d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                    if d_7_stoppedOnEos_:
                        raise _dafny.Break("0")
                    elif d_6_stoppedOnOpen_:
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        generated = out4_
                        insideConstrainedOut = out5_
                        currentConstrainedOut = out6_
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            generated = out7_
            insideConstrainedOut = out8_
            currentConstrainedOut = out9_
            d_1_steps_ = (d_1_steps_) + (1)
        d_9_minSpanTokens_: int
        d_9_minSpanTokens_ = 5
        d_10_reserveForClose_: int
        d_10_reserveForClose_ = 10
        d_11_phase2Budget_: int
        d_11_phase2Budget_ = 0
        if (d_1_steps_) < (maxSteps):
            d_12_rem_: int
            d_12_rem_ = (maxSteps) - (d_1_steps_)
            if (d_12_rem_) > (d_10_reserveForClose_):
                d_11_phase2Budget_ = (d_12_rem_) - (d_10_reserveForClose_)
            elif True:
                d_11_phase2Budget_ = d_12_rem_
        if (d_11_phase2Budget_) > (200):
            d_11_phase2Budget_ = 200
        d_13_phase2Steps_: int
        d_13_phase2Steps_ = 0
        with _dafny.label("1"):
            while ((insideConstrainedOut) and ((d_13_phase2Steps_) < (d_11_phase2Budget_))) and ((d_1_steps_) < (maxSteps)):
                with _dafny.c_label("1"):
                    if (len(currentConstrainedOut)) >= (d_9_minSpanTokens_):
                        d_14_cg_: _dafny.Seq
                        d_15_ci_: bool
                        d_16_cc_: _dafny.Seq
                        d_17_closed_: bool
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out13_: bool
                        out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_14_cg_ = out10_
                        d_15_ci_ = out11_
                        d_16_cc_ = out12_
                        d_17_closed_ = out13_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_13_phase2Steps_ = (d_13_phase2Steps_) + (1)
                        if d_17_closed_:
                            generated = d_14_cg_
                            insideConstrainedOut = d_15_ci_
                            currentConstrainedOut = d_16_cc_
                    elif True:
                        d_18_isComplete_: bool
                        d_18_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if not(d_18_isComplete_):
                            d_19_constrainedPrompt_: _dafny.Seq
                            d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_20_next_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_20_next_ = out14_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_13_phase2Steps_ = (d_13_phase2Steps_) + (1)
                            if (d_20_next_) == (eosToken):
                                raise _dafny.Break("1")
                            elif True:
                                d_21_ag_: _dafny.Seq
                                d_22_ai_: bool
                                d_23_ac_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                d_21_ag_ = out15_
                                d_22_ai_ = out16_
                                d_23_ac_ = out17_
                                generated = d_21_ag_
                                insideConstrainedOut = d_22_ai_
                                currentConstrainedOut = d_23_ac_
                        elif True:
                            d_24_cg2_: _dafny.Seq
                            d_25_ci2_: bool
                            d_26_cc2_: _dafny.Seq
                            d_27_closed2_: bool
                            out18_: _dafny.Seq
                            out19_: bool
                            out20_: _dafny.Seq
                            out21_: bool
                            out18_, out19_, out20_, out21_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_24_cg2_ = out18_
                            d_25_ci2_ = out19_
                            d_26_cc2_ = out20_
                            d_27_closed2_ = out21_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_13_phase2Steps_ = (d_13_phase2Steps_) + (1)
                            if d_27_closed2_:
                                generated = d_24_cg2_
                                insideConstrainedOut = d_25_ci2_
                                currentConstrainedOut = d_26_cc2_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_28_closeBudget_: int
            d_28_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_29_cg3_: _dafny.Seq
            d_30_ci3_: bool
            d_31_cc3_: _dafny.Seq
            out22_: _dafny.Seq
            out23_: bool
            out24_: _dafny.Seq
            out22_, out23_, out24_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_28_closeBudget_)
            d_29_cg3_ = out22_
            d_30_ci3_ = out23_
            d_31_cc3_ = out24_
            generated = d_29_cg3_
            insideConstrainedOut = d_30_ci3_
            currentConstrainedOut = d_31_cc3_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


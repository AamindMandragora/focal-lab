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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the FINAL answer as a single arithmetic expression using the template variable names (without braces). Wrap it in << >>. Write ONLY the expression inside << >> — no text, no explanation. Close >> immediately when the expression is complete. Examples: <<total - n1 - n2>>, <<n * (mult + 1)>>, <<n * p1 / 100 * p2 / 100 * frac>>.")))
        d_2_phase1Cap_: int
        d_2_phase1Cap_ = 300
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
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_9_phase2Budget_: int
            d_9_phase2Budget_ = (maxSteps) - (d_1_steps_)
            d_10_reserveForClose_: int
            d_10_reserveForClose_ = 10
            d_11_innerBudget_: int
            d_11_innerBudget_ = 0
            if (d_9_phase2Budget_) > (d_10_reserveForClose_):
                d_11_innerBudget_ = (d_9_phase2Budget_) - (d_10_reserveForClose_)
            if (d_11_innerBudget_) > (100):
                d_11_innerBudget_ = 100
            d_12_innerSteps_: int
            d_12_innerSteps_ = 0
            with _dafny.label("3_0"):
                while (((d_12_innerSteps_) < (d_11_innerBudget_)) and ((d_1_steps_) < (maxSteps))) and (insideConstrainedOut):
                    with _dafny.c_label("3_0"):
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_13_cg2_: _dafny.Seq
                            d_14_ci2_: bool
                            d_15_cc2_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_13_cg2_ = out10_
                            d_14_ci2_ = out11_
                            d_15_cc2_ = out12_
                            generated = d_13_cg2_
                            insideConstrainedOut = d_14_ci2_
                            currentConstrainedOut = d_15_cc2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_12_innerSteps_ = (d_12_innerSteps_) + (1)
                            raise _dafny.Break("3_0")
                        elif True:
                            d_16_stableLen_: int
                            d_16_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
                            d_17_constrainedPrompt_: _dafny.Seq
                            d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:d_16_stableLen_:]))
                            d_18_next_: _dafny.Seq
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 15, eosToken)
                            d_18_next_ = out13_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_12_innerSteps_ = (d_12_innerSteps_) + (1)
                            if (d_18_next_) == (eosToken):
                                raise _dafny.Break("3_0")
                            elif True:
                                if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                    d_19_ag_: _dafny.Seq
                                    d_20_ai_: bool
                                    d_21_ac_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                    d_19_ag_ = out14_
                                    d_20_ai_ = out15_
                                    d_21_ac_ = out16_
                                    generated = d_19_ag_
                                    insideConstrainedOut = d_20_ai_
                                    currentConstrainedOut = d_21_ac_
                        pass
                pass
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_22_closeBudget_: int
                d_22_closeBudget_ = (maxSteps) - (d_1_steps_)
                d_23_cg_: _dafny.Seq
                d_24_ci_: bool
                d_25_cc_: _dafny.Seq
                out17_: _dafny.Seq
                out18_: bool
                out19_: _dafny.Seq
                out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_closeBudget_)
                d_23_cg_ = out17_
                d_24_ci_ = out18_
                d_25_cc_ = out19_
                generated = d_23_cg_
                insideConstrainedOut = d_24_ci_
                currentConstrainedOut = d_25_cc_
                d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


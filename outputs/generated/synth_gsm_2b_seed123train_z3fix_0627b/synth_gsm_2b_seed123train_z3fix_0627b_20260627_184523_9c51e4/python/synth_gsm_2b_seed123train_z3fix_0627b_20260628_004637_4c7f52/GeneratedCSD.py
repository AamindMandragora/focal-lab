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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Use the exact variable names from the problem (without curly braces, e.g. n1, price, frac). At the very end write the complete arithmetic expression inside << >>. Use only the variable names, operators (+, -, *, /), numbers, and parentheses. Example: <<n * p / 100 + extra_price>> or <<count * (n1 + n2 + n3)>>.")))
        d_2_phase1Cap_: int
        d_2_phase1Cap_ = 380
        if (d_2_phase1Cap_) > (maxSteps):
            d_2_phase1Cap_ = maxSteps
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_2_phase1Cap_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_3_remaining_: int
                    d_3_remaining_ = (d_2_phase1Cap_) - (d_1_steps_)
                    d_4_chunkSize_: int
                    d_4_chunkSize_ = d_3_remaining_
                    if (d_4_chunkSize_) > (80):
                        d_4_chunkSize_ = 80
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
            d_10_reserveForClose_ = 25
            d_11_innerBudget_: int
            d_11_innerBudget_ = 0
            if (d_9_phase2Budget_) > (d_10_reserveForClose_):
                d_11_innerBudget_ = (d_9_phase2Budget_) - (d_10_reserveForClose_)
            if (d_11_innerBudget_) > (150):
                d_11_innerBudget_ = 150
            d_12_minSpanTokens_: int
            d_12_minSpanTokens_ = 8
            d_13_innerSteps_: int
            d_13_innerSteps_ = 0
            with _dafny.label("3_0"):
                while (((d_13_innerSteps_) < (d_11_innerBudget_)) and ((d_1_steps_) < (maxSteps))) and (insideConstrainedOut):
                    with _dafny.c_label("3_0"):
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_12_minSpanTokens_)):
                            d_14_cg2_: _dafny.Seq
                            d_15_ci2_: bool
                            d_16_cc2_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_14_cg2_ = out10_
                            d_15_ci2_ = out11_
                            d_16_cc2_ = out12_
                            generated = d_14_cg2_
                            insideConstrainedOut = d_15_ci2_
                            currentConstrainedOut = d_16_cc2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_13_innerSteps_ = (d_13_innerSteps_) + (1)
                            raise _dafny.Break("3_0")
                        elif True:
                            d_17_stableLen_: int
                            d_17_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
                            d_18_constrainedPrompt_: _dafny.Seq
                            d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:d_17_stableLen_:]))
                            d_19_next_: _dafny.Seq
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_19_next_ = out13_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_13_innerSteps_ = (d_13_innerSteps_) + (1)
                            if (d_19_next_) == (eosToken):
                                raise _dafny.Break("3_0")
                            elif True:
                                if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                    d_20_ag_: _dafny.Seq
                                    d_21_ai_: bool
                                    d_22_ac_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                    d_20_ag_ = out14_
                                    d_21_ai_ = out15_
                                    d_22_ac_ = out16_
                                    generated = d_20_ag_
                                    insideConstrainedOut = d_21_ai_
                                    currentConstrainedOut = d_22_ac_
                        pass
                pass
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_23_closeBudget_: int
                d_23_closeBudget_ = (maxSteps) - (d_1_steps_)
                d_24_cg_: _dafny.Seq
                d_25_ci_: bool
                d_26_cc_: _dafny.Seq
                out17_: _dafny.Seq
                out18_: bool
                out19_: _dafny.Seq
                out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_closeBudget_)
                d_24_cg_ = out17_
                d_25_ci_ = out18_
                d_26_cc_ = out19_
                generated = d_24_cg_
                insideConstrainedOut = d_25_ci_
                currentConstrainedOut = d_26_cc_
                d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


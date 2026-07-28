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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step using the template variable names (without curly braces). At the very end, write <<EXPR>> where EXPR is the complete arithmetic expression (e.g. <<count * (n1 + n2 + n3)>> or <<total - fraction * total - current>>). The expression must use operators like +, -, *, / and include all relevant variables.")))
        d_2_phase1Cap_: int
        d_2_phase1Cap_ = 350
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
            d_9_stableLen_: int
            d_9_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
            d_10_stablePrefix_: _dafny.Seq
            d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:d_9_stableLen_:])
            d_11_exprBudget_: int
            d_11_exprBudget_ = (maxSteps) - (d_1_steps_)
            d_12_reserveForClose_: int
            d_12_reserveForClose_ = 5
            if (d_11_exprBudget_) > ((80) + (d_12_reserveForClose_)):
                d_11_exprBudget_ = 80
            elif (d_11_exprBudget_) > (d_12_reserveForClose_):
                d_11_exprBudget_ = (d_11_exprBudget_) - (d_12_reserveForClose_)
            elif True:
                d_11_exprBudget_ = 0
            if (d_11_exprBudget_) >= (1):
                d_13_constrainedPrompt_: _dafny.Seq
                d_13_constrainedPrompt_ = (prompt) + (d_10_stablePrefix_)
                d_14_constrainedResult_: _dafny.Seq
                d_15_terminatedByEos_: bool
                out10_: _dafny.Seq
                out11_: bool
                out10_, out11_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, d_13_constrainedPrompt_, d_11_exprBudget_, eosToken)
                d_14_constrainedResult_ = out10_
                d_15_terminatedByEos_ = out11_
                generated = (d_10_stablePrefix_) + (d_14_constrainedResult_)
                currentConstrainedOut = d_14_constrainedResult_
                d_1_steps_ = (d_1_steps_) + (d_11_exprBudget_)
                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                    d_16_cg_: _dafny.Seq
                    d_17_ci_: bool
                    d_18_cc_: _dafny.Seq
                    out12_: _dafny.Seq
                    out13_: bool
                    out14_: _dafny.Seq
                    out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_16_cg_ = out12_
                    d_17_ci_ = out13_
                    d_18_cc_ = out14_
                    generated = d_16_cg_
                    insideConstrainedOut = d_17_ci_
                    currentConstrainedOut = d_18_cc_
                    d_1_steps_ = (d_1_steps_) + (1)
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_19_closeBudget_: int
                d_19_closeBudget_ = (maxSteps) - (d_1_steps_)
                d_20_cg_: _dafny.Seq
                d_21_ci_: bool
                d_22_cc_: _dafny.Seq
                out15_: _dafny.Seq
                out16_: bool
                out17_: _dafny.Seq
                out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_closeBudget_)
                d_20_cg_ = out15_
                d_21_ci_ = out16_
                d_22_cc_ = out17_
                generated = d_20_cg_
                insideConstrainedOut = d_21_ci_
                currentConstrainedOut = d_22_cc_
                d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


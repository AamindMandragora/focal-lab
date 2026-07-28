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
        d_2_spanCount_: int
        d_2_spanCount_ = 0
        d_3_maxCandidateSpans_: int
        d_3_maxCandidateSpans_ = 6
        d_4_closeBudgetPerSpan_: int
        d_4_closeBudgetPerSpan_ = 48
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Create up to six independent arithmetic candidate expressions. Use route labels A through F in plain text, and put each candidate expression itself inside its own visible constrained span like Candidate A: <<expr>>. The text outside spans can explain the route briefly, but inside each << >> span use only variables, numbers, parentheses, and arithmetic operators. No words, units, LaTeX, equals signs with prose, or repeated junk inside spans. After the candidate spans, choose the simplest candidate supported by the routes and put the final chosen expression in one last << >> span.")))
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and ((d_2_spanCount_) < (d_3_maxCandidateSpans_)):
                with _dafny.c_label("0"):
                    d_5_remaining_: int
                    d_5_remaining_ = (maxSteps) - (d_1_steps_)
                    d_6_chunkSize_: int
                    d_6_chunkSize_ = d_5_remaining_
                    if (d_6_chunkSize_) > (64):
                        d_6_chunkSize_ = 64
                    if (d_6_chunkSize_) == (0):
                        raise _dafny.Break("0")
                    d_7_genOut_: _dafny.Seq
                    d_8_stoppedOnOpen_: bool
                    d_9_stoppedOnEos_: bool
                    d_10_stepsUsed_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_7_genOut_ = out0_
                    d_8_stoppedOnOpen_ = out1_
                    d_9_stoppedOnEos_ = out2_
                    d_10_stepsUsed_ = out3_
                    generated = d_7_genOut_
                    d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                    if (d_8_stoppedOnOpen_) and ((d_1_steps_) < (maxSteps)):
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        generated = out4_
                        insideConstrainedOut = out5_
                        currentConstrainedOut = out6_
                        d_11_closeBudget_: int
                        d_11_closeBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_11_closeBudget_) > (d_4_closeBudgetPerSpan_):
                            d_11_closeBudget_ = d_4_closeBudgetPerSpan_
                        d_12_cg_: _dafny.Seq
                        d_13_ci_: bool
                        d_14_cc_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_closeBudget_)
                        d_12_cg_ = out7_
                        d_13_ci_ = out8_
                        d_14_cc_ = out9_
                        generated = d_12_cg_
                        insideConstrainedOut = d_13_ci_
                        currentConstrainedOut = d_14_cc_
                        d_1_steps_ = (d_1_steps_) + (d_11_closeBudget_)
                        d_2_spanCount_ = (d_2_spanCount_) + (1)
                    elif d_9_stoppedOnEos_:
                        raise _dafny.Break("0")
                    pass
            pass
        if ((not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps))) and ((d_2_spanCount_) == (0)):
            out10_: _dafny.Seq
            out11_: bool
            out12_: _dafny.Seq
            out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            generated = out10_
            insideConstrainedOut = out11_
            currentConstrainedOut = out12_
            d_1_steps_ = (d_1_steps_) + (1)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_15_finalBudget_: int
            d_15_finalBudget_ = (maxSteps) - (d_1_steps_)
            if (d_15_finalBudget_) > (d_4_closeBudgetPerSpan_):
                d_15_finalBudget_ = d_4_closeBudgetPerSpan_
            d_16_fg_: _dafny.Seq
            d_17_fi_: bool
            d_18_fc_: _dafny.Seq
            out13_: _dafny.Seq
            out14_: bool
            out15_: _dafny.Seq
            out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_finalBudget_)
            d_16_fg_ = out13_
            d_17_fi_ = out14_
            d_18_fc_ = out15_
            generated = d_16_fg_
            insideConstrainedOut = d_17_fi_
            currentConstrainedOut = d_18_fc_
            d_1_steps_ = (d_1_steps_) + (d_15_finalBudget_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


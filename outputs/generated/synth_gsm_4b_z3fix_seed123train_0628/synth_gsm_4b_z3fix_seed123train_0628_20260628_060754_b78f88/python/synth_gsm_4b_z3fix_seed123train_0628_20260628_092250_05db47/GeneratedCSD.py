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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Write the final answer as a single arithmetic expression inside << >>. Use variable names from the problem (like n1, n2, k1, etc.). Use only +, -, *, /, //, % operators and parentheses. Do not use ** or {curly braces}. Keep the expression concise and correct."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_maxFreeSteps_: int
        d_3_maxFreeSteps_ = 200
        d_4_spanOpened_: bool
        d_4_spanOpened_ = False
        with _dafny.label("0"):
            while ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_5_remaining_: int
                    d_5_remaining_ = (maxSteps) - (d_2_steps_)
                    if (d_5_remaining_) < (5):
                        raise _dafny.Break("0")
                    if ((d_2_steps_) >= (d_3_maxFreeSteps_)) and (not(d_4_spanOpened_)):
                        if (d_5_remaining_) >= (5):
                            d_6_og_: _dafny.Seq
                            d_7_oi_: bool
                            d_8_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_og_ = out0_
                            d_7_oi_ = out1_
                            d_8_oc_ = out2_
                            generated = d_6_og_
                            insideConstrainedOut = d_7_oi_
                            currentConstrainedOut = d_8_oc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_4_spanOpened_ = True
                        raise _dafny.Break("0")
                    d_9_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_9_next_ = out3_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_9_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                    if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_4_spanOpened_ = True
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_10_remaining2_: int
            d_10_remaining2_ = (maxSteps) - (d_2_steps_)
            d_11_closeReserve_: int
            if (d_10_remaining2_) >= (20):
                d_11_closeReserve_ = 10
            elif True:
                d_11_closeReserve_ = _dafny.euclidian_division(d_10_remaining2_, 2)
            d_12_fillBudget_: int
            if (d_10_remaining2_) > (d_11_closeReserve_):
                d_12_fillBudget_ = (d_10_remaining2_) - (d_11_closeReserve_)
            elif True:
                d_12_fillBudget_ = 0
            if (d_12_fillBudget_) >= (2):
                d_13_penaltyTokens_: _dafny.Seq
                out4_: _dafny.Seq
                out4_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                d_13_penaltyTokens_ = out4_
                d_14_stable_: _dafny.Seq
                d_14_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_15_constrainedPrompt_: _dafny.Seq
                d_15_constrainedPrompt_ = (prompt) + (d_14_stable_)
                d_16_rolloutGen_: _dafny.Seq
                d_17_rolloutSteps_: int
                d_18_rolloutEos_: bool
                out5_: _dafny.Seq
                out6_: int
                out7_: bool
                out5_, out6_, out7_ = (d_0_helpers_).RolloutConstrainedWithPenalties(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, d_12_fillBudget_, d_13_penaltyTokens_, _dafny.BigRational('3e0'), eosToken)
                d_16_rolloutGen_ = out5_
                d_17_rolloutSteps_ = out6_
                d_18_rolloutEos_ = out7_
                generated = (d_14_stable_) + (d_16_rolloutGen_)
                currentConstrainedOut = d_16_rolloutGen_
                d_2_steps_ = (d_2_steps_) + (d_17_rolloutSteps_)
                if d_18_rolloutEos_:
                    d_19_remaining3_: int
                    d_19_remaining3_ = (maxSteps) - (d_2_steps_)
                    if (d_19_remaining3_) >= (1):
                        d_20_cg_: _dafny.Seq
                        d_21_ci_: bool
                        d_22_cc_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_remaining3_)
                        d_20_cg_ = out8_
                        d_21_ci_ = out9_
                        d_22_cc_ = out10_
                        generated = d_20_cg_
                        insideConstrainedOut = d_21_ci_
                        currentConstrainedOut = d_22_cc_
                        d_2_steps_ = maxSteps
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_23_closeBudget2_: int
                d_23_closeBudget2_ = (maxSteps) - (d_2_steps_)
                d_24_cg2_: _dafny.Seq
                d_25_ci2_: bool
                d_26_cc2_: _dafny.Seq
                out11_: _dafny.Seq
                out12_: bool
                out13_: _dafny.Seq
                out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_closeBudget2_)
                d_24_cg2_ = out11_
                d_25_ci2_ = out12_
                d_26_cc2_ = out13_
                generated = d_24_cg2_
                insideConstrainedOut = d_25_ci2_
                currentConstrainedOut = d_26_cc2_
                d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


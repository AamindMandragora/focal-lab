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
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_prefixBudget_: int
            d_2_prefixBudget_ = 3
            d_3_hitEos_: bool
            d_3_hitEos_ = False
            while ((((d_1_steps_) < (d_2_prefixBudget_)) and ((d_1_steps_) < (maxSteps))) and (not(insideConstrainedOut))) and (not(d_3_hitEos_)):
                d_4_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_4_next_ = out0_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_4_next_) == (eosToken):
                    d_3_hitEos_ = True
                elif True:
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                    if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            if ((not(insideConstrainedOut)) and (not(d_3_hitEos_))) and ((d_1_steps_) < (maxSteps)):
                d_5_og_: _dafny.Seq
                d_6_oi_: bool
                d_7_oc_: _dafny.Seq
                out1_: _dafny.Seq
                out2_: bool
                out3_: _dafny.Seq
                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_5_og_ = out1_
                d_6_oi_ = out2_
                d_7_oc_ = out3_
                generated = d_5_og_
                insideConstrainedOut = d_6_oi_
                currentConstrainedOut = d_7_oc_
                d_1_steps_ = (d_1_steps_) + (1)
            d_8_constrainedTokenCount_: int
            d_8_constrainedTokenCount_ = 0
            d_9_minConstrainedTokens_: int
            d_9_minConstrainedTokens_ = 50
            d_10_earlyPhaseTokens_: int
            d_10_earlyPhaseTokens_ = 20
            d_11_midPhaseTokens_: int
            d_11_midPhaseTokens_ = 60
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                d_12_remainingBudget_: int
                d_12_remainingBudget_ = (maxSteps) - (d_1_steps_)
                d_13_allowClose_: bool
                d_13_allowClose_ = ((d_8_constrainedTokenCount_) >= (d_9_minConstrainedTokens_)) or ((d_12_remainingBudget_) <= (5))
                if (d_13_allowClose_) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                    d_14_cg_: _dafny.Seq
                    d_15_ci_: bool
                    d_16_cc_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_14_cg_ = out4_
                    d_15_ci_ = out5_
                    d_16_cc_ = out6_
                    generated = d_14_cg_
                    insideConstrainedOut = d_15_ci_
                    currentConstrainedOut = d_16_cc_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_17_constrainedPrompt_: _dafny.Seq
                    d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_18_next_: _dafny.Seq
                    d_18_next_ = eosToken
                    if (d_8_constrainedTokenCount_) < (d_10_earlyPhaseTokens_):
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('7e0'), eosToken)
                        d_18_next_ = out7_
                    elif (d_8_constrainedTokenCount_) < (d_11_midPhaseTokens_):
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), 25, eosToken)
                        d_18_next_ = out8_
                    elif True:
                        out9_: _dafny.Seq
                        out9_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_18_next_ = out9_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_18_next_) == (eosToken):
                        d_19_closeBudget_: int
                        d_19_closeBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_19_closeBudget_) > (0):
                            d_20_cg_: _dafny.Seq
                            d_21_ci_: bool
                            d_22_cc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_closeBudget_)
                            d_20_cg_ = out10_
                            d_21_ci_ = out11_
                            d_22_cc_ = out12_
                            generated = d_20_cg_
                            insideConstrainedOut = d_21_ci_
                            currentConstrainedOut = d_22_cc_
                            d_1_steps_ = maxSteps
                    elif True:
                        d_23_ag_: _dafny.Seq
                        d_24_ai_: bool
                        d_25_ac_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                        d_23_ag_ = out13_
                        d_24_ai_ = out14_
                        d_25_ac_ = out15_
                        generated = d_23_ag_
                        insideConstrainedOut = d_24_ai_
                        currentConstrainedOut = d_25_ac_
                        d_8_constrainedTokenCount_ = (d_8_constrainedTokenCount_) + (1)
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_26_closeBudget_: int
                d_26_closeBudget_ = (maxSteps) - (d_1_steps_)
                d_27_cg_: _dafny.Seq
                d_28_ci_: bool
                d_29_cc_: _dafny.Seq
                out16_: _dafny.Seq
                out17_: bool
                out18_: _dafny.Seq
                out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_26_closeBudget_)
                d_27_cg_ = out16_
                d_28_ci_ = out17_
                d_29_cc_ = out18_
                generated = d_27_cg_
                insideConstrainedOut = d_28_ci_
                currentConstrainedOut = d_29_cc_
                d_1_steps_ = maxSteps
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


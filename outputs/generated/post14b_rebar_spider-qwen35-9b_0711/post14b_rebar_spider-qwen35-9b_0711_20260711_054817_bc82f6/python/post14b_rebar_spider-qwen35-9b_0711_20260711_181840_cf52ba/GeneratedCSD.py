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
            if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_2_og_: _dafny.Seq
                d_3_oi_: bool
                d_4_oc_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_2_og_ = out0_
                d_3_oi_ = out1_
                d_4_oc_ = out2_
                generated = d_2_og_
                insideConstrainedOut = d_3_oi_
                currentConstrainedOut = d_4_oc_
                d_1_steps_ = (d_1_steps_) + (1)
            d_5_constrainedTokenCount_: int
            d_5_constrainedTokenCount_ = 0
            d_6_minConstrainedTokens_: int
            d_6_minConstrainedTokens_ = 8
            d_7_earlyPhaseTokens_: int
            d_7_earlyPhaseTokens_ = 60
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                d_8_remainingBudget_: int
                d_8_remainingBudget_ = (maxSteps) - (d_1_steps_)
                d_9_allowClose_: bool
                d_9_allowClose_ = ((d_5_constrainedTokenCount_) >= (d_6_minConstrainedTokens_)) or ((d_8_remainingBudget_) <= (5))
                if (d_9_allowClose_) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                    d_10_cg_: _dafny.Seq
                    d_11_ci_: bool
                    d_12_cc_: _dafny.Seq
                    out3_: _dafny.Seq
                    out4_: bool
                    out5_: _dafny.Seq
                    out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_10_cg_ = out3_
                    d_11_ci_ = out4_
                    d_12_cc_ = out5_
                    generated = d_10_cg_
                    insideConstrainedOut = d_11_ci_
                    currentConstrainedOut = d_12_cc_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_13_constrainedPrompt_: _dafny.Seq
                    d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_14_next_: _dafny.Seq
                    d_14_next_ = eosToken
                    if (d_5_constrainedTokenCount_) < (d_7_earlyPhaseTokens_):
                        out6_: _dafny.Seq
                        out6_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), eosToken)
                        d_14_next_ = out6_
                    elif True:
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), 200, eosToken)
                        d_14_next_ = out7_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_14_next_) == (eosToken):
                        d_15_closeBudget_: int
                        d_15_closeBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_15_closeBudget_) > (0):
                            d_16_actualBudget_: int
                            if (d_15_closeBudget_) > (50):
                                d_16_actualBudget_ = 50
                            elif True:
                                d_16_actualBudget_ = d_15_closeBudget_
                            d_17_cg_: _dafny.Seq
                            d_18_ci_: bool
                            d_19_cc_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_16_actualBudget_)
                            d_17_cg_ = out8_
                            d_18_ci_ = out9_
                            d_19_cc_ = out10_
                            generated = d_17_cg_
                            insideConstrainedOut = d_18_ci_
                            currentConstrainedOut = d_19_cc_
                            d_1_steps_ = (d_1_steps_) + (d_16_actualBudget_)
                    elif True:
                        d_20_ag_: _dafny.Seq
                        d_21_ai_: bool
                        d_22_ac_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                        d_20_ag_ = out11_
                        d_21_ai_ = out12_
                        d_22_ac_ = out13_
                        generated = d_20_ag_
                        insideConstrainedOut = d_21_ai_
                        currentConstrainedOut = d_22_ac_
                        d_5_constrainedTokenCount_ = (d_5_constrainedTokenCount_) + (1)
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_23_closeBudget_: int
                d_23_closeBudget_ = (maxSteps) - (d_1_steps_)
                d_24_cg_: _dafny.Seq
                d_25_ci_: bool
                d_26_cc_: _dafny.Seq
                out14_: _dafny.Seq
                out15_: bool
                out16_: _dafny.Seq
                out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_closeBudget_)
                d_24_cg_ = out14_
                d_25_ci_ = out15_
                d_26_cc_ = out16_
                generated = d_24_cg_
                insideConstrainedOut = d_25_ci_
                currentConstrainedOut = d_26_cc_
                d_1_steps_ = maxSteps
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


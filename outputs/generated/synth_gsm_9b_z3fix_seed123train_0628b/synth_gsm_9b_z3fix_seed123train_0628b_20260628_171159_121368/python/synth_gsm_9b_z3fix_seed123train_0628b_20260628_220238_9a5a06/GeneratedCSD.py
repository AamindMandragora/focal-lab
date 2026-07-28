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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. Show all reasoning. At the very end, write the COMPLETE final symbolic expression inside << >>. Use ALL relevant variable names combined with operators. Operators allowed: +, -, *, /, //, %, (, ), int(). Use int() for integer results. NEVER use { } braces inside << >>. NEVER use ** for exponentiation. Examples: <<n1 * p1 + n2 * p2 + n3 * p3>> or <<int(n * p * (100 + r1) * (100 - r2) / 10000)>> or <<int((length / (plant_width + space)) * cost)>>. The expression must be COMPLETE, not just a single variable.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanEverOpened_: bool
        d_2_spanEverOpened_ = insideConstrained
        d_3_spanTokens_: int
        d_3_spanTokens_ = 0
        d_4_reserved_: int = int(0)
        d_5_fracRes_: int
        d_5_fracRes_ = _dafny.euclidian_division((maxSteps) * (15), 100)
        if (d_5_fracRes_) >= (80):
            d_4_reserved_ = d_5_fracRes_
        elif True:
            d_4_reserved_ = 80
        if (d_4_reserved_) >= (maxSteps):
            if (maxSteps) >= (2):
                d_4_reserved_ = _dafny.euclidian_division(maxSteps, 2)
            elif True:
                d_4_reserved_ = 0
        d_6_forceOpenAt_: int
        d_6_forceOpenAt_ = (maxSteps) - (d_4_reserved_)
        d_7_minSpanTokens_: int
        d_7_minSpanTokens_ = 3
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_2_spanEverOpened_)) and ((d_1_steps_) >= (d_6_forceOpenAt_)):
                            d_8_og_: _dafny.Seq
                            d_9_oi_: bool
                            d_10_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_og_ = out0_
                            d_9_oi_ = out1_
                            d_10_oc_ = out2_
                            generated = d_8_og_
                            insideConstrainedOut = d_9_oi_
                            currentConstrainedOut = d_10_oc_
                            d_2_spanEverOpened_ = True
                            d_3_spanTokens_ = 0
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                if (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    generated = out4_
                                    insideConstrainedOut = out5_
                                    currentConstrainedOut = out6_
                                    d_2_spanEverOpened_ = True
                                    d_3_spanTokens_ = 0
                    elif True:
                        if (d_3_spanTokens_) >= (d_7_minSpanTokens_):
                            d_12_cg_: _dafny.Seq
                            d_13_ci_: bool
                            d_14_cc_: _dafny.Seq
                            d_15_closed_: bool
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out10_: bool
                            out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_12_cg_ = out7_
                            d_13_ci_ = out8_
                            d_14_cc_ = out9_
                            d_15_closed_ = out10_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_15_closed_:
                                generated = d_12_cg_
                                insideConstrainedOut = d_13_ci_
                                currentConstrainedOut = d_14_cc_
                                d_3_spanTokens_ = 0
                            elif True:
                                d_16_constrainedPrompt_: _dafny.Seq
                                d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_17_next_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_17_next_ = out11_
                                if (d_17_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                        d_18_ag_: _dafny.Seq
                                        d_19_ai_: bool
                                        d_20_ac_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out13_: bool
                                        out14_: _dafny.Seq
                                        out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                        d_18_ag_ = out12_
                                        d_19_ai_ = out13_
                                        d_20_ac_ = out14_
                                        generated = d_18_ag_
                                        insideConstrainedOut = d_19_ai_
                                        currentConstrainedOut = d_20_ac_
                                        d_3_spanTokens_ = (d_3_spanTokens_) + (1)
                        elif True:
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_21_cg2_: _dafny.Seq
                                d_22_ci2_: bool
                                d_23_cc2_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_21_cg2_ = out15_
                                d_22_ci2_ = out16_
                                d_23_cc2_ = out17_
                                generated = d_21_cg2_
                                insideConstrainedOut = d_22_ci2_
                                currentConstrainedOut = d_23_cc2_
                                d_3_spanTokens_ = 0
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_24_constrainedPrompt_: _dafny.Seq
                                d_24_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_25_next_: _dafny.Seq
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_25_next_ = out18_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_25_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                        d_26_ag_: _dafny.Seq
                                        d_27_ai_: bool
                                        d_28_ac_: _dafny.Seq
                                        out19_: _dafny.Seq
                                        out20_: bool
                                        out21_: _dafny.Seq
                                        out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                                        d_26_ag_ = out19_
                                        d_27_ai_ = out20_
                                        d_28_ac_ = out21_
                                        generated = d_26_ag_
                                        insideConstrainedOut = d_27_ai_
                                        currentConstrainedOut = d_28_ac_
                                        d_3_spanTokens_ = (d_3_spanTokens_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_29_closeBudget_: int
            d_29_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_30_cg_: _dafny.Seq
            d_31_ci_: bool
            d_32_cc_: _dafny.Seq
            out22_: _dafny.Seq
            out23_: bool
            out24_: _dafny.Seq
            out22_, out23_, out24_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_29_closeBudget_)
            d_30_cg_ = out22_
            d_31_ci_ = out23_
            d_32_cc_ = out24_
            generated = d_30_cg_
            insideConstrainedOut = d_31_ci_
            currentConstrainedOut = d_32_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


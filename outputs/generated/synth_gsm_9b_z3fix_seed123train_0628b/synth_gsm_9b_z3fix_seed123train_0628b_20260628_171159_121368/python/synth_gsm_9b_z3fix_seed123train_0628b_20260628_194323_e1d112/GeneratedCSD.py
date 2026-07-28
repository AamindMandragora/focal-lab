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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. At the very end, write EXACTLY ONE final answer expression inside << >> using only the symbolic variable names from the problem (like n, price, frac, total, etc.) and operators +, -, *, /, //, %, (, ), int(). NEVER use curly braces { } inside << >>. NEVER use $ or LaTeX inside << >>. Use int() for floor/integer division results. Example: <<int(n * p1 / 100 * p2 / 100 * frac)>> or <<(w1 + w2 + w3) * price>>. Write ALL reasoning BEFORE the << >>, then end with EXACTLY ONE <<expression>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanEverOpened_: bool
        d_2_spanEverOpened_ = insideConstrained
        d_3_reservedForSpan_: int
        d_3_reservedForSpan_ = 80
        if (_dafny.euclidian_division(maxSteps, 5)) > (d_3_reservedForSpan_):
            d_3_reservedForSpan_ = _dafny.euclidian_division(maxSteps, 5)
        if (d_3_reservedForSpan_) >= (maxSteps):
            d_3_reservedForSpan_ = _dafny.euclidian_division(maxSteps, 2)
        d_4_forceOpenThreshold_: int
        d_4_forceOpenThreshold_ = (maxSteps) - (d_3_reservedForSpan_)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_2_spanEverOpened_)) and ((d_1_steps_) >= (d_4_forceOpenThreshold_)):
                            d_5_og_: _dafny.Seq
                            d_6_oi_: bool
                            d_7_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_og_ = out0_
                            d_6_oi_ = out1_
                            d_7_oc_ = out2_
                            generated = d_5_og_
                            insideConstrainedOut = d_6_oi_
                            currentConstrainedOut = d_7_oc_
                            d_2_spanEverOpened_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    generated = out4_
                                    insideConstrainedOut = out5_
                                    currentConstrainedOut = out6_
                                    d_2_spanEverOpened_ = True
                    elif True:
                        d_9_cg_: _dafny.Seq
                        d_10_ci_: bool
                        d_11_cc_: _dafny.Seq
                        d_12_closed_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out10_: bool
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_9_cg_ = out7_
                        d_10_ci_ = out8_
                        d_11_cc_ = out9_
                        d_12_closed_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_12_closed_:
                            generated = d_9_cg_
                            insideConstrainedOut = d_10_ci_
                            currentConstrainedOut = d_11_cc_
                        elif True:
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_14_next_: _dafny.Seq
                            d_15_usedFallback_: bool
                            out11_: _dafny.Seq
                            out12_: bool
                            out11_, out12_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                            d_14_next_ = out11_
                            d_15_usedFallback_ = out12_
                            if (d_14_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_16_ag_: _dafny.Seq
                                d_17_ai_: bool
                                d_18_ac_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                d_16_ag_ = out13_
                                d_17_ai_ = out14_
                                d_18_ac_ = out15_
                                generated = d_16_ag_
                                insideConstrainedOut = d_17_ai_
                                currentConstrainedOut = d_18_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_19_closeBudget_: int
            d_19_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_20_cg_: _dafny.Seq
            d_21_ci_: bool
            d_22_cc_: _dafny.Seq
            out16_: _dafny.Seq
            out17_: bool
            out18_: _dafny.Seq
            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_closeBudget_)
            d_20_cg_ = out16_
            d_21_ci_ = out17_
            d_22_cc_ = out18_
            generated = d_20_cg_
            insideConstrainedOut = d_21_ci_
            currentConstrainedOut = d_22_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


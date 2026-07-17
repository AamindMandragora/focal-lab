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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. Show ALL reasoning and calculations using the variable names from the problem. At the very end, write ONE final expression inside << >> using ONLY: variable names, operators +, -, *, /, //, %, (, ), int(). Use int() when the answer must be a whole number. NEVER use { } braces or ** inside << >>. NEVER put text or sentences inside << >>. The final << >> must contain a complete arithmetic expression. Examples: <<int(n * price - discount)>> or <<a + b * c // d>> or <<int((total_cost + extra) / people)>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanEverOpened_: bool
        d_2_spanEverOpened_ = insideConstrained
        d_3_finalSpanMode_: bool
        d_3_finalSpanMode_ = False
        d_4_reservedForFinal_: int = int(0)
        d_5_fracFinal_: int
        d_5_fracFinal_ = _dafny.euclidian_division(maxSteps, 5)
        if (d_5_fracFinal_) >= (120):
            d_4_reservedForFinal_ = d_5_fracFinal_
        elif (maxSteps) >= (120):
            d_4_reservedForFinal_ = 120
        elif True:
            d_4_reservedForFinal_ = _dafny.euclidian_division(maxSteps, 2)
        if (d_4_reservedForFinal_) >= (maxSteps):
            d_4_reservedForFinal_ = _dafny.euclidian_division(maxSteps, 2)
        d_6_forceOpenAt_: int
        d_6_forceOpenAt_ = (maxSteps) - (d_4_reservedForFinal_)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_2_spanEverOpened_)) and ((d_1_steps_) >= (d_6_forceOpenAt_)):
                            d_7_og_: _dafny.Seq
                            d_8_oi_: bool
                            d_9_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_og_ = out0_
                            d_8_oi_ = out1_
                            d_9_oc_ = out2_
                            generated = d_7_og_
                            insideConstrainedOut = d_8_oi_
                            currentConstrainedOut = d_9_oc_
                            d_2_spanEverOpened_ = True
                            d_3_finalSpanMode_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif ((d_2_spanEverOpened_) and (not(d_3_finalSpanMode_))) and ((d_1_steps_) >= (d_6_forceOpenAt_)):
                            d_10_og_: _dafny.Seq
                            d_11_oi_: bool
                            d_12_oc_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_og_ = out3_
                            d_11_oi_ = out4_
                            d_12_oc_ = out5_
                            generated = d_10_og_
                            insideConstrainedOut = d_11_oi_
                            currentConstrainedOut = d_12_oc_
                            d_3_finalSpanMode_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_13_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_13_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_13_next_) == (eosToken):
                                if (not(d_3_finalSpanMode_)) and ((d_1_steps_) < (maxSteps)):
                                    d_14_og_: _dafny.Seq
                                    d_15_oi_: bool
                                    d_16_oc_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_14_og_ = out7_
                                    d_15_oi_ = out8_
                                    d_16_oc_ = out9_
                                    generated = d_14_og_
                                    insideConstrainedOut = d_15_oi_
                                    currentConstrainedOut = d_16_oc_
                                    d_2_spanEverOpened_ = True
                                    d_3_finalSpanMode_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                                if (d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    generated = out10_
                                    insideConstrainedOut = out11_
                                    currentConstrainedOut = out12_
                                    d_2_spanEverOpened_ = True
                                    if (d_1_steps_) >= ((d_6_forceOpenAt_) - (20)):
                                        d_3_finalSpanMode_ = True
                    elif True:
                        if d_3_finalSpanMode_:
                            d_17_cg_: _dafny.Seq
                            d_18_ci_: bool
                            d_19_cc_: _dafny.Seq
                            d_20_closed_: bool
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out16_: bool
                            out13_, out14_, out15_, out16_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_17_cg_ = out13_
                            d_18_ci_ = out14_
                            d_19_cc_ = out15_
                            d_20_closed_ = out16_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_20_closed_:
                                generated = d_17_cg_
                                insideConstrainedOut = d_18_ci_
                                currentConstrainedOut = d_19_cc_
                                raise _dafny.Break("0")
                            elif True:
                                d_21_constrainedPrompt_: _dafny.Seq
                                d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_22_next_: _dafny.Seq
                                out17_: _dafny.Seq
                                out17_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_22_next_ = out17_
                                if (d_22_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_23_ag_: _dafny.Seq
                                    d_24_ai_: bool
                                    d_25_ac_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                    d_23_ag_ = out18_
                                    d_24_ai_ = out19_
                                    d_25_ac_ = out20_
                                    generated = d_23_ag_
                                    insideConstrainedOut = d_24_ai_
                                    currentConstrainedOut = d_25_ac_
                        elif True:
                            d_26_cg_: _dafny.Seq
                            d_27_ci_: bool
                            d_28_cc_: _dafny.Seq
                            d_29_closed_: bool
                            out21_: _dafny.Seq
                            out22_: bool
                            out23_: _dafny.Seq
                            out24_: bool
                            out21_, out22_, out23_, out24_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_26_cg_ = out21_
                            d_27_ci_ = out22_
                            d_28_cc_ = out23_
                            d_29_closed_ = out24_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_29_closed_:
                                generated = d_26_cg_
                                insideConstrainedOut = d_27_ci_
                                currentConstrainedOut = d_28_cc_
                            elif True:
                                d_30_constrainedPrompt_: _dafny.Seq
                                d_30_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_31_next_: _dafny.Seq
                                out25_: _dafny.Seq
                                out25_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_30_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_31_next_ = out25_
                                if (d_31_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_32_ag_: _dafny.Seq
                                    d_33_ai_: bool
                                    d_34_ac_: _dafny.Seq
                                    out26_: _dafny.Seq
                                    out27_: bool
                                    out28_: _dafny.Seq
                                    out26_, out27_, out28_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_31_next_)
                                    d_32_ag_ = out26_
                                    d_33_ai_ = out27_
                                    d_34_ac_ = out28_
                                    generated = d_32_ag_
                                    insideConstrainedOut = d_33_ai_
                                    currentConstrainedOut = d_34_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_35_closeBudget_: int
            d_35_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_36_cg_: _dafny.Seq
            d_37_ci_: bool
            d_38_cc_: _dafny.Seq
            out29_: _dafny.Seq
            out30_: bool
            out31_: _dafny.Seq
            out29_, out30_, out31_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_35_closeBudget_)
            d_36_cg_ = out29_
            d_37_ci_ = out30_
            d_38_cc_ = out31_
            generated = d_36_cg_
            insideConstrainedOut = d_37_ci_
            currentConstrainedOut = d_38_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


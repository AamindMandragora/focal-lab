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
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        (d_0_helpers_).AppendTaskGuidance(lm, (((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write reasoning in plain text, then wrap EACH arithmetic expression in << >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Inside << >> use ONLY: integers, variable names, +, -, *, /, (, ). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NO percent signs, NO braces {}, NO ** exponentiation, NO function calls inside << >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Keep each << >> expression concise. Close >> immediately after the expression. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "The LAST << >> must contain the final numeric answer expression."))))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_maxSpanTokens_: int
        d_3_maxSpanTokens_ = 30
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_4_remaining_) <= (2):
                            d_5_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_5_next_ = out0_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_5_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                                d_6_og_: _dafny.Seq
                                d_7_oi_: bool
                                d_8_oc_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_6_og_ = out1_
                                d_7_oi_ = out2_
                                d_8_oc_ = out3_
                                generated = d_6_og_
                                insideConstrainedOut = d_7_oi_
                                currentConstrainedOut = d_8_oc_
                                d_2_spanSteps_ = 0
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        elif True:
                            d_9_next_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out4_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                d_10_og_: _dafny.Seq
                                d_11_oi_: bool
                                d_12_oc_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out5_, out6_, out7_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_10_og_ = out5_
                                d_11_oi_ = out6_
                                d_12_oc_ = out7_
                                generated = d_10_og_
                                insideConstrainedOut = d_11_oi_
                                currentConstrainedOut = d_12_oc_
                                d_2_spanSteps_ = 0
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                    elif True:
                        if ((d_2_spanSteps_) >= (d_3_maxSpanTokens_)) and ((d_1_steps_) < (maxSteps)):
                            d_13_rg_: _dafny.Seq
                            d_14_rc_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: _dafny.Seq
                            out8_, out9_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_13_rg_ = out8_
                            d_14_rc_ = out9_
                            generated = d_13_rg_
                            currentConstrainedOut = d_14_rc_
                            d_15_isComplete_: bool
                            d_15_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_15_isComplete_:
                                d_16_fg_: _dafny.Seq
                                d_17_fi_: bool
                                d_18_fc_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_16_fg_ = out10_
                                d_17_fi_ = out11_
                                d_18_fc_ = out12_
                                d_1_steps_ = (d_1_steps_) + (1)
                                generated = d_16_fg_
                                insideConstrainedOut = d_17_fi_
                                currentConstrainedOut = d_18_fc_
                                d_2_spanSteps_ = 0
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_spanSteps_ = 0
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_19_cg_: _dafny.Seq
                            d_20_ci_: bool
                            d_21_cc_: _dafny.Seq
                            d_22_closed_: bool
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out16_: bool
                            out13_, out14_, out15_, out16_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_19_cg_ = out13_
                            d_20_ci_ = out14_
                            d_21_cc_ = out15_
                            d_22_closed_ = out16_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_22_closed_:
                                generated = d_19_cg_
                                insideConstrainedOut = d_20_ci_
                                currentConstrainedOut = d_21_cc_
                                d_2_spanSteps_ = 0
                            elif True:
                                d_23_constrainedPrompt_: _dafny.Seq
                                d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_24_next_: _dafny.Seq
                                out17_: _dafny.Seq
                                out17_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_24_next_ = out17_
                                if (d_24_next_) == (eosToken):
                                    d_25_rg_: _dafny.Seq
                                    d_26_rc_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out18_, out19_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_25_rg_ = out18_
                                    d_26_rc_ = out19_
                                    generated = d_25_rg_
                                    currentConstrainedOut = d_26_rc_
                                    d_27_isComplete_: bool
                                    d_27_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if (d_27_isComplete_) and ((d_1_steps_) < (maxSteps)):
                                        d_28_fg_: _dafny.Seq
                                        d_29_fi_: bool
                                        d_30_fc_: _dafny.Seq
                                        out20_: _dafny.Seq
                                        out21_: bool
                                        out22_: _dafny.Seq
                                        out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_28_fg_ = out20_
                                        d_29_fi_ = out21_
                                        d_30_fc_ = out22_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        generated = d_28_fg_
                                        insideConstrainedOut = d_29_fi_
                                        currentConstrainedOut = d_30_fc_
                                    elif True:
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    raise _dafny.Break("0")
                                elif True:
                                    d_31_ag_: _dafny.Seq
                                    d_32_ai_: bool
                                    d_33_ac_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out24_: bool
                                    out25_: _dafny.Seq
                                    out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                    d_31_ag_ = out23_
                                    d_32_ai_ = out24_
                                    d_33_ac_ = out25_
                                    generated = d_31_ag_
                                    insideConstrainedOut = d_32_ai_
                                    currentConstrainedOut = d_33_ac_
                                    d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_34_rg_: _dafny.Seq
            d_35_rc_: _dafny.Seq
            out26_: _dafny.Seq
            out27_: _dafny.Seq
            out26_, out27_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_34_rg_ = out26_
            d_35_rc_ = out27_
            generated = d_34_rg_
            currentConstrainedOut = d_35_rc_
            d_36_isComplete_: bool
            d_36_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_36_isComplete_:
                d_37_fg_: _dafny.Seq
                d_38_fi_: bool
                d_39_fc_: _dafny.Seq
                out28_: _dafny.Seq
                out29_: bool
                out30_: _dafny.Seq
                out28_, out29_, out30_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_37_fg_ = out28_
                d_38_fi_ = out29_
                d_39_fc_ = out30_
                d_1_steps_ = (d_1_steps_) + (1)
                generated = d_37_fg_
                insideConstrainedOut = d_38_fi_
                currentConstrainedOut = d_39_fc_
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
        (d_0_helpers_).AppendTaskGuidance(lm, ((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Show each calculation inside << >> delimiters. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use only +, -, *, /, (, ), numbers, and variable names inside << >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Example: The total is <<n1 + n2>>. Final answer: <<n1 + n2>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Open << only when ready to write a complete expression, then close >> immediately."))))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeSteps_: int
        d_2_freeSteps_ = 0
        d_3_spanSteps_: int
        d_3_spanSteps_ = 0
        d_4_spanBudget_: int
        d_4_spanBudget_ = 40
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_2_freeSteps_) >= (60)) and (((d_1_steps_) + (1)) < (maxSteps)):
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
                            d_1_steps_ = (d_1_steps_) + (1)
                            generated = d_5_og_
                            insideConstrainedOut = d_6_oi_
                            currentConstrainedOut = d_7_oc_
                            d_3_spanSteps_ = 0
                            d_2_freeSteps_ = 0
                        elif True:
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_freeSteps_ = (d_2_freeSteps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                d_9_og_: _dafny.Seq
                                d_10_oi_: bool
                                d_11_oc_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_9_og_ = out4_
                                d_10_oi_ = out5_
                                d_11_oc_ = out6_
                                generated = d_9_og_
                                insideConstrainedOut = d_10_oi_
                                currentConstrainedOut = d_11_oc_
                                d_3_spanSteps_ = 0
                                d_2_freeSteps_ = 0
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                    elif True:
                        if (d_3_spanSteps_) >= (d_4_spanBudget_):
                            d_12_rg_: _dafny.Seq
                            d_13_rc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: _dafny.Seq
                            out7_, out8_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_12_rg_ = out7_
                            d_13_rc_ = out8_
                            generated = d_12_rg_
                            currentConstrainedOut = d_13_rc_
                            d_14_isComplete_: bool
                            d_14_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_14_isComplete_:
                                d_15_fg_: _dafny.Seq
                                d_16_fi_: bool
                                d_17_fc_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_15_fg_ = out9_
                                d_16_fi_ = out10_
                                d_17_fc_ = out11_
                                generated = d_15_fg_
                                insideConstrainedOut = d_16_fi_
                                currentConstrainedOut = d_17_fc_
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanSteps_ = 0
                            d_2_freeSteps_ = 0
                        elif True:
                            d_18_cg_: _dafny.Seq
                            d_19_ci_: bool
                            d_20_cc_: _dafny.Seq
                            d_21_closed_: bool
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out15_: bool
                            out12_, out13_, out14_, out15_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_18_cg_ = out12_
                            d_19_ci_ = out13_
                            d_20_cc_ = out14_
                            d_21_closed_ = out15_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanSteps_ = (d_3_spanSteps_) + (1)
                            if d_21_closed_:
                                generated = d_18_cg_
                                insideConstrainedOut = d_19_ci_
                                currentConstrainedOut = d_20_cc_
                                d_3_spanSteps_ = 0
                                d_2_freeSteps_ = 0
                            elif True:
                                d_22_constrainedPrompt_: _dafny.Seq
                                d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_23_next_: _dafny.Seq
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_23_next_ = out16_
                                if (d_23_next_) == (eosToken):
                                    d_24_rg_: _dafny.Seq
                                    d_25_rc_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out17_, out18_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_24_rg_ = out17_
                                    d_25_rc_ = out18_
                                    generated = d_24_rg_
                                    currentConstrainedOut = d_25_rc_
                                    d_26_isComplete_: bool
                                    d_26_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if (d_26_isComplete_) and ((d_1_steps_) < (maxSteps)):
                                        d_27_fg_: _dafny.Seq
                                        d_28_fi_: bool
                                        d_29_fc_: _dafny.Seq
                                        out19_: _dafny.Seq
                                        out20_: bool
                                        out21_: _dafny.Seq
                                        out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_27_fg_ = out19_
                                        d_28_fi_ = out20_
                                        d_29_fc_ = out21_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        generated = d_27_fg_
                                        insideConstrainedOut = d_28_fi_
                                        currentConstrainedOut = d_29_fc_
                                    elif True:
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    raise _dafny.Break("0")
                                elif True:
                                    d_30_ag_: _dafny.Seq
                                    d_31_ai_: bool
                                    d_32_ac_: _dafny.Seq
                                    out22_: _dafny.Seq
                                    out23_: bool
                                    out24_: _dafny.Seq
                                    out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                    d_30_ag_ = out22_
                                    d_31_ai_ = out23_
                                    d_32_ac_ = out24_
                                    generated = d_30_ag_
                                    insideConstrainedOut = d_31_ai_
                                    currentConstrainedOut = d_32_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_33_rg_: _dafny.Seq
            d_34_rc_: _dafny.Seq
            out25_: _dafny.Seq
            out26_: _dafny.Seq
            out25_, out26_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_33_rg_ = out25_
            d_34_rc_ = out26_
            generated = d_33_rg_
            currentConstrainedOut = d_34_rc_
            d_35_isComplete_: bool
            d_35_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_35_isComplete_:
                d_36_fg_: _dafny.Seq
                d_37_fi_: bool
                d_38_fc_: _dafny.Seq
                out27_: _dafny.Seq
                out28_: bool
                out29_: _dafny.Seq
                out27_, out28_, out29_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_36_fg_ = out27_
                d_37_fi_ = out28_
                d_38_fc_ = out29_
                d_1_steps_ = (d_1_steps_) + (1)
                generated = d_36_fg_
                insideConstrainedOut = d_37_fi_
                currentConstrainedOut = d_38_fc_
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


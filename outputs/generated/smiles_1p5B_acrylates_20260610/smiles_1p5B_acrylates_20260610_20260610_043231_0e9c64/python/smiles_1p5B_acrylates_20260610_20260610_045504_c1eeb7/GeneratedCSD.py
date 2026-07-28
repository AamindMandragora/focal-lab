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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Task: output one valid SMILES for a novel acrylate molecule. Acrylates have the substructure C=CC(=O)O (vinyl ester). Start the SMILES with C=C and include the acryloyl ester group. Good examples: C=CC(=O)OCC, C=CC(=O)OCCO, C=C(C)C(=O)OCC, C=CC(=O)OCCCCC. Do not output just a single atom.")))
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
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
        d_5_minLength_: int
        d_5_minLength_ = 10
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if (len(currentConstrainedOut)) >= (d_5_minLength_):
                        d_6_cg_: _dafny.Seq
                        d_7_ci_: bool
                        d_8_cc_: _dafny.Seq
                        d_9_closed_: bool
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out6_: bool
                        out3_, out4_, out5_, out6_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_6_cg_ = out3_
                        d_7_ci_ = out4_
                        d_8_cc_ = out5_
                        d_9_closed_ = out6_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_9_closed_:
                            generated = d_6_cg_
                            insideConstrainedOut = d_7_ci_
                            currentConstrainedOut = d_8_cc_
                        elif True:
                            if (d_1_steps_) < (maxSteps):
                                d_10_isComplete_: bool
                                d_10_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_10_isComplete_:
                                    d_11_fg_: _dafny.Seq
                                    d_12_fi_: bool
                                    d_13_fc_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_11_fg_ = out7_
                                    d_12_fi_ = out8_
                                    d_13_fc_ = out9_
                                    generated = d_11_fg_
                                    insideConstrainedOut = d_12_fi_
                                    currentConstrainedOut = d_13_fc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_14_constrainedPrompt_: _dafny.Seq
                                    d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_15_next_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                                    d_15_next_ = out10_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_15_next_) == (eosToken):
                                        d_16_rg_: _dafny.Seq
                                        d_17_rc_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out11_, out12_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                        d_16_rg_ = out11_
                                        d_17_rc_ = out12_
                                        generated = d_16_rg_
                                        currentConstrainedOut = d_17_rc_
                                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                            d_18_fg_: _dafny.Seq
                                            d_19_fi_: bool
                                            d_20_fc_: _dafny.Seq
                                            out13_: _dafny.Seq
                                            out14_: bool
                                            out15_: _dafny.Seq
                                            out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_18_fg_ = out13_
                                            d_19_fi_ = out14_
                                            d_20_fc_ = out15_
                                            generated = d_18_fg_
                                            insideConstrainedOut = d_19_fi_
                                            currentConstrainedOut = d_20_fc_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_21_alreadyComplete_: bool
                                        d_21_alreadyComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                        if not(d_21_alreadyComplete_):
                                            d_22_ag_: _dafny.Seq
                                            d_23_ai_: bool
                                            d_24_ac_: _dafny.Seq
                                            out16_: _dafny.Seq
                                            out17_: bool
                                            out18_: _dafny.Seq
                                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                            d_22_ag_ = out16_
                                            d_23_ai_ = out17_
                                            d_24_ac_ = out18_
                                            generated = d_22_ag_
                                            insideConstrainedOut = d_23_ai_
                                            currentConstrainedOut = d_24_ac_
                                        elif True:
                                            if (d_1_steps_) < (maxSteps):
                                                d_25_fg_: _dafny.Seq
                                                d_26_fi_: bool
                                                d_27_fc_: _dafny.Seq
                                                out19_: _dafny.Seq
                                                out20_: bool
                                                out21_: _dafny.Seq
                                                out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                d_25_fg_ = out19_
                                                d_26_fi_ = out20_
                                                d_27_fc_ = out21_
                                                generated = d_25_fg_
                                                insideConstrainedOut = d_26_fi_
                                                currentConstrainedOut = d_27_fc_
                                                d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_28_isComplete_: bool
                        d_28_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if (d_28_isComplete_) and ((len(currentConstrainedOut)) > (0)):
                            if (d_1_steps_) < (maxSteps):
                                d_29_fg_: _dafny.Seq
                                d_30_fi_: bool
                                d_31_fc_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: bool
                                out24_: _dafny.Seq
                                out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_29_fg_ = out22_
                                d_30_fi_ = out23_
                                d_31_fc_ = out24_
                                generated = d_29_fg_
                                insideConstrainedOut = d_30_fi_
                                currentConstrainedOut = d_31_fc_
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_32_constrainedPrompt_: _dafny.Seq
                            d_32_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_33_next_: _dafny.Seq
                            out25_: _dafny.Seq
                            out25_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_32_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                            d_33_next_ = out25_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_33_next_) == (eosToken):
                                d_34_rg_: _dafny.Seq
                                d_35_rc_: _dafny.Seq
                                out26_: _dafny.Seq
                                out27_: _dafny.Seq
                                out26_, out27_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_34_rg_ = out26_
                                d_35_rc_ = out27_
                                generated = d_34_rg_
                                currentConstrainedOut = d_35_rc_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_36_fg_: _dafny.Seq
                                    d_37_fi_: bool
                                    d_38_fc_: _dafny.Seq
                                    out28_: _dafny.Seq
                                    out29_: bool
                                    out30_: _dafny.Seq
                                    out28_, out29_, out30_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_36_fg_ = out28_
                                    d_37_fi_ = out29_
                                    d_38_fc_ = out30_
                                    generated = d_36_fg_
                                    insideConstrainedOut = d_37_fi_
                                    currentConstrainedOut = d_38_fc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                d_39_alreadyComplete_: bool
                                d_39_alreadyComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if not(d_39_alreadyComplete_):
                                    d_40_ag_: _dafny.Seq
                                    d_41_ai_: bool
                                    d_42_ac_: _dafny.Seq
                                    out31_: _dafny.Seq
                                    out32_: bool
                                    out33_: _dafny.Seq
                                    out31_, out32_, out33_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_33_next_)
                                    d_40_ag_ = out31_
                                    d_41_ai_ = out32_
                                    d_42_ac_ = out33_
                                    generated = d_40_ag_
                                    insideConstrainedOut = d_41_ai_
                                    currentConstrainedOut = d_42_ac_
                                elif True:
                                    if (d_1_steps_) < (maxSteps):
                                        d_43_fg_: _dafny.Seq
                                        d_44_fi_: bool
                                        d_45_fc_: _dafny.Seq
                                        out34_: _dafny.Seq
                                        out35_: bool
                                        out36_: _dafny.Seq
                                        out34_, out35_, out36_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_43_fg_ = out34_
                                        d_44_fi_ = out35_
                                        d_45_fc_ = out36_
                                        generated = d_43_fg_
                                        insideConstrainedOut = d_44_fi_
                                        currentConstrainedOut = d_45_fc_
                                        d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_46_rg_: _dafny.Seq
            d_47_rc_: _dafny.Seq
            out37_: _dafny.Seq
            out38_: _dafny.Seq
            out37_, out38_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_46_rg_ = out37_
            d_47_rc_ = out38_
            generated = d_46_rg_
            currentConstrainedOut = d_47_rc_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_48_fg_: _dafny.Seq
                d_49_fi_: bool
                d_50_fc_: _dafny.Seq
                out39_: _dafny.Seq
                out40_: bool
                out41_: _dafny.Seq
                out39_, out40_, out41_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_48_fg_ = out39_
                d_49_fi_ = out40_
                d_50_fc_ = out41_
                generated = d_48_fg_
                insideConstrainedOut = d_49_fi_
                currentConstrainedOut = d_50_fc_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


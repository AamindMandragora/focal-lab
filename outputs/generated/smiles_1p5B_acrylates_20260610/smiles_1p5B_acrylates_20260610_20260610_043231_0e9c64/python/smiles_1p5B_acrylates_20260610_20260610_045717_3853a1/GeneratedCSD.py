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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SMILES for a novel acrylate molecule. Acrylates have the acryloyl group C=CC(=O)O. Generate a complete SMILES like C=CC(=O)OCCC or C=C(C)C(=O)OCCCO. The SMILES must be at least 8 characters and contain the acrylate ester substructure.")))
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
        d_5_minLength_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    if (len(currentConstrainedOut)) >= (d_5_minLength_):
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            if (d_1_steps_) < (maxSteps):
                                d_6_cg_: _dafny.Seq
                                d_7_ci_: bool
                                d_8_cc_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_6_cg_ = out3_
                                d_7_ci_ = out4_
                                d_8_cc_ = out5_
                                generated = d_6_cg_
                                insideConstrainedOut = d_7_ci_
                                currentConstrainedOut = d_8_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_9_constrainedPrompt_: _dafny.Seq
                            d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_10_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                            d_10_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                d_11_rg_: _dafny.Seq
                                d_12_rc_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: _dafny.Seq
                                out7_, out8_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_11_rg_ = out7_
                                d_12_rc_ = out8_
                                generated = d_11_rg_
                                currentConstrainedOut = d_12_rc_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_13_fg_: _dafny.Seq
                                    d_14_fi_: bool
                                    d_15_fc_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_13_fg_ = out9_
                                    d_14_fi_ = out10_
                                    d_15_fc_ = out11_
                                    generated = d_13_fg_
                                    insideConstrainedOut = d_14_fi_
                                    currentConstrainedOut = d_15_fc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                    d_16_ag_: _dafny.Seq
                                    d_17_ai_: bool
                                    d_18_ac_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                                    d_16_ag_ = out12_
                                    d_17_ai_ = out13_
                                    d_18_ac_ = out14_
                                    generated = d_16_ag_
                                    insideConstrainedOut = d_17_ai_
                                    currentConstrainedOut = d_18_ac_
                                if ((d_1_steps_) < (maxSteps)) and ((len(currentConstrainedOut)) >= (d_5_minLength_)):
                                    d_19_cg2_: _dafny.Seq
                                    d_20_ci2_: bool
                                    d_21_cc2_: _dafny.Seq
                                    d_22_closed_: bool
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out15_, out16_, out17_, out18_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                    d_19_cg2_ = out15_
                                    d_20_ci2_ = out16_
                                    d_21_cc2_ = out17_
                                    d_22_closed_ = out18_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if d_22_closed_:
                                        generated = d_19_cg2_
                                        insideConstrainedOut = d_20_ci2_
                                        currentConstrainedOut = d_21_cc2_
                                        raise _dafny.Break("0")
                    elif True:
                        d_23_constrainedPrompt_: _dafny.Seq
                        d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_24_next_: _dafny.Seq
                        out19_: _dafny.Seq
                        out19_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                        d_24_next_ = out19_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_24_next_) == (eosToken):
                            d_25_rg_: _dafny.Seq
                            d_26_rc_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: _dafny.Seq
                            out20_, out21_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_25_rg_ = out20_
                            d_26_rc_ = out21_
                            generated = d_25_rg_
                            currentConstrainedOut = d_26_rc_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_27_fg_: _dafny.Seq
                                d_28_fi_: bool
                                d_29_fc_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: bool
                                out24_: _dafny.Seq
                                out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_27_fg_ = out22_
                                d_28_fi_ = out23_
                                d_29_fc_ = out24_
                                generated = d_27_fg_
                                insideConstrainedOut = d_28_fi_
                                currentConstrainedOut = d_29_fc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_30_ag_: _dafny.Seq
                                d_31_ai_: bool
                                d_32_ac_: _dafny.Seq
                                out25_: _dafny.Seq
                                out26_: bool
                                out27_: _dafny.Seq
                                out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                d_30_ag_ = out25_
                                d_31_ai_ = out26_
                                d_32_ac_ = out27_
                                generated = d_30_ag_
                                insideConstrainedOut = d_31_ai_
                                currentConstrainedOut = d_32_ac_
                            elif True:
                                if (d_1_steps_) < (maxSteps):
                                    d_33_fg_: _dafny.Seq
                                    d_34_fi_: bool
                                    d_35_fc_: _dafny.Seq
                                    out28_: _dafny.Seq
                                    out29_: bool
                                    out30_: _dafny.Seq
                                    out28_, out29_, out30_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_33_fg_ = out28_
                                    d_34_fi_ = out29_
                                    d_35_fc_ = out30_
                                    generated = d_33_fg_
                                    insideConstrainedOut = d_34_fi_
                                    currentConstrainedOut = d_35_fc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_36_rg_: _dafny.Seq
            d_37_rc_: _dafny.Seq
            out31_: _dafny.Seq
            out32_: _dafny.Seq
            out31_, out32_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_36_rg_ = out31_
            d_37_rc_ = out32_
            generated = d_36_rg_
            currentConstrainedOut = d_37_rc_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_38_fg_: _dafny.Seq
                d_39_fi_: bool
                d_40_fc_: _dafny.Seq
                out33_: _dafny.Seq
                out34_: bool
                out35_: _dafny.Seq
                out33_, out34_, out35_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_38_fg_ = out33_
                d_39_fi_ = out34_
                d_40_fc_ = out35_
                generated = d_38_fg_
                insideConstrainedOut = d_39_fi_
                currentConstrainedOut = d_40_fc_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


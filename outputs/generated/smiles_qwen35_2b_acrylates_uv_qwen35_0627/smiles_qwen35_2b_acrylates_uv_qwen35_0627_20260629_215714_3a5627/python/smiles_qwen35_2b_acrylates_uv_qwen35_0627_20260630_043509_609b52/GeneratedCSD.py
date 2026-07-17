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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Task: generate a novel acrylate ester SMILES string. Must contain acryloyl ester core C=CC(=O)O bonded to a carbon group R. Output ONLY the SMILES. Examples of valid acrylates: C=CC(=O)OCC, C=CC(=O)OCCO, C=CC(=O)OC(C)C, C=CC(=O)OCCC, C=CC(=O)OCCCC, C=CC(=O)OC(C)(C)C, C=CC(=O)OCC(C)C. Generate ONE novel acrylate SMILES.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_minConstrainedTokens_: int
        d_2_minConstrainedTokens_ = 12
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_3_og_: _dafny.Seq
            d_4_oi_: bool
            d_5_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_3_og_ = out0_
            d_4_oi_ = out1_
            d_5_oc_ = out2_
            generated = d_3_og_
            insideConstrainedOut = d_4_oi_
            currentConstrainedOut = d_5_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    d_6_remaining_: int
                    d_6_remaining_ = (maxSteps) - (d_1_steps_)
                    if (d_6_remaining_) <= (40):
                        d_7_csg_: _dafny.Seq
                        d_8_csi_: bool
                        d_9_csc_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_6_remaining_)
                        d_7_csg_ = out3_
                        d_8_csi_ = out4_
                        d_9_csc_ = out5_
                        generated = d_7_csg_
                        insideConstrainedOut = d_8_csi_
                        currentConstrainedOut = d_9_csc_
                        d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    d_10_constrainedLen_: int
                    d_10_constrainedLen_ = len(currentConstrainedOut)
                    d_11_constrainedPrompt_: _dafny.Seq
                    d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (d_10_constrainedLen_):]))
                    if (d_10_constrainedLen_) >= (d_2_minConstrainedTokens_):
                        d_12_cg_: _dafny.Seq
                        d_13_ci_: bool
                        d_14_cc_: _dafny.Seq
                        d_15_closed_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out9_: bool
                        out6_, out7_, out8_, out9_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_12_cg_ = out6_
                        d_13_ci_ = out7_
                        d_14_cc_ = out8_
                        d_15_closed_ = out9_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_15_closed_:
                            generated = d_12_cg_
                            insideConstrainedOut = d_13_ci_
                            currentConstrainedOut = d_14_cc_
                            raise _dafny.Break("0")
                        d_16_narrow_: bool
                        out10_: bool
                        out10_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                        d_16_narrow_ = out10_
                        if (d_16_narrow_) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                            if (d_1_steps_) < (maxSteps):
                                d_17_cg2_: _dafny.Seq
                                d_18_ci2_: bool
                                d_19_cc2_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_17_cg2_ = out11_
                                d_18_ci2_ = out12_
                                d_19_cc2_ = out13_
                                generated = d_17_cg2_
                                insideConstrainedOut = d_18_ci2_
                                currentConstrainedOut = d_19_cc2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif d_16_narrow_:
                            d_20_rg_: _dafny.Seq
                            d_21_rc_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: _dafny.Seq
                            out14_, out15_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_20_rg_ = out14_
                            d_21_rc_ = out15_
                            if ((parser).IsCompletePrefix(d_21_rc_)) and ((len(d_21_rc_)) >= (8)):
                                generated = d_20_rg_
                                currentConstrainedOut = d_21_rc_
                                if (d_1_steps_) < (maxSteps):
                                    d_22_cg3_: _dafny.Seq
                                    d_23_ci3_: bool
                                    d_24_cc3_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_22_cg3_ = out16_
                                    d_23_ci3_ = out17_
                                    d_24_cc3_ = out18_
                                    generated = d_22_cg3_
                                    insideConstrainedOut = d_23_ci3_
                                    currentConstrainedOut = d_24_cc3_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                        if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
                            d_25_next_: _dafny.Seq
                            out19_: _dafny.Seq
                            out19_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('1e0'), eosToken)
                            d_25_next_ = out19_
                            if (d_25_next_) == (eosToken):
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_26_cg4_: _dafny.Seq
                                    d_27_ci4_: bool
                                    d_28_cc4_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_26_cg4_ = out20_
                                    d_27_ci4_ = out21_
                                    d_28_cc4_ = out22_
                                    generated = d_26_cg4_
                                    insideConstrainedOut = d_27_ci4_
                                    currentConstrainedOut = d_28_cc4_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                d_29_valid_: bool
                                out23_: bool
                                out23_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_25_next_)
                                d_29_valid_ = out23_
                                if (d_29_valid_) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                                    d_30_ag_: _dafny.Seq
                                    d_31_ai_: bool
                                    d_32_ac_: _dafny.Seq
                                    out24_: _dafny.Seq
                                    out25_: bool
                                    out26_: _dafny.Seq
                                    out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                                    d_30_ag_ = out24_
                                    d_31_ai_ = out25_
                                    d_32_ac_ = out26_
                                    generated = d_30_ag_
                                    insideConstrainedOut = d_31_ai_
                                    currentConstrainedOut = d_32_ac_
                                elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_33_cg5_: _dafny.Seq
                                    d_34_ci5_: bool
                                    d_35_cc5_: _dafny.Seq
                                    out27_: _dafny.Seq
                                    out28_: bool
                                    out29_: _dafny.Seq
                                    out27_, out28_, out29_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_33_cg5_ = out27_
                                    d_34_ci5_ = out28_
                                    d_35_cc5_ = out29_
                                    generated = d_33_cg5_
                                    insideConstrainedOut = d_34_ci5_
                                    currentConstrainedOut = d_35_cc5_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                        elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_36_cg6_: _dafny.Seq
                            d_37_ci6_: bool
                            d_38_cc6_: _dafny.Seq
                            out30_: _dafny.Seq
                            out31_: bool
                            out32_: _dafny.Seq
                            out30_, out31_, out32_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_36_cg6_ = out30_
                            d_37_ci6_ = out31_
                            d_38_cc6_ = out32_
                            generated = d_36_cg6_
                            insideConstrainedOut = d_37_ci6_
                            currentConstrainedOut = d_38_cc6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                    elif True:
                        if not((parser).IsCompletePrefix(currentConstrainedOut)):
                            d_39_next_: _dafny.Seq
                            out33_: _dafny.Seq
                            out33_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_39_next_ = out33_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_39_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_40_valid_: bool
                                out34_: bool
                                out34_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_39_next_)
                                d_40_valid_ = out34_
                                if (d_40_valid_) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                                    d_41_ag_: _dafny.Seq
                                    d_42_ai_: bool
                                    d_43_ac_: _dafny.Seq
                                    out35_: _dafny.Seq
                                    out36_: bool
                                    out37_: _dafny.Seq
                                    out35_, out36_, out37_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_39_next_)
                                    d_41_ag_ = out35_
                                    d_42_ai_ = out36_
                                    d_43_ac_ = out37_
                                    generated = d_41_ag_
                                    insideConstrainedOut = d_42_ai_
                                    currentConstrainedOut = d_43_ac_
                        elif True:
                            d_44_next_: _dafny.Seq
                            out38_: _dafny.Seq
                            out38_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('1e0'), eosToken)
                            d_44_next_ = out38_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_44_next_) == (eosToken):
                                if (d_1_steps_) < (maxSteps):
                                    d_45_cg7_: _dafny.Seq
                                    d_46_ci7_: bool
                                    d_47_cc7_: _dafny.Seq
                                    out39_: _dafny.Seq
                                    out40_: bool
                                    out41_: _dafny.Seq
                                    out39_, out40_, out41_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_45_cg7_ = out39_
                                    d_46_ci7_ = out40_
                                    d_47_cc7_ = out41_
                                    generated = d_45_cg7_
                                    insideConstrainedOut = d_46_ci7_
                                    currentConstrainedOut = d_47_cc7_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                d_48_valid_: bool
                                out42_: bool
                                out42_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_44_next_)
                                d_48_valid_ = out42_
                                if (d_48_valid_) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                                    d_49_ag_: _dafny.Seq
                                    d_50_ai_: bool
                                    d_51_ac_: _dafny.Seq
                                    out43_: _dafny.Seq
                                    out44_: bool
                                    out45_: _dafny.Seq
                                    out43_, out44_, out45_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_44_next_)
                                    d_49_ag_ = out43_
                                    d_50_ai_ = out44_
                                    d_51_ac_ = out45_
                                    generated = d_49_ag_
                                    insideConstrainedOut = d_50_ai_
                                    currentConstrainedOut = d_51_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_52_remaining2_: int
            d_52_remaining2_ = (maxSteps) - (d_1_steps_)
            d_53_csg2_: _dafny.Seq
            d_54_csi2_: bool
            d_55_csc2_: _dafny.Seq
            out46_: _dafny.Seq
            out47_: bool
            out48_: _dafny.Seq
            out46_, out47_, out48_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_52_remaining2_)
            d_53_csg2_ = out46_
            d_54_csi2_ = out47_
            d_55_csc2_ = out48_
            generated = d_53_csg2_
            insideConstrainedOut = d_54_csi2_
            currentConstrainedOut = d_55_csc2_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


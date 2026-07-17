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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a NOVEL, UNIQUE acrylate ester SMILES. Must contain C=CC(=O)O core. Create diverse structures: branched alkyl esters, fluorinated esters, cycloalkyl esters, glycol esters, or long chain esters. Examples: C=CC(=O)OCCC(C)C, C=CC(=O)OC(CC)CC, C=CC(=O)OCC(F)(F)F, C=CC(=O)OCCCCCC, C=CC(=O)OC1CCCCC1. Output ONLY the SMILES string.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_minConstrainedTokens_: int
        d_2_minConstrainedTokens_ = 10
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
                        if d_16_narrow_:
                            d_17_rg_: _dafny.Seq
                            d_18_rc_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out11_, out12_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_17_rg_ = out11_
                            d_18_rc_ = out12_
                            d_19_rcComplete_: bool
                            d_19_rcComplete_ = (parser).IsCompletePrefix(d_18_rc_)
                            if (d_19_rcComplete_) and ((len(d_18_rc_)) >= (8)):
                                generated = d_17_rg_
                                currentConstrainedOut = d_18_rc_
                                if (d_1_steps_) < (maxSteps):
                                    d_20_cg2_: _dafny.Seq
                                    d_21_ci2_: bool
                                    d_22_cc2_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_20_cg2_ = out13_
                                    d_21_ci2_ = out14_
                                    d_22_cc2_ = out15_
                                    generated = d_20_cg2_
                                    insideConstrainedOut = d_21_ci2_
                                    currentConstrainedOut = d_22_cc2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            d_23_remaining2_: int
                            d_23_remaining2_ = (maxSteps) - (d_1_steps_)
                            d_24_csg2_: _dafny.Seq
                            d_25_csi2_: bool
                            d_26_csc2_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_remaining2_)
                            d_24_csg2_ = out16_
                            d_25_csi2_ = out17_
                            d_26_csc2_ = out18_
                            generated = d_24_csg2_
                            insideConstrainedOut = d_25_csi2_
                            currentConstrainedOut = d_26_csc2_
                            d_1_steps_ = maxSteps
                            raise _dafny.Break("0")
                        d_27_isComplete_: bool
                        d_27_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if not(d_27_isComplete_):
                            d_28_next_: _dafny.Seq
                            out19_: _dafny.Seq
                            out19_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('9e-1'), eosToken)
                            d_28_next_ = out19_
                            if (d_28_next_) == (eosToken):
                                d_29_isComplete2_: bool
                                d_29_isComplete2_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if (d_29_isComplete2_) and ((d_1_steps_) < (maxSteps)):
                                    d_30_cg3_: _dafny.Seq
                                    d_31_ci3_: bool
                                    d_32_cc3_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_30_cg3_ = out20_
                                    d_31_ci3_ = out21_
                                    d_32_cc3_ = out22_
                                    generated = d_30_cg3_
                                    insideConstrainedOut = d_31_ci3_
                                    currentConstrainedOut = d_32_cc3_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                d_33_valid_: bool
                                out23_: bool
                                out23_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_28_next_)
                                d_33_valid_ = out23_
                                d_34_notComplete3_: bool
                                d_34_notComplete3_ = not((parser).IsCompletePrefix(currentConstrainedOut))
                                if (d_33_valid_) and (d_34_notComplete3_):
                                    d_35_ag_: _dafny.Seq
                                    d_36_ai_: bool
                                    d_37_ac_: _dafny.Seq
                                    out24_: _dafny.Seq
                                    out25_: bool
                                    out26_: _dafny.Seq
                                    out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_next_)
                                    d_35_ag_ = out24_
                                    d_36_ai_ = out25_
                                    d_37_ac_ = out26_
                                    generated = d_35_ag_
                                    insideConstrainedOut = d_36_ai_
                                    currentConstrainedOut = d_37_ac_
                        elif True:
                            if (d_1_steps_) < (maxSteps):
                                d_38_cg4_: _dafny.Seq
                                d_39_ci4_: bool
                                d_40_cc4_: _dafny.Seq
                                out27_: _dafny.Seq
                                out28_: bool
                                out29_: _dafny.Seq
                                out27_, out28_, out29_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_38_cg4_ = out27_
                                d_39_ci4_ = out28_
                                d_40_cc4_ = out29_
                                generated = d_38_cg4_
                                insideConstrainedOut = d_39_ci4_
                                currentConstrainedOut = d_40_cc4_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                    elif True:
                        d_41_isComplete4_: bool
                        d_41_isComplete4_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if not(d_41_isComplete4_):
                            d_42_next_: _dafny.Seq
                            out30_: _dafny.Seq
                            out30_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_42_next_ = out30_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_42_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_43_valid_: bool
                                out31_: bool
                                out31_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_42_next_)
                                d_43_valid_ = out31_
                                d_44_notComplete5_: bool
                                d_44_notComplete5_ = not((parser).IsCompletePrefix(currentConstrainedOut))
                                if (d_43_valid_) and (d_44_notComplete5_):
                                    d_45_ag_: _dafny.Seq
                                    d_46_ai_: bool
                                    d_47_ac_: _dafny.Seq
                                    out32_: _dafny.Seq
                                    out33_: bool
                                    out34_: _dafny.Seq
                                    out32_, out33_, out34_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_42_next_)
                                    d_45_ag_ = out32_
                                    d_46_ai_ = out33_
                                    d_47_ac_ = out34_
                                    generated = d_45_ag_
                                    insideConstrainedOut = d_46_ai_
                                    currentConstrainedOut = d_47_ac_
                        elif True:
                            if (d_1_steps_) < (maxSteps):
                                d_48_cg5_: _dafny.Seq
                                d_49_ci5_: bool
                                d_50_cc5_: _dafny.Seq
                                out35_: _dafny.Seq
                                out36_: bool
                                out37_: _dafny.Seq
                                out35_, out36_, out37_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_48_cg5_ = out35_
                                d_49_ci5_ = out36_
                                d_50_cc5_ = out37_
                                generated = d_48_cg5_
                                insideConstrainedOut = d_49_ci5_
                                currentConstrainedOut = d_50_cc5_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_51_remaining3_: int
            d_51_remaining3_ = (maxSteps) - (d_1_steps_)
            d_52_csg3_: _dafny.Seq
            d_53_csi3_: bool
            d_54_csc3_: _dafny.Seq
            out38_: _dafny.Seq
            out39_: bool
            out40_: _dafny.Seq
            out38_, out39_, out40_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_51_remaining3_)
            d_52_csg3_ = out38_
            d_53_csi3_ = out39_
            d_54_csc3_ = out40_
            generated = d_52_csg3_
            insideConstrainedOut = d_53_csi3_
            currentConstrainedOut = d_54_csc3_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


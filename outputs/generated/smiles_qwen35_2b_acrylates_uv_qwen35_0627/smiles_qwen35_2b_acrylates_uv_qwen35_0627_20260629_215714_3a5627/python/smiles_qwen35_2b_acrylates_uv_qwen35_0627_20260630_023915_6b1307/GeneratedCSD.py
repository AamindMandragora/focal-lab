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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are generating a SMILES string for a novel acrylate molecule. Acrylates have core structure C=CC(=O)O or CC(=C)C(=O)O (methyl acrylate variant). Generate diverse ester/functional group substitution. Output only the SMILES.")))
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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    d_5_remaining_: int
                    d_5_remaining_ = (maxSteps) - (d_1_steps_)
                    if (d_5_remaining_) <= (40):
                        d_6_csg_: _dafny.Seq
                        d_7_csi_: bool
                        d_8_csc_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_5_remaining_)
                        d_6_csg_ = out3_
                        d_7_csi_ = out4_
                        d_8_csc_ = out5_
                        generated = d_6_csg_
                        insideConstrainedOut = d_7_csi_
                        currentConstrainedOut = d_8_csc_
                        d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    d_9_constrainedLen_: int
                    d_9_constrainedLen_ = len(currentConstrainedOut)
                    d_10_constrainedPrompt_: _dafny.Seq
                    d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (d_9_constrainedLen_):]))
                    if (d_9_constrainedLen_) >= (12):
                        d_11_cg_: _dafny.Seq
                        d_12_ci_: bool
                        d_13_cc_: _dafny.Seq
                        d_14_closed_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out9_: bool
                        out6_, out7_, out8_, out9_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_11_cg_ = out6_
                        d_12_ci_ = out7_
                        d_13_cc_ = out8_
                        d_14_closed_ = out9_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_14_closed_:
                            generated = d_11_cg_
                            insideConstrainedOut = d_12_ci_
                            currentConstrainedOut = d_13_cc_
                            raise _dafny.Break("0")
                        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_15_cg3_: _dafny.Seq
                                d_16_ci3_: bool
                                d_17_cc3_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_15_cg3_ = out10_
                                d_16_ci3_ = out11_
                                d_17_cc3_ = out12_
                                generated = d_15_cg3_
                                insideConstrainedOut = d_16_ci3_
                                currentConstrainedOut = d_17_cc3_
                                d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                d_18_next_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e-1'), eosToken)
                                d_18_next_ = out13_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_18_next_) == (eosToken):
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_19_cg4_: _dafny.Seq
                                        d_20_ci4_: bool
                                        d_21_cc4_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out15_: bool
                                        out16_: _dafny.Seq
                                        out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_19_cg4_ = out14_
                                        d_20_ci4_ = out15_
                                        d_21_cc4_ = out16_
                                        generated = d_19_cg4_
                                        insideConstrainedOut = d_20_ci4_
                                        currentConstrainedOut = d_21_cc4_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                                elif True:
                                    if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                        d_22_valid_: bool
                                        out17_: bool
                                        out17_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_18_next_)
                                        d_22_valid_ = out17_
                                        if d_22_valid_:
                                            d_23_ag_: _dafny.Seq
                                            d_24_ai_: bool
                                            d_25_ac_: _dafny.Seq
                                            out18_: _dafny.Seq
                                            out19_: bool
                                            out20_: _dafny.Seq
                                            out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                            d_23_ag_ = out18_
                                            d_24_ai_ = out19_
                                            d_25_ac_ = out20_
                                            generated = d_23_ag_
                                            insideConstrainedOut = d_24_ai_
                                            currentConstrainedOut = d_25_ac_
                                    elif True:
                                        if (d_1_steps_) < (maxSteps):
                                            d_26_cg5_: _dafny.Seq
                                            d_27_ci5_: bool
                                            d_28_cc5_: _dafny.Seq
                                            out21_: _dafny.Seq
                                            out22_: bool
                                            out23_: _dafny.Seq
                                            out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_26_cg5_ = out21_
                                            d_27_ci5_ = out22_
                                            d_28_cc5_ = out23_
                                            generated = d_26_cg5_
                                            insideConstrainedOut = d_27_ci5_
                                            currentConstrainedOut = d_28_cc5_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                        raise _dafny.Break("0")
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            if (d_1_steps_) < (maxSteps):
                                d_29_cg6_: _dafny.Seq
                                d_30_ci6_: bool
                                d_31_cc6_: _dafny.Seq
                                out24_: _dafny.Seq
                                out25_: bool
                                out26_: _dafny.Seq
                                out24_, out25_, out26_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_29_cg6_ = out24_
                                d_30_ci6_ = out25_
                                d_31_cc6_ = out26_
                                generated = d_29_cg6_
                                insideConstrainedOut = d_30_ci6_
                                currentConstrainedOut = d_31_cc6_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_32_next2_: _dafny.Seq
                            out27_: _dafny.Seq
                            out27_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e-1'), eosToken)
                            d_32_next2_ = out27_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_32_next2_) == (eosToken):
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_33_cg7_: _dafny.Seq
                                    d_34_ci7_: bool
                                    d_35_cc7_: _dafny.Seq
                                    out28_: _dafny.Seq
                                    out29_: bool
                                    out30_: _dafny.Seq
                                    out28_, out29_, out30_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_33_cg7_ = out28_
                                    d_34_ci7_ = out29_
                                    d_35_cc7_ = out30_
                                    generated = d_33_cg7_
                                    insideConstrainedOut = d_34_ci7_
                                    currentConstrainedOut = d_35_cc7_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                    d_36_valid2_: bool
                                    out31_: bool
                                    out31_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_32_next2_)
                                    d_36_valid2_ = out31_
                                    if d_36_valid2_:
                                        d_37_ag2_: _dafny.Seq
                                        d_38_ai2_: bool
                                        d_39_ac2_: _dafny.Seq
                                        out32_: _dafny.Seq
                                        out33_: bool
                                        out34_: _dafny.Seq
                                        out32_, out33_, out34_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_32_next2_)
                                        d_37_ag2_ = out32_
                                        d_38_ai2_ = out33_
                                        d_39_ac2_ = out34_
                                        generated = d_37_ag2_
                                        insideConstrainedOut = d_38_ai2_
                                        currentConstrainedOut = d_39_ac2_
                                elif True:
                                    if (d_1_steps_) < (maxSteps):
                                        d_40_cg8_: _dafny.Seq
                                        d_41_ci8_: bool
                                        d_42_cc8_: _dafny.Seq
                                        out35_: _dafny.Seq
                                        out36_: bool
                                        out37_: _dafny.Seq
                                        out35_, out36_, out37_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_40_cg8_ = out35_
                                        d_41_ci8_ = out36_
                                        d_42_cc8_ = out37_
                                        generated = d_40_cg8_
                                        insideConstrainedOut = d_41_ci8_
                                        currentConstrainedOut = d_42_cc8_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_43_remaining2_: int
            d_43_remaining2_ = (maxSteps) - (d_1_steps_)
            d_44_csg2_: _dafny.Seq
            d_45_csi2_: bool
            d_46_csc2_: _dafny.Seq
            out38_: _dafny.Seq
            out39_: bool
            out40_: _dafny.Seq
            out38_, out39_, out40_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_43_remaining2_)
            d_44_csg2_ = out38_
            d_45_csi2_ = out39_
            d_46_csc2_ = out40_
            generated = d_44_csg2_
            insideConstrainedOut = d_45_csi2_
            currentConstrainedOut = d_46_csc2_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid novel SMILES for an acrylate ester. The SMILES must contain C=CC(=O)O as the core acryloyl ester group. Generate complete, syntactically valid SMILES with all rings properly closed. Short diverse examples: C=CC(=O)OCC, C=CC(=O)OCCO, C=CC(=O)OC(C)C, C=CC(=O)OCCCC, C=CC(=O)OCC(C)C, C=CC(=O)OCCOCCO. Output ONLY the SMILES string.")))
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
                    d_6_constrainedLen_: int
                    d_6_constrainedLen_ = len(currentConstrainedOut)
                    if (d_5_remaining_) <= (80):
                        d_7_csg_: _dafny.Seq
                        d_8_csi_: bool
                        d_9_csc_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_5_remaining_)
                        d_7_csg_ = out3_
                        d_8_csi_ = out4_
                        d_9_csc_ = out5_
                        generated = d_7_csg_
                        insideConstrainedOut = d_8_csi_
                        currentConstrainedOut = d_9_csc_
                        d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    if (d_6_constrainedLen_) >= (15):
                        d_10_cg_: _dafny.Seq
                        d_11_ci_: bool
                        d_12_cc_: _dafny.Seq
                        d_13_closed_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out9_: bool
                        out6_, out7_, out8_, out9_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_10_cg_ = out6_
                        d_11_ci_ = out7_
                        d_12_cc_ = out8_
                        d_13_closed_ = out9_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_13_closed_:
                            generated = d_10_cg_
                            insideConstrainedOut = d_11_ci_
                            currentConstrainedOut = d_12_cc_
                            raise _dafny.Break("0")
                        d_14_narrow_: bool
                        out10_: bool
                        out10_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                        d_14_narrow_ = out10_
                        if d_14_narrow_:
                            d_15_rg_: _dafny.Seq
                            d_16_rc_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out11_, out12_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_15_rg_ = out11_
                            d_16_rc_ = out12_
                            if ((parser).IsCompletePrefix(d_16_rc_)) and ((len(d_16_rc_)) >= (8)):
                                generated = d_15_rg_
                                currentConstrainedOut = d_16_rc_
                                if (d_1_steps_) < (maxSteps):
                                    d_17_cg2_: _dafny.Seq
                                    d_18_ci2_: bool
                                    d_19_cc2_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_17_cg2_ = out13_
                                    d_18_ci2_ = out14_
                                    d_19_cc2_ = out15_
                                    generated = d_17_cg2_
                                    insideConstrainedOut = d_18_ci2_
                                    currentConstrainedOut = d_19_cc2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            d_20_remaining2_: int
                            d_20_remaining2_ = (maxSteps) - (d_1_steps_)
                            d_21_csg2_: _dafny.Seq
                            d_22_csi2_: bool
                            d_23_csc2_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_20_remaining2_)
                            d_21_csg2_ = out16_
                            d_22_csi2_ = out17_
                            d_23_csc2_ = out18_
                            generated = d_21_csg2_
                            insideConstrainedOut = d_22_csi2_
                            currentConstrainedOut = d_23_csc2_
                            d_1_steps_ = maxSteps
                            raise _dafny.Break("0")
                    d_24_constrainedPrompt_: _dafny.Seq
                    d_24_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (d_6_constrainedLen_):]))
                    d_25_next_: _dafny.Seq
                    out19_: _dafny.Seq
                    out19_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, eosToken)
                    d_25_next_ = out19_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_25_next_) == (eosToken):
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_26_cg3_: _dafny.Seq
                            d_27_ci3_: bool
                            d_28_cc3_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_26_cg3_ = out20_
                            d_27_ci3_ = out21_
                            d_28_cc3_ = out22_
                            generated = d_26_cg3_
                            insideConstrainedOut = d_27_ci3_
                            currentConstrainedOut = d_28_cc3_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_29_rg2_: _dafny.Seq
                            d_30_rc2_: _dafny.Seq
                            out23_: _dafny.Seq
                            out24_: _dafny.Seq
                            out23_, out24_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_29_rg2_ = out23_
                            d_30_rc2_ = out24_
                            if ((parser).IsCompletePrefix(d_30_rc2_)) and ((len(d_30_rc2_)) >= (8)):
                                generated = d_29_rg2_
                                currentConstrainedOut = d_30_rc2_
                                if (d_1_steps_) < (maxSteps):
                                    d_31_cg4_: _dafny.Seq
                                    d_32_ci4_: bool
                                    d_33_cc4_: _dafny.Seq
                                    out25_: _dafny.Seq
                                    out26_: bool
                                    out27_: _dafny.Seq
                                    out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_31_cg4_ = out25_
                                    d_32_ci4_ = out26_
                                    d_33_cc4_ = out27_
                                    generated = d_31_cg4_
                                    insideConstrainedOut = d_32_ci4_
                                    currentConstrainedOut = d_33_cc4_
                                    d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        if not((parser).IsCompletePrefix(currentConstrainedOut)):
                            d_34_valid_: bool
                            out28_: bool
                            out28_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_25_next_)
                            d_34_valid_ = out28_
                            if d_34_valid_:
                                d_35_ag_: _dafny.Seq
                                d_36_ai_: bool
                                d_37_ac_: _dafny.Seq
                                out29_: _dafny.Seq
                                out30_: bool
                                out31_: _dafny.Seq
                                out29_, out30_, out31_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                                d_35_ag_ = out29_
                                d_36_ai_ = out30_
                                d_37_ac_ = out31_
                                generated = d_35_ag_
                                insideConstrainedOut = d_36_ai_
                                currentConstrainedOut = d_37_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_38_remaining3_: int
            d_38_remaining3_ = (maxSteps) - (d_1_steps_)
            d_39_csg3_: _dafny.Seq
            d_40_csi3_: bool
            d_41_csc3_: _dafny.Seq
            out32_: _dafny.Seq
            out33_: bool
            out34_: _dafny.Seq
            out32_, out33_, out34_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_38_remaining3_)
            d_39_csg3_ = out32_
            d_40_csi3_ = out33_
            d_41_csc3_ = out34_
            generated = d_39_csg3_
            insideConstrainedOut = d_40_csi3_
            currentConstrainedOut = d_41_csc3_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


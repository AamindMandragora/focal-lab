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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SMILES string for an acrylate monomer. Acrylates contain the CH2=CH-C(=O)-O- or CH2=C(CH3)-C(=O)-O- group. Example outputs: C=CC(=O)OCC C=CC(=O)OCCC C=C(C)C(=O)OCC C=CC(=O)OC C=CC(=O)OCCCC C=CC(=O)OC(C)C. The answer must be a multi-atom SMILES with vinyl ester group."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_minLength_: int
        d_3_minLength_ = 6
        d_4_vinylGroup_: _dafny.Seq
        d_4_vinylGroup_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))])
        d_5_carbonylGroup_: _dafny.Seq
        d_5_carbonylGroup_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")"))])
        d_6_esterGroup_: _dafny.Seq
        d_6_esterGroup_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))])
        d_7_acrylateGroups_: _dafny.Seq
        d_7_acrylateGroups_ = _dafny.SeqWithoutIsStrInference([d_4_vinylGroup_, d_5_carbonylGroup_, d_6_esterGroup_])
        d_8_combined_: _dafny.Seq
        d_8_combined_ = (validTokenGroups) + (d_7_acrylateGroups_)
        d_9_eosOnlyPenalty_: _dafny.Seq
        d_9_eosOnlyPenalty_ = _dafny.SeqWithoutIsStrInference([eosToken])
        if ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_10_og_: _dafny.Seq
            d_11_oi_: bool
            d_12_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_10_og_ = out0_
            d_11_oi_ = out1_
            d_12_oc_ = out2_
            generated = d_10_og_
            insideConstrainedOut = d_11_oi_
            currentConstrainedOut = d_12_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    d_13_isComplete_: bool
                    d_13_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    d_14_curLen_: int
                    d_14_curLen_ = len(currentConstrainedOut)
                    d_15_lenOk_: bool
                    d_15_lenOk_ = (d_14_curLen_) >= (d_3_minLength_)
                    if (d_13_isComplete_) and (d_15_lenOk_):
                        d_16_cg_: _dafny.Seq
                        d_17_ci_: bool
                        d_18_cc_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_cg_ = out3_
                        d_17_ci_ = out4_
                        d_18_cc_ = out5_
                        generated = d_16_cg_
                        insideConstrainedOut = d_17_ci_
                        currentConstrainedOut = d_18_cc_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    d_19_stableLen_: int
                    d_19_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
                    d_20_constrainedPrompt_: _dafny.Seq
                    d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:d_19_stableLen_:]))
                    d_21_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                    if not(d_15_lenOk_):
                        out6_: _dafny.Seq
                        out6_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, d_8_combined_, _dafny.BigRational('8e0'), d_9_eosOnlyPenalty_, _dafny.BigRational('5e1'), 100, eosToken)
                        d_21_next_ = out6_
                    elif True:
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, d_8_combined_, _dafny.BigRational('4e0'), 20, eosToken)
                        d_21_next_ = out7_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_21_next_) == (eosToken):
                        d_22_rg_: _dafny.Seq
                        d_23_rc_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: _dafny.Seq
                        out8_, out9_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_22_rg_ = out8_
                        d_23_rc_ = out9_
                        generated = d_22_rg_
                        currentConstrainedOut = d_23_rc_
                        d_24_rComplete_: bool
                        d_24_rComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if (d_24_rComplete_) and ((d_2_steps_) < (maxSteps)):
                            d_25_cg_: _dafny.Seq
                            d_26_ci_: bool
                            d_27_cc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_25_cg_ = out10_
                            d_26_ci_ = out11_
                            d_27_cc_ = out12_
                            generated = d_25_cg_
                            insideConstrainedOut = d_26_ci_
                            currentConstrainedOut = d_27_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                        elif not(d_24_rComplete_):
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        raise _dafny.Break("0")
                    elif True:
                        d_28_notComplete_: bool
                        d_28_notComplete_ = not((parser).IsCompletePrefix(currentConstrainedOut))
                        if d_28_notComplete_:
                            d_29_valid_: bool
                            out13_: bool
                            out13_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_21_next_)
                            d_29_valid_ = out13_
                            if d_29_valid_:
                                d_30_ag_: _dafny.Seq
                                d_31_ai_: bool
                                d_32_ac_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                d_30_ag_ = out14_
                                d_31_ai_ = out15_
                                d_32_ac_ = out16_
                                generated = d_30_ag_
                                insideConstrainedOut = d_31_ai_
                                currentConstrainedOut = d_32_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_33_rg_: _dafny.Seq
            d_34_rc_: _dafny.Seq
            out17_: _dafny.Seq
            out18_: _dafny.Seq
            out17_, out18_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_33_rg_ = out17_
            d_34_rc_ = out18_
            generated = d_33_rg_
            currentConstrainedOut = d_34_rc_
            d_35_rComplete_: bool
            d_35_rComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if (d_35_rComplete_) and ((d_2_steps_) < (maxSteps)):
                d_36_cg_: _dafny.Seq
                d_37_ci_: bool
                d_38_cc_: _dafny.Seq
                out19_: _dafny.Seq
                out20_: bool
                out21_: _dafny.Seq
                out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_36_cg_ = out19_
                d_37_ci_ = out20_
                d_38_cc_ = out21_
                generated = d_36_cg_
                insideConstrainedOut = d_37_ci_
                currentConstrainedOut = d_38_cc_
                d_2_steps_ = (d_2_steps_) + (1)
            elif not(d_35_rComplete_):
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


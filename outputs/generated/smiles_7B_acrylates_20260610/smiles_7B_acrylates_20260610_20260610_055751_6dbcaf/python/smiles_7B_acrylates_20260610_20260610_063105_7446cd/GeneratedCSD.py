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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a SMILES string for an acrylate molecule. The acrylate MUST contain the acryloyl group: a vinyl group connected to an ester. Valid examples: C=CC(=O)OCC, C=CC(=O)OCCC, C=CC(=O)OC, C=C(C)C(=O)OCC. Start with bracket atom or multi-char prefix to avoid trivial single-atom completions."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_acrylateTokens_: _dafny.Seq
        d_3_acrylateTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "c")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "S")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "s")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "o"))])
        d_4_vinylTokens_: _dafny.Seq
        d_4_vinylTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "("))])
        d_5_esterTokens_: _dafny.Seq
        d_5_esterTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")"))])
        d_6_acrylateGroups_: _dafny.Seq
        d_6_acrylateGroups_ = _dafny.SeqWithoutIsStrInference([d_3_acrylateTokens_, d_4_vinylTokens_, d_5_esterTokens_])
        d_7_minLength_: int
        d_7_minLength_ = 8
        if ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_8_og_: _dafny.Seq
            d_9_oi_: bool
            d_10_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_8_og_ = out0_
            d_9_oi_ = out1_
            d_10_oc_ = out2_
            generated = d_8_og_
            insideConstrainedOut = d_9_oi_
            currentConstrainedOut = d_10_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    d_11_isComplete_: bool
                    d_11_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    d_12_lenOk_: bool
                    d_12_lenOk_ = (len(currentConstrainedOut)) >= (d_7_minLength_)
                    if (d_11_isComplete_) and (d_12_lenOk_):
                        d_13_cg_: _dafny.Seq
                        d_14_ci_: bool
                        d_15_cc_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_cg_ = out3_
                        d_14_ci_ = out4_
                        d_15_cc_ = out5_
                        generated = d_13_cg_
                        insideConstrainedOut = d_14_ci_
                        currentConstrainedOut = d_15_cc_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    d_16_stableLen_: int
                    d_16_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
                    d_17_constrainedPrompt_: _dafny.Seq
                    d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:d_16_stableLen_:]))
                    d_18_combined_: _dafny.Seq
                    d_18_combined_ = (validTokenGroups) + (d_6_acrylateGroups_)
                    d_19_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                    d_20_tokenCount_: int
                    d_20_tokenCount_ = len(currentConstrainedOut)
                    if (d_20_tokenCount_) == (0):
                        d_21_startGroups_: _dafny.Seq
                        d_21_startGroups_ = (d_18_combined_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))])]))
                        out6_: _dafny.Seq
                        out6_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, d_21_startGroups_, _dafny.BigRational('1e1'), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "S")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "F")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "B")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "P")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "I")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "[")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "c")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "o")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "s"))]), _dafny.BigRational('6e0'), 100, eosToken)
                        d_19_next_ = out6_
                    elif (d_20_tokenCount_) < (d_7_minLength_):
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, d_18_combined_, _dafny.BigRational('6e0'), 50, eosToken)
                        d_19_next_ = out7_
                    elif True:
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                        d_19_next_ = out8_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_19_next_) == (eosToken):
                        d_22_rg_: _dafny.Seq
                        d_23_rc_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: _dafny.Seq
                        out9_, out10_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_22_rg_ = out9_
                        d_23_rc_ = out10_
                        generated = d_22_rg_
                        currentConstrainedOut = d_23_rc_
                        d_24_rComplete_: bool
                        d_24_rComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if (d_24_rComplete_) and ((d_2_steps_) < (maxSteps)):
                            d_25_cg_: _dafny.Seq
                            d_26_ci_: bool
                            d_27_cc_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_25_cg_ = out11_
                            d_26_ci_ = out12_
                            d_27_cc_ = out13_
                            generated = d_25_cg_
                            insideConstrainedOut = d_26_ci_
                            currentConstrainedOut = d_27_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_28_curComplete_: bool
                        d_28_curComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_28_curComplete_:
                            if (d_2_steps_) < (maxSteps):
                                d_29_cg_: _dafny.Seq
                                d_30_ci_: bool
                                d_31_cc_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_29_cg_ = out14_
                                d_30_ci_ = out15_
                                d_31_cc_ = out16_
                                generated = d_29_cg_
                                insideConstrainedOut = d_30_ci_
                                currentConstrainedOut = d_31_cc_
                                d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_32_valid_: bool
                            out17_: bool
                            out17_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_19_next_)
                            d_32_valid_ = out17_
                            if d_32_valid_:
                                d_33_ag_: _dafny.Seq
                                d_34_ai_: bool
                                d_35_ac_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                d_33_ag_ = out18_
                                d_34_ai_ = out19_
                                d_35_ac_ = out20_
                                generated = d_33_ag_
                                insideConstrainedOut = d_34_ai_
                                currentConstrainedOut = d_35_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_36_rg_: _dafny.Seq
            d_37_rc_: _dafny.Seq
            out21_: _dafny.Seq
            out22_: _dafny.Seq
            out21_, out22_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_36_rg_ = out21_
            d_37_rc_ = out22_
            generated = d_36_rg_
            currentConstrainedOut = d_37_rc_
            d_38_rComplete_: bool
            d_38_rComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if (d_38_rComplete_) and ((d_2_steps_) < (maxSteps)):
                d_39_cg_: _dafny.Seq
                d_40_ci_: bool
                d_41_cc_: _dafny.Seq
                out23_: _dafny.Seq
                out24_: bool
                out25_: _dafny.Seq
                out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_39_cg_ = out23_
                d_40_ci_ = out24_
                d_41_cc_ = out25_
                generated = d_39_cg_
                insideConstrainedOut = d_40_ci_
                currentConstrainedOut = d_41_cc_
                d_2_steps_ = (d_2_steps_) + (1)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


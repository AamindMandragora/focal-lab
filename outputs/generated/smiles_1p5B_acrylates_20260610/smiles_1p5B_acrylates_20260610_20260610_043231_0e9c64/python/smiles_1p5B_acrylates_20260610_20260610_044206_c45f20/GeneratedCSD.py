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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a complete SMILES string for a novel acrylate ester molecule. Acrylates have the vinyl acrylate scaffold CH2=CH-C(=O)-O-R where R is an alkyl or substituted group. Example scaffolds: C=CC(=O)OCC, C=CC(=O)OCCC, C=CC(=O)OC(C)C. Output a full multi-atom SMILES with at least 8 heavy atoms. Do not output just C alone.")))
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
        d_5_minConstrainedTokens_: int
        d_5_minConstrainedTokens_ = 10
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_5_minConstrainedTokens_)):
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
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_10_next_: _dafny.Seq
                        out6_: _dafny.Seq
                        out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_10_next_ = out6_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_10_next_) == (eosToken):
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_11_cg_: _dafny.Seq
                                d_12_ci_: bool
                                d_13_cc_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_11_cg_ = out7_
                                d_12_ci_ = out8_
                                d_13_cc_ = out9_
                                generated = d_11_cg_
                                insideConstrainedOut = d_12_ci_
                                currentConstrainedOut = d_13_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_14_valid_: bool
                            out10_: bool
                            out10_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_10_next_)
                            d_14_valid_ = out10_
                            if d_14_valid_:
                                if (d_1_steps_) < (maxSteps):
                                    d_15_cg_: _dafny.Seq
                                    d_16_ci_: bool
                                    d_17_cc_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_15_cg_ = out11_
                                    d_16_ci_ = out12_
                                    d_17_cc_ = out13_
                                    generated = d_15_cg_
                                    insideConstrainedOut = d_16_ci_
                                    currentConstrainedOut = d_17_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                if (d_1_steps_) < (maxSteps):
                                    d_18_cg_: _dafny.Seq
                                    d_19_ci_: bool
                                    d_20_cc_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_18_cg_ = out14_
                                    d_19_ci_ = out15_
                                    d_20_cc_ = out16_
                                    generated = d_18_cg_
                                    insideConstrainedOut = d_19_ci_
                                    currentConstrainedOut = d_20_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                    elif True:
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_22_next_: _dafny.Seq
                        out17_: _dafny.Seq
                        out17_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_22_next_ = out17_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_22_next_) == (eosToken):
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_23_cg_: _dafny.Seq
                                d_24_ci_: bool
                                d_25_cc_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_23_cg_ = out18_
                                d_24_ci_ = out19_
                                d_25_cc_ = out20_
                                generated = d_23_cg_
                                insideConstrainedOut = d_24_ci_
                                currentConstrainedOut = d_25_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_26_ag_: _dafny.Seq
                            d_27_ai_: bool
                            d_28_ac_: _dafny.Seq
                            out21_: _dafny.Seq
                            out22_: bool
                            out23_: _dafny.Seq
                            out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                            d_26_ag_ = out21_
                            d_27_ai_ = out22_
                            d_28_ac_ = out23_
                            generated = d_26_ag_
                            insideConstrainedOut = d_27_ai_
                            currentConstrainedOut = d_28_ac_
                            if (((d_1_steps_) < (maxSteps)) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) >= (d_5_minConstrainedTokens_)):
                                d_29_cg2_: _dafny.Seq
                                d_30_ci2_: bool
                                d_31_cc2_: _dafny.Seq
                                d_32_closed_: bool
                                out24_: _dafny.Seq
                                out25_: bool
                                out26_: _dafny.Seq
                                out27_: bool
                                out24_, out25_, out26_, out27_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_29_cg2_ = out24_
                                d_30_ci2_ = out25_
                                d_31_cc2_ = out26_
                                d_32_closed_ = out27_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if d_32_closed_:
                                    generated = d_29_cg2_
                                    insideConstrainedOut = d_30_ci2_
                                    currentConstrainedOut = d_31_cc2_
                                    raise _dafny.Break("0")
                    pass
            pass
        if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
            d_33_cg_: _dafny.Seq
            d_34_ci_: bool
            d_35_cc_: _dafny.Seq
            out28_: _dafny.Seq
            out29_: bool
            out30_: _dafny.Seq
            out28_, out29_, out30_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_33_cg_ = out28_
            d_34_ci_ = out29_
            d_35_cc_ = out30_
            generated = d_33_cg_
            insideConstrainedOut = d_34_ci_
            currentConstrainedOut = d_35_cc_
            d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


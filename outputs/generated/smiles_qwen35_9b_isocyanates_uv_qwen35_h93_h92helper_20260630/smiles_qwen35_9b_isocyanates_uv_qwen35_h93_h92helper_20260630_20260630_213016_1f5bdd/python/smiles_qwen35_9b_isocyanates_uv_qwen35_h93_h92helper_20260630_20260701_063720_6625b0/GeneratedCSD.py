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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a unique and novel isocyanate SMILES. REQUIRED: The SMILES must contain exactly one N=C=O group. REQUIRED: Include a non-trivial organic substituent before N. REQUIRED: Output must be a valid SMILES with N=C=O. Do NOT generate CO2 (O=C=O). Do NOT generate carbodiimide. Good examples: CCN=C=O, CCCN=C=O, CC(C)N=C=O, c1ccccc1N=C=O, CCCCN=C=O, CCC(C)N=C=O, CC(CC)N=C=O, c1ccc(N=C=O)cc1, OCCCN=C=O, ClCCN=C=O. Generate a UNIQUE molecule not in the examples above.")))
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
        d_5_stepCap_: int
        d_5_stepCap_ = 50
        if (maxSteps) < (d_5_stepCap_):
            d_5_stepCap_ = maxSteps
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_5_stepCap_)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_6_currentStr_: _dafny.Seq
                    d_6_currentStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                    d_7_hasIsocyanate_: bool
                    d_7_hasIsocyanate_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_6_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))) > (0)
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and (d_7_hasIsocyanate_):
                        d_8_cg_: _dafny.Seq
                        d_9_ci_: bool
                        d_10_cc_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_8_cg_ = out3_
                        d_9_ci_ = out4_
                        d_10_cc_ = out5_
                        generated = d_8_cg_
                        insideConstrainedOut = d_9_ci_
                        currentConstrainedOut = d_10_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif ((parser).IsCompletePrefix(currentConstrainedOut)) and (not(d_7_hasIsocyanate_)):
                        d_11_closeBudget_: int
                        d_11_closeBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_11_closeBudget_) > (0):
                            d_12_cg_: _dafny.Seq
                            d_13_ci_: bool
                            d_14_cc_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_closeBudget_)
                            d_12_cg_ = out6_
                            d_13_ci_ = out7_
                            d_14_cc_ = out8_
                            generated = d_12_cg_
                            insideConstrainedOut = d_13_ci_
                            currentConstrainedOut = d_14_cc_
                            d_1_steps_ = maxSteps
                        elif True:
                            d_15_cg_: _dafny.Seq
                            d_16_ci_: bool
                            d_17_cc_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_15_cg_ = out9_
                            d_16_ci_ = out10_
                            d_17_cc_ = out11_
                            generated = d_15_cg_
                            insideConstrainedOut = d_16_ci_
                            currentConstrainedOut = d_17_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_hasN_: bool
                        d_19_hasN_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_6_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N")))) > (0)
                        if not(d_19_hasN_):
                            d_20_nValid_: bool
                            out12_: bool
                            out12_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N")))
                            d_20_nValid_ = out12_
                            if (d_20_nValid_) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N"))) in ((lm).Tokens)):
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N"))]), _dafny.BigRational('5e0'))
                        if d_7_hasIsocyanate_:
                            d_21_dotValid_: bool
                            out13_: bool
                            out13_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")))
                            d_21_dotValid_ = out13_
                            d_22_next_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).TemperatureConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('3e-1'), eosToken)
                            d_22_next_ = out14_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_23_valid_: bool
                                out15_: bool
                                out15_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_22_next_)
                                d_23_valid_ = out15_
                                if (d_23_valid_) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                                    d_24_ag_: _dafny.Seq
                                    d_25_ai_: bool
                                    d_26_ac_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                    d_24_ag_ = out16_
                                    d_25_ai_ = out17_
                                    d_26_ac_ = out18_
                                    generated = d_24_ag_
                                    insideConstrainedOut = d_25_ai_
                                    currentConstrainedOut = d_26_ac_
                        elif True:
                            d_27_next_: _dafny.Seq
                            out19_: _dafny.Seq
                            out19_ = (d_0_helpers_).TemperatureConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('12e-1'), eosToken)
                            d_27_next_ = out19_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_27_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_28_valid_: bool
                                out20_: bool
                                out20_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_27_next_)
                                d_28_valid_ = out20_
                                if (d_28_valid_) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                                    d_29_ag_: _dafny.Seq
                                    d_30_ai_: bool
                                    d_31_ac_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out22_: bool
                                    out23_: _dafny.Seq
                                    out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                                    d_29_ag_ = out21_
                                    d_30_ai_ = out22_
                                    d_31_ac_ = out23_
                                    generated = d_29_ag_
                                    insideConstrainedOut = d_30_ai_
                                    currentConstrainedOut = d_31_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_32_closeBudget_: int
            d_32_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_33_cg_: _dafny.Seq
            d_34_ci_: bool
            d_35_cc_: _dafny.Seq
            out24_: _dafny.Seq
            out25_: bool
            out26_: _dafny.Seq
            out24_, out25_, out26_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_32_closeBudget_)
            d_33_cg_ = out24_
            d_34_ci_ = out25_
            d_35_cc_ = out26_
            generated = d_33_cg_
            insideConstrainedOut = d_34_ci_
            currentConstrainedOut = d_35_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


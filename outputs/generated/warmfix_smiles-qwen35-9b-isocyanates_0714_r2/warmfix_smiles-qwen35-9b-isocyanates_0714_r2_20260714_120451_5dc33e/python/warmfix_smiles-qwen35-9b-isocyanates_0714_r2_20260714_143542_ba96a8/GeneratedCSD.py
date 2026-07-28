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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SMILES for an isocyanate molecule. The molecule MUST contain the N=C=O fragment. Generate diverse structures: aliphatic chains, rings, halogenated, branched. Valid examples: CCN=C=O, O=C=Nc1ccccc1, O=C=NCCCC, BrCCN=C=O, O=C=NCC(C)C, O=C=NCc1ccccc1, ClCCN=C=O, O=C=NC1CCCCC1. Pick a structure different from these examples.")))
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
        d_5_minSpanLength_: int
        d_5_minSpanLength_ = 5
        d_6_tokenCount_: int
        d_6_tokenCount_ = 0
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_5_minSpanLength_)):
                        d_7_smilesStr_: _dafny.Seq
                        d_7_smilesStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                        d_8_hasNCO_: int
                        d_8_hasNCO_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_7_smilesStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))
                        if (d_8_hasNCO_) > (0):
                            if (d_1_steps_) < (maxSteps):
                                d_9_cg_: _dafny.Seq
                                d_10_ci_: bool
                                d_11_cc_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_9_cg_ = out3_
                                d_10_ci_ = out4_
                                d_11_cc_ = out5_
                                generated = d_9_cg_
                                insideConstrainedOut = d_10_ci_
                                currentConstrainedOut = d_11_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_12_constrainedPrompt_: _dafny.Seq
                            d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_13_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('18e-1'), eosToken)
                            d_13_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_13_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_14_ag_: _dafny.Seq
                                d_15_ai_: bool
                                d_16_ac_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                d_14_ag_ = out7_
                                d_15_ai_ = out8_
                                d_16_ac_ = out9_
                                generated = d_14_ag_
                                insideConstrainedOut = d_15_ai_
                                currentConstrainedOut = d_16_ac_
                                d_6_tokenCount_ = (d_6_tokenCount_) + (1)
                    elif True:
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_18_next_: _dafny.Seq
                        d_18_next_ = eosToken
                        if (d_6_tokenCount_) < (5):
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('18e-1'), eosToken)
                            d_18_next_ = out10_
                        elif (d_6_tokenCount_) < (15):
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('15e-1'), eosToken)
                            d_18_next_ = out11_
                        elif True:
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_18_next_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_18_next_) == (eosToken):
                            if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_5_minSpanLength_))) and ((d_1_steps_) < (maxSteps)):
                                d_19_smilesStr_: _dafny.Seq
                                d_19_smilesStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                                d_20_hasNCO_: int
                                d_20_hasNCO_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_19_smilesStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))
                                if (d_20_hasNCO_) > (0):
                                    d_21_cg_: _dafny.Seq
                                    d_22_ci_: bool
                                    d_23_cc_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_21_cg_ = out13_
                                    d_22_ci_ = out14_
                                    d_23_cc_ = out15_
                                    generated = d_21_cg_
                                    insideConstrainedOut = d_22_ci_
                                    currentConstrainedOut = d_23_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_24_ag_: _dafny.Seq
                            d_25_ai_: bool
                            d_26_ac_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                            d_24_ag_ = out16_
                            d_25_ai_ = out17_
                            d_26_ac_ = out18_
                            generated = d_24_ag_
                            insideConstrainedOut = d_25_ai_
                            currentConstrainedOut = d_26_ac_
                            d_6_tokenCount_ = (d_6_tokenCount_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_27_closeBudget_: int
            d_27_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_28_cg_: _dafny.Seq
            d_29_ci_: bool
            d_30_cc_: _dafny.Seq
            out19_: _dafny.Seq
            out20_: bool
            out21_: _dafny.Seq
            out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_27_closeBudget_)
            d_28_cg_ = out19_
            d_29_ci_ = out20_
            d_30_cc_ = out21_
            generated = d_28_cg_
            insideConstrainedOut = d_29_ci_
            currentConstrainedOut = d_30_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


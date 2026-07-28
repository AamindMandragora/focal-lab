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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate ONE valid SMILES for a chain_extender molecule. Chain extenders are SMALL bifunctional molecules with 2-8 carbons: diols like OCCO, OCCCCO, OCCCCCO; diamines like NCCN, NCCCN; amino alcohols like NCCO, NCCCO. Output ONLY the SMILES string, nothing else."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_preambleMax_: int
        d_3_preambleMax_ = 20
        d_4_preambleSteps_: int
        d_4_preambleSteps_ = 0
        with _dafny.label("0"):
            while (((d_2_steps_) < (maxSteps)) and ((d_4_preambleSteps_) < (d_3_preambleMax_))) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_5_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_5_next_ = out0_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_4_preambleSteps_ = (d_4_preambleSteps_) + (1)
                    if (d_5_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                    if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_6_og_: _dafny.Seq
                        d_7_oi_: bool
                        d_8_oc_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_6_og_ = out1_
                        d_7_oi_ = out2_
                        d_8_oc_ = out3_
                        generated = d_6_og_
                        insideConstrainedOut = d_7_oi_
                        currentConstrainedOut = d_8_oc_
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_9_og_: _dafny.Seq
            d_10_oi_: bool
            d_11_oc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_9_og_ = out4_
            d_10_oi_ = out5_
            d_11_oc_ = out6_
            generated = d_9_og_
            insideConstrainedOut = d_10_oi_
            currentConstrainedOut = d_11_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        with _dafny.label("1"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("1"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("1")
                    elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (2)):
                        d_12_cg_: _dafny.Seq
                        d_13_ci_: bool
                        d_14_cc_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_cg_ = out7_
                        d_13_ci_ = out8_
                        d_14_cc_ = out9_
                        generated = d_12_cg_
                        insideConstrainedOut = d_13_ci_
                        currentConstrainedOut = d_14_cc_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("1")
                    elif True:
                        d_15_isComplete_: bool
                        d_15_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_15_isComplete_:
                            d_16_cg_: _dafny.Seq
                            d_17_ci_: bool
                            d_18_cc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_16_cg_ = out10_
                            d_17_ci_ = out11_
                            d_18_cc_ = out12_
                            generated = d_16_cg_
                            insideConstrainedOut = d_17_ci_
                            currentConstrainedOut = d_18_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("1")
                        elif True:
                            d_19_constrainedPrompt_: _dafny.Seq
                            d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_20_validCount_: int
                            out13_: int
                            out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_20_validCount_ = out13_
                            d_21_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            if (d_20_validCount_) <= (6):
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), eosToken)
                                d_21_next_ = out14_
                            elif (d_20_validCount_) <= (15):
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 15, eosToken)
                                d_21_next_ = out15_
                            elif True:
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('7e-1'), eosToken)
                                d_21_next_ = out16_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_21_next_) == (eosToken):
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                    d_22_cg_: _dafny.Seq
                                    d_23_ci_: bool
                                    d_24_cc_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_22_cg_ = out17_
                                    d_23_ci_ = out18_
                                    d_24_cc_ = out19_
                                    generated = d_22_cg_
                                    insideConstrainedOut = d_23_ci_
                                    currentConstrainedOut = d_24_cc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                raise _dafny.Break("1")
                            elif True:
                                d_25_completeCheck_: bool
                                d_25_completeCheck_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_25_completeCheck_:
                                    d_26_cg_: _dafny.Seq
                                    d_27_ci_: bool
                                    d_28_cc_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_26_cg_ = out20_
                                    d_27_ci_ = out21_
                                    d_28_cc_ = out22_
                                    generated = d_26_cg_
                                    insideConstrainedOut = d_27_ci_
                                    currentConstrainedOut = d_28_cc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    raise _dafny.Break("1")
                                elif True:
                                    d_29_ag_: _dafny.Seq
                                    d_30_ai_: bool
                                    d_31_ac_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out24_: bool
                                    out25_: _dafny.Seq
                                    out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                    d_29_ag_ = out23_
                                    d_30_ai_ = out24_
                                    d_31_ac_ = out25_
                                    generated = d_29_ag_
                                    insideConstrainedOut = d_30_ai_
                                    currentConstrainedOut = d_31_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


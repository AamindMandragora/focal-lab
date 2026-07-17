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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Task: output exactly one SMILES string for an isocyanate molecule. Isocyanates must contain the N=C=O functional group. Start the SMILES with the isocyanate nitrogen: begin with O=C=N and then add a substituent R group (alkyl or aryl). Valid examples: O=C=NC, O=C=NCC, O=C=NCCCl, O=C=Nc1ccccc1. Do NOT output a single atom. Output at least 5 atoms.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_minPhaseTokens_: int
        d_2_minPhaseTokens_ = 5
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
        d_6_phaseCount_: int
        d_6_phaseCount_ = 0
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((d_6_phaseCount_) < (d_2_minPhaseTokens_)):
                with _dafny.c_label("0"):
                    d_7_isComp_: bool
                    d_7_isComp_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if d_7_isComp_:
                        if (d_1_steps_) < (maxSteps):
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
                        raise _dafny.Break("0")
                    elif True:
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_12_next_: _dafny.Seq
                        out6_: _dafny.Seq
                        out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                        d_12_next_ = out6_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_12_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_13_isCompCheck_: bool
                            d_13_isCompCheck_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if not(d_13_isCompCheck_):
                                d_14_ag_: _dafny.Seq
                                d_15_ai_: bool
                                d_16_ac_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                d_14_ag_ = out7_
                                d_15_ai_ = out8_
                                d_16_ac_ = out9_
                                generated = d_14_ag_
                                insideConstrainedOut = d_15_ai_
                                currentConstrainedOut = d_16_ac_
                                d_6_phaseCount_ = (d_6_phaseCount_) + (1)
                            elif True:
                                if (d_1_steps_) < (maxSteps):
                                    d_17_cg2_: _dafny.Seq
                                    d_18_ci2_: bool
                                    d_19_cc2_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_17_cg2_ = out10_
                                    d_18_ci2_ = out11_
                                    d_19_cc2_ = out12_
                                    generated = d_17_cg2_
                                    insideConstrainedOut = d_18_ci2_
                                    currentConstrainedOut = d_19_cc2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                    pass
            pass
        with _dafny.label("1"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    d_20_cg_: _dafny.Seq
                    d_21_ci_: bool
                    d_22_cc_: _dafny.Seq
                    d_23_closed_: bool
                    out13_: _dafny.Seq
                    out14_: bool
                    out15_: _dafny.Seq
                    out16_: bool
                    out13_, out14_, out15_, out16_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_20_cg_ = out13_
                    d_21_ci_ = out14_
                    d_22_cc_ = out15_
                    d_23_closed_ = out16_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_23_closed_:
                        generated = d_20_cg_
                        insideConstrainedOut = d_21_ci_
                        currentConstrainedOut = d_22_cc_
                        raise _dafny.Break("1")
                    elif True:
                        if (d_1_steps_) < (maxSteps):
                            d_24_isComp2_: bool
                            d_24_isComp2_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_24_isComp2_:
                                raise _dafny.Break("1")
                            elif True:
                                d_25_constrainedPrompt2_: _dafny.Seq
                                d_25_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_26_next2_: _dafny.Seq
                                out17_: _dafny.Seq
                                out17_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_25_constrainedPrompt2_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_26_next2_ = out17_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_26_next2_) == (eosToken):
                                    raise _dafny.Break("1")
                                elif True:
                                    d_27_isComp3_: bool
                                    d_27_isComp3_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if not(d_27_isComp3_):
                                        d_28_ag2_: _dafny.Seq
                                        d_29_ai2_: bool
                                        d_30_ac2_: _dafny.Seq
                                        out18_: _dafny.Seq
                                        out19_: bool
                                        out20_: _dafny.Seq
                                        out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next2_)
                                        d_28_ag2_ = out18_
                                        d_29_ai2_ = out19_
                                        d_30_ac2_ = out20_
                                        generated = d_28_ag2_
                                        insideConstrainedOut = d_29_ai2_
                                        currentConstrainedOut = d_30_ac2_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_31_isComp_: bool
            d_31_isComp_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_31_isComp_:
                d_32_cg_: _dafny.Seq
                d_33_ci_: bool
                d_34_cc_: _dafny.Seq
                out21_: _dafny.Seq
                out22_: bool
                out23_: _dafny.Seq
                out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_32_cg_ = out21_
                d_33_ci_ = out22_
                d_34_cc_ = out23_
                generated = d_32_cg_
                insideConstrainedOut = d_33_ci_
                currentConstrainedOut = d_34_cc_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


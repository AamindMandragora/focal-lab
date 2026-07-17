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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are generating a SMILES string for an isocyanate molecule. Isocyanates contain the functional group -N=C=O. The SMILES must include N=C=O. Examples of valid isocyanate SMILES: O=C=NCC, O=C=NCCCl, O=C=NCCO, O=C=Nc1ccc(F)cc1, O=C=NCC(C)C. Generate a novel isocyanate SMILES string now.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if not(insideConstrainedOut):
            if (d_1_steps_) < (maxSteps):
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
                    d_5_cg_: _dafny.Seq
                    d_6_ci_: bool
                    d_7_cc_: _dafny.Seq
                    d_8_closed_: bool
                    out3_: _dafny.Seq
                    out4_: bool
                    out5_: _dafny.Seq
                    out6_: bool
                    out3_, out4_, out5_, out6_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_5_cg_ = out3_
                    d_6_ci_ = out4_
                    d_7_cc_ = out5_
                    d_8_closed_ = out6_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_8_closed_:
                        generated = d_5_cg_
                        insideConstrainedOut = d_6_ci_
                        currentConstrainedOut = d_7_cc_
                        raise _dafny.Break("0")
                    if ((d_1_steps_) + (2)) > (maxSteps):
                        d_9_rg_: _dafny.Seq
                        d_10_rc_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: _dafny.Seq
                        out7_, out8_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_9_rg_ = out7_
                        d_10_rc_ = out8_
                        generated = d_9_rg_
                        currentConstrainedOut = d_10_rc_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_11_cg2_: _dafny.Seq
                            d_12_ci2_: bool
                            d_13_cc2_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_cg2_ = out9_
                            d_12_ci2_ = out10_
                            d_13_cc2_ = out11_
                            generated = d_11_cg2_
                            insideConstrainedOut = d_12_ci2_
                            currentConstrainedOut = d_13_cc2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    d_14_stableLen_: int
                    d_14_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
                    d_15_constrainedPrompt_: _dafny.Seq
                    d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:d_14_stableLen_:]))
                    d_16_next_: _dafny.Seq
                    out12_: _dafny.Seq
                    out12_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                    d_16_next_ = out12_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_16_next_) == (eosToken):
                        d_17_rg_: _dafny.Seq
                        d_18_rc_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_17_rg_ = out13_
                        d_18_rc_ = out14_
                        generated = d_17_rg_
                        currentConstrainedOut = d_18_rc_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_19_cg2_: _dafny.Seq
                            d_20_ci2_: bool
                            d_21_cc2_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_19_cg2_ = out15_
                            d_20_ci2_ = out16_
                            d_21_cc2_ = out17_
                            generated = d_19_cg2_
                            insideConstrainedOut = d_20_ci2_
                            currentConstrainedOut = d_21_cc2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_22_ag_: _dafny.Seq
                        d_23_ai_: bool
                        d_24_ac_: _dafny.Seq
                        out18_: _dafny.Seq
                        out19_: bool
                        out20_: _dafny.Seq
                        out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                        d_22_ag_ = out18_
                        d_23_ai_ = out19_
                        d_24_ac_ = out20_
                        generated = d_22_ag_
                        insideConstrainedOut = d_23_ai_
                        currentConstrainedOut = d_24_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


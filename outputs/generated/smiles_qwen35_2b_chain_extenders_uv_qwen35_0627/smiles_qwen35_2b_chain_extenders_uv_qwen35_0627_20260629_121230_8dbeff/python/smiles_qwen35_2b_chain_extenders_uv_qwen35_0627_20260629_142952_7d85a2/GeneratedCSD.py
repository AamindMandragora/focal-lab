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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate ONE valid SMILES for a chain_extender molecule. Chain extenders are bifunctional molecules: diols (OCCO, OCCCCO, OCCCCCO, OCC(O)CCO), diamines (NCCN, NCCCN, NCCCCN), amino alcohols (NCCO, NCCCO, NCCCCO), or similar. The SMILES must have at least 2 functional groups and at least 4 heavy atoms. Output the SMILES string only."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_preambleMax_: int
        d_3_preambleMax_ = 10
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
        d_12_constrainedBudget_: int
        d_12_constrainedBudget_ = 50
        d_13_constrainedSteps_: int
        d_13_constrainedSteps_ = 0
        with _dafny.label("1"):
            while (((d_2_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((d_13_constrainedSteps_) < (d_12_constrainedBudget_)):
                with _dafny.c_label("1"):
                    d_14_cg_: _dafny.Seq
                    d_15_ci_: bool
                    d_16_cc_: _dafny.Seq
                    d_17_closed_: bool
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out10_: bool
                    out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_14_cg_ = out7_
                    d_15_ci_ = out8_
                    d_16_cc_ = out9_
                    d_17_closed_ = out10_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_13_constrainedSteps_ = (d_13_constrainedSteps_) + (1)
                    if d_17_closed_:
                        generated = d_14_cg_
                        insideConstrainedOut = d_15_ci_
                        currentConstrainedOut = d_16_cc_
                    elif True:
                        if ((d_2_steps_) < (maxSteps)) and ((d_13_constrainedSteps_) < (d_12_constrainedBudget_)):
                            d_18_constrainedPrompt_: _dafny.Seq
                            d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_19_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_19_next_ = out11_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_13_constrainedSteps_ = (d_13_constrainedSteps_) + (1)
                            if (d_19_next_) == (eosToken):
                                raise _dafny.Break("1")
                            elif True:
                                d_20_valid_: bool
                                out12_: bool
                                out12_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_19_next_)
                                d_20_valid_ = out12_
                                if d_20_valid_:
                                    d_21_ag_: _dafny.Seq
                                    d_22_ai_: bool
                                    d_23_ac_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                    d_21_ag_ = out13_
                                    d_22_ai_ = out14_
                                    d_23_ac_ = out15_
                                    generated = d_21_ag_
                                    insideConstrainedOut = d_22_ai_
                                    currentConstrainedOut = d_23_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_24_remaining_: int
            d_24_remaining_ = (maxSteps) - (d_2_steps_)
            d_25_closeBudget_: int = int(0)
            if (d_24_remaining_) < (30):
                d_25_closeBudget_ = d_24_remaining_
            elif True:
                d_25_closeBudget_ = 30
            if (d_25_closeBudget_) > (0):
                d_26_cg_: _dafny.Seq
                d_27_ci_: bool
                d_28_cc_: _dafny.Seq
                out16_: _dafny.Seq
                out17_: bool
                out18_: _dafny.Seq
                out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_25_closeBudget_)
                d_26_cg_ = out16_
                d_27_ci_ = out17_
                d_28_cc_ = out18_
                generated = d_26_cg_
                insideConstrainedOut = d_27_ci_
                currentConstrainedOut = d_28_cc_
                d_2_steps_ = (d_2_steps_) + (d_25_closeBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


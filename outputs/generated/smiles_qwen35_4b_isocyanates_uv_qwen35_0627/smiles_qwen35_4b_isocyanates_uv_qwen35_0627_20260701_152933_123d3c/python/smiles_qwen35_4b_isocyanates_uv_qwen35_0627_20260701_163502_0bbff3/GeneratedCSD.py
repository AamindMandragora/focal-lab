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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a SMILES string for an organic isocyanate molecule. Isocyanates contain the functional group -N=C=O. Start with a carbon chain, then add N=C=O. Good examples: CN=C=O, CCN=C=O, CCCN=C=O, CC(C)N=C=O, c1ccccc1N=C=O, CCCCN=C=O, ClCCN=C=O, BrCCN=C=O. The SMILES must contain N=C=O as a substructure. Output ONLY the SMILES, nothing else."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
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
            d_2_steps_ = (d_2_steps_) + (1)
        d_6_forcedSteps_: int
        d_6_forcedSteps_ = 0
        d_7_minForced_: int
        d_7_minForced_ = 6
        with _dafny.label("0"):
            while (((d_2_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((d_6_forcedSteps_) < (d_7_minForced_)):
                with _dafny.c_label("0"):
                    d_8_constrainedPrompt_: _dafny.Seq
                    d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_9_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                    if (d_6_forcedSteps_) == (0):
                        d_10_boostTokens_: _dafny.Seq
                        d_10_boostTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))])
                        d_11_penaltyTokens_: _dafny.Seq
                        d_11_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "S")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "P")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "F")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "I"))])
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, d_10_boostTokens_, _dafny.BigRational('8e0'), eosToken)
                        d_9_next_ = out3_
                    elif True:
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                        d_9_next_ = out4_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_6_forcedSteps_ = (d_6_forcedSteps_) + (1)
                    if (d_9_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        d_12_isComplete_: bool
                        d_12_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_12_isComplete_:
                            raise _dafny.Break("0")
                        elif True:
                            d_13_valid_: bool
                            out5_: bool
                            out5_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_9_next_)
                            d_13_valid_ = out5_
                            if d_13_valid_:
                                d_14_ag_: _dafny.Seq
                                d_15_ai_: bool
                                d_16_ac_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                                d_14_ag_ = out6_
                                d_15_ai_ = out7_
                                d_16_ac_ = out8_
                                generated = d_14_ag_
                                insideConstrainedOut = d_15_ai_
                                currentConstrainedOut = d_16_ac_
                    pass
            pass
        with _dafny.label("1"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    d_17_cg_: _dafny.Seq
                    d_18_ci_: bool
                    d_19_cc_: _dafny.Seq
                    d_20_closed_: bool
                    out9_: _dafny.Seq
                    out10_: bool
                    out11_: _dafny.Seq
                    out12_: bool
                    out9_, out10_, out11_, out12_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_17_cg_ = out9_
                    d_18_ci_ = out10_
                    d_19_cc_ = out11_
                    d_20_closed_ = out12_
                    d_2_steps_ = (d_2_steps_) + (1)
                    generated = d_17_cg_
                    insideConstrainedOut = d_18_ci_
                    currentConstrainedOut = d_19_cc_
                    if d_20_closed_:
                        raise _dafny.Break("1")
                    elif True:
                        if (d_2_steps_) < (maxSteps):
                            d_21_constrainedPrompt_: _dafny.Seq
                            d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_22_next_: _dafny.Seq
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_22_next_ = out13_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_22_next_) == (eosToken):
                                raise _dafny.Break("1")
                            elif True:
                                d_23_ag_: _dafny.Seq
                                d_24_ai_: bool
                                d_25_ac_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                d_23_ag_ = out14_
                                d_24_ai_ = out15_
                                d_25_ac_ = out16_
                                generated = d_23_ag_
                                insideConstrainedOut = d_24_ai_
                                currentConstrainedOut = d_25_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_26_closeBudget_: int
            d_26_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_27_cg_: _dafny.Seq
            d_28_ci_: bool
            d_29_cc_: _dafny.Seq
            out17_: _dafny.Seq
            out18_: bool
            out19_: _dafny.Seq
            out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_26_closeBudget_)
            d_27_cg_ = out17_
            d_28_ci_ = out18_
            d_29_cc_ = out19_
            generated = d_27_cg_
            insideConstrainedOut = d_28_ci_
            currentConstrainedOut = d_29_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


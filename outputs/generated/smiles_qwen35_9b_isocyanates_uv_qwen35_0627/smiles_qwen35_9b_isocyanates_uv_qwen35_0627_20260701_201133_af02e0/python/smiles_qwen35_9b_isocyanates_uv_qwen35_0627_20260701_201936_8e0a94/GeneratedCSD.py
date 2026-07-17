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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SMILES string for a novel isocyanate molecule. Isocyanates contain the functional group R-N=C=O where R is an organic substituent. Example patterns: CH3N=C=O, C2H5N=C=O, ClCCN=C=O. Output ONLY the SMILES string. The SMILES must contain the N=C=O substructure. Do not copy any prompt example."))
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
        d_6_mainBudget_: int
        d_6_mainBudget_ = 0
        if (maxSteps) > ((d_2_steps_) + (2)):
            d_6_mainBudget_ = ((maxSteps) - (d_2_steps_)) - (2)
        elif (maxSteps) > (d_2_steps_):
            d_6_mainBudget_ = (maxSteps) - (d_2_steps_)
        d_7_innerSteps_: int
        d_7_innerSteps_ = 0
        with _dafny.label("0"):
            while ((((d_2_steps_) < (maxSteps)) and ((d_7_innerSteps_) < (d_6_mainBudget_))) and (insideConstrainedOut)) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                with _dafny.c_label("0"):
                    d_8_constrainedPrompt_: _dafny.Seq
                    d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_9_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                    d_9_next_ = out3_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_7_innerSteps_ = (d_7_innerSteps_) + (1)
                    if (d_9_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        d_10_ag_: _dafny.Seq
                        d_11_ai_: bool
                        d_12_ac_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                        d_10_ag_ = out4_
                        d_11_ai_ = out5_
                        d_12_ac_ = out6_
                        generated = d_10_ag_
                        insideConstrainedOut = d_11_ai_
                        currentConstrainedOut = d_12_ac_
                    pass
            pass
        if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_2_steps_) < (maxSteps)):
            d_13_cg_: _dafny.Seq
            d_14_ci_: bool
            d_15_cc_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_13_cg_ = out7_
            d_14_ci_ = out8_
            d_15_cc_ = out9_
            generated = d_13_cg_
            insideConstrainedOut = d_14_ci_
            currentConstrainedOut = d_15_cc_
            d_2_steps_ = (d_2_steps_) + (1)
        elif (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_16_closeBudget_: int
            d_16_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_17_cg_: _dafny.Seq
            d_18_ci_: bool
            d_19_cc_: _dafny.Seq
            out10_: _dafny.Seq
            out11_: bool
            out12_: _dafny.Seq
            out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_16_closeBudget_)
            d_17_cg_ = out10_
            d_18_ci_ = out11_
            d_19_cc_ = out12_
            generated = d_17_cg_
            insideConstrainedOut = d_18_ci_
            currentConstrainedOut = d_19_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


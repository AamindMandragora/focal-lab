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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You must generate a novel isocyanate molecule SMILES string. Isocyanates have the R-N=C=O functional group. First write a brief description of the molecule, then output the SMILES string enclosed in << and >> delimiters. For example: 'Propyl isocyanate: <<CCCN=C=O>>' or 'Chloromethyl isocyanate: <<ClCN=C=O>>'. The SMILES inside << >> must contain N=C=O and must not be copied from the prompt examples."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_preambleBudget_: int
        d_3_preambleBudget_ = 0
        if (maxSteps) > (10):
            d_3_preambleBudget_ = _dafny.euclidian_division(maxSteps, 2)
        elif True:
            d_3_preambleBudget_ = maxSteps
        with _dafny.label("0"):
            while ((d_2_steps_) < (d_3_preambleBudget_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_4_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_4_next_ = out0_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_4_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                    if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_5_og_: _dafny.Seq
            d_6_oi_: bool
            d_7_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_5_og_ = out1_
            d_6_oi_ = out2_
            d_7_oc_ = out3_
            generated = d_5_og_
            insideConstrainedOut = d_6_oi_
            currentConstrainedOut = d_7_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        d_8_closingReserve_: int
        d_8_closingReserve_ = 3
        d_9_constrainedBudget_: int
        d_9_constrainedBudget_ = 0
        if (maxSteps) > ((d_2_steps_) + (d_8_closingReserve_)):
            d_9_constrainedBudget_ = ((maxSteps) - (d_2_steps_)) - (d_8_closingReserve_)
        elif (maxSteps) > (d_2_steps_):
            d_9_constrainedBudget_ = (maxSteps) - (d_2_steps_)
        d_10_innerSteps_: int
        d_10_innerSteps_ = 0
        with _dafny.label("1"):
            while ((((d_2_steps_) < (maxSteps)) and ((d_10_innerSteps_) < (d_9_constrainedBudget_))) and (insideConstrainedOut)) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                with _dafny.c_label("1"):
                    d_11_constrainedPrompt_: _dafny.Seq
                    d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_12_next_: _dafny.Seq
                    out4_: _dafny.Seq
                    out4_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                    d_12_next_ = out4_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_10_innerSteps_ = (d_10_innerSteps_) + (1)
                    if (d_12_next_) == (eosToken):
                        raise _dafny.Break("1")
                    elif True:
                        d_13_ag_: _dafny.Seq
                        d_14_ai_: bool
                        d_15_ac_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                        d_13_ag_ = out5_
                        d_14_ai_ = out6_
                        d_15_ac_ = out7_
                        generated = d_13_ag_
                        insideConstrainedOut = d_14_ai_
                        currentConstrainedOut = d_15_ac_
                    pass
            pass
        if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_2_steps_) < (maxSteps)):
            d_16_cg_: _dafny.Seq
            d_17_ci_: bool
            d_18_cc_: _dafny.Seq
            out8_: _dafny.Seq
            out9_: bool
            out10_: _dafny.Seq
            out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_16_cg_ = out8_
            d_17_ci_ = out9_
            d_18_cc_ = out10_
            generated = d_16_cg_
            insideConstrainedOut = d_17_ci_
            currentConstrainedOut = d_18_cc_
            d_2_steps_ = (d_2_steps_) + (1)
        elif (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_19_closeBudget_: int
            d_19_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_20_cg_: _dafny.Seq
            d_21_ci_: bool
            d_22_cc_: _dafny.Seq
            out11_: _dafny.Seq
            out12_: bool
            out13_: _dafny.Seq
            out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_closeBudget_)
            d_20_cg_ = out11_
            d_21_ci_ = out12_
            d_22_cc_ = out13_
            generated = d_20_cg_
            insideConstrainedOut = d_21_ci_
            currentConstrainedOut = d_22_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


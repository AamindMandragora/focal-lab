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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SMILES string for a novel acrylate ester molecule (containing CH2=CH-C(=O)-O- group). The SMILES must be chemically valid and represent a real acrylate. Output only the SMILES string inside the constrained span delimiters."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_preambleBudget_: int
        d_3_preambleBudget_ = 6
        d_4_preambleSteps_: int
        d_4_preambleSteps_ = 0
        with _dafny.label("0"):
            while (((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut))) and ((d_4_preambleSteps_) < (d_3_preambleBudget_)):
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
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_6_og_: _dafny.Seq
            d_7_oi_: bool
            d_8_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_6_og_ = out1_
            d_7_oi_ = out2_
            d_8_oc_ = out3_
            generated = d_6_og_
            insideConstrainedOut = d_7_oi_
            currentConstrainedOut = d_8_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        d_9_constrainedTokenCount_: int
        d_9_constrainedTokenCount_ = 0
        d_10_minConstrainedTokens_: int
        d_10_minConstrainedTokens_ = 6
        with _dafny.label("1"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    if ((d_9_constrainedTokenCount_) >= (d_10_minConstrainedTokens_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                        d_11_cg_: _dafny.Seq
                        d_12_ci_: bool
                        d_13_cc_: _dafny.Seq
                        d_14_closed_: bool
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_11_cg_ = out4_
                        d_12_ci_ = out5_
                        d_13_cc_ = out6_
                        d_14_closed_ = out7_
                        d_2_steps_ = (d_2_steps_) + (1)
                        generated = d_11_cg_
                        insideConstrainedOut = d_12_ci_
                        currentConstrainedOut = d_13_cc_
                    elif not((parser).IsCompletePrefix(currentConstrainedOut)):
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_16_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (d_9_constrainedTokenCount_) < (4):
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('15e-1'), eosToken)
                            d_16_next_ = out8_
                        elif (d_9_constrainedTokenCount_) < (15):
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('25e-1'), eosToken)
                            d_16_next_ = out9_
                        elif True:
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 15, eosToken)
                            d_16_next_ = out10_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_16_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_17_ag_: _dafny.Seq
                            d_18_ai_: bool
                            d_19_ac_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                            d_17_ag_ = out11_
                            d_18_ai_ = out12_
                            d_19_ac_ = out13_
                            generated = d_17_ag_
                            insideConstrainedOut = d_18_ai_
                            currentConstrainedOut = d_19_ac_
                            d_9_constrainedTokenCount_ = (d_9_constrainedTokenCount_) + (1)
                    elif True:
                        d_20_cg_: _dafny.Seq
                        d_21_ci_: bool
                        d_22_cc_: _dafny.Seq
                        d_23_closed_: bool
                        out14_: _dafny.Seq
                        out15_: bool
                        out16_: _dafny.Seq
                        out17_: bool
                        out14_, out15_, out16_, out17_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_20_cg_ = out14_
                        d_21_ci_ = out15_
                        d_22_cc_ = out16_
                        d_23_closed_ = out17_
                        d_2_steps_ = (d_2_steps_) + (1)
                        generated = d_20_cg_
                        insideConstrainedOut = d_21_ci_
                        currentConstrainedOut = d_22_cc_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_24_closeBudget_: int
            d_24_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_25_cg_: _dafny.Seq
            d_26_ci_: bool
            d_27_cc_: _dafny.Seq
            out18_: _dafny.Seq
            out19_: bool
            out20_: _dafny.Seq
            out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_24_closeBudget_)
            d_25_cg_ = out18_
            d_26_ci_ = out19_
            d_27_cc_ = out20_
            generated = d_25_cg_
            insideConstrainedOut = d_26_ci_
            currentConstrainedOut = d_27_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


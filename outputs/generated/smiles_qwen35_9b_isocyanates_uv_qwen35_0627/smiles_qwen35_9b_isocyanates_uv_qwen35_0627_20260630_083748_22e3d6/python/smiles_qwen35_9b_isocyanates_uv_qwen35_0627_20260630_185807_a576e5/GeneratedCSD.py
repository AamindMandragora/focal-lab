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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a SMILES for an isocyanate compound. An isocyanate MUST contain the N=C=O group. Valid examples: CCN=C=O (ethyl isocyanate), CCCN=C=O (propyl isocyanate), O=C=Nc1ccccc1 (phenyl isocyanate), ClCCN=C=O, BrCCN=C=O, CCCCN=C=O, O=C=NCC, O=C=NCCC. The SMILES must contain N=C=O or [N]=C=O pattern.")))
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
        d_5_minTokens_: int
        d_5_minTokens_ = 10
        d_6_minCount_: int
        d_6_minCount_ = 0
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((d_6_minCount_) < (d_5_minTokens_)):
                with _dafny.c_label("0"):
                    d_7_constrainedPrompt_: _dafny.Seq
                    d_7_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_8_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, eosToken)
                    d_8_next_ = out3_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_6_minCount_ = (d_6_minCount_) + (1)
                    if (d_8_next_) == (eosToken):
                        raise _dafny.Break("0")
                    d_9_isComplete_: bool
                    d_9_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if not(d_9_isComplete_):
                        d_10_valid_: bool
                        out4_: bool
                        out4_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_8_next_)
                        d_10_valid_ = out4_
                        if d_10_valid_:
                            d_11_ag_: _dafny.Seq
                            d_12_ai_: bool
                            d_13_ac_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_next_)
                            d_11_ag_ = out5_
                            d_12_ai_ = out6_
                            d_13_ac_ = out7_
                            generated = d_11_ag_
                            insideConstrainedOut = d_12_ai_
                            currentConstrainedOut = d_13_ac_
                    pass
            pass
        d_14_extraCount_: int
        d_14_extraCount_ = 0
        d_15_maxExtra_: int
        d_15_maxExtra_ = 30
        with _dafny.label("1"):
            while (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((d_14_extraCount_) < (d_15_maxExtra_)):
                with _dafny.c_label("1"):
                    d_16_currentStr_: _dafny.Seq
                    d_16_currentStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                    d_17_hasIsocyanate_: int
                    d_17_hasIsocyanate_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_16_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))
                    if (d_17_hasIsocyanate_) > (0):
                        raise _dafny.Break("1")
                    d_18_constrainedPrompt_: _dafny.Seq
                    d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_19_next_: _dafny.Seq
                    out8_: _dafny.Seq
                    out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                    d_19_next_ = out8_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_14_extraCount_ = (d_14_extraCount_) + (1)
                    if (d_19_next_) == (eosToken):
                        raise _dafny.Break("1")
                    d_20_isComplete_: bool
                    d_20_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if not(d_20_isComplete_):
                        d_21_valid_: bool
                        out9_: bool
                        out9_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_19_next_)
                        d_21_valid_ = out9_
                        if d_21_valid_:
                            d_22_ag_: _dafny.Seq
                            d_23_ai_: bool
                            d_24_ac_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                            d_22_ag_ = out10_
                            d_23_ai_ = out11_
                            d_24_ac_ = out12_
                            generated = d_22_ag_
                            insideConstrainedOut = d_23_ai_
                            currentConstrainedOut = d_24_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_25_closeBudget_: int
            d_25_closeBudget_ = (maxSteps) - (d_1_steps_)
            if (d_25_closeBudget_) > (60):
                d_25_closeBudget_ = 60
            d_26_cg_: _dafny.Seq
            d_27_ci_: bool
            d_28_cc_: _dafny.Seq
            out13_: _dafny.Seq
            out14_: bool
            out15_: _dafny.Seq
            out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_25_closeBudget_)
            d_26_cg_ = out13_
            d_27_ci_ = out14_
            d_28_cc_ = out15_
            generated = d_26_cg_
            insideConstrainedOut = d_27_ci_
            currentConstrainedOut = d_28_cc_
            d_1_steps_ = (d_1_steps_) + (d_25_closeBudget_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output one valid isocyanate SMILES with N=C=O group. Example: CCN=C=O")))
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
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_5_cg0_: _dafny.Seq
                    d_6_ci0_: bool
                    d_7_cc0_: _dafny.Seq
                    d_8_closed_: bool
                    out3_: _dafny.Seq
                    out4_: bool
                    out5_: _dafny.Seq
                    out6_: bool
                    out3_, out4_, out5_, out6_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_5_cg0_ = out3_
                    d_6_ci0_ = out4_
                    d_7_cc0_ = out5_
                    d_8_closed_ = out6_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_8_closed_:
                        d_9_currentStr_: _dafny.Seq
                        d_9_currentStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(d_7_cc0_)
                        d_10_prevStr_: _dafny.Seq
                        d_10_prevStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                        d_11_hasIsocyanate_: bool
                        d_11_hasIsocyanate_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_10_prevStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))) > (0)
                        if d_11_hasIsocyanate_:
                            generated = d_5_cg0_
                            insideConstrainedOut = d_6_ci0_
                            currentConstrainedOut = d_7_cc0_
                        elif True:
                            generated = d_5_cg0_
                            insideConstrainedOut = d_6_ci0_
                            currentConstrainedOut = d_7_cc0_
                    elif True:
                        d_12_constrainedPrompt_: _dafny.Seq
                        d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_13_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                        d_13_next_ = out7_
                        if (d_13_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_14_valid_: bool
                            out8_: bool
                            out8_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_13_next_)
                            d_14_valid_ = out8_
                            if (d_14_valid_) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                                d_15_ag_: _dafny.Seq
                                d_16_ai_: bool
                                d_17_ac_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                d_15_ag_ = out9_
                                d_16_ai_ = out10_
                                d_17_ac_ = out11_
                                generated = d_15_ag_
                                insideConstrainedOut = d_16_ai_
                                currentConstrainedOut = d_17_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_18_closeBudget_: int
            d_18_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_19_cg_: _dafny.Seq
            d_20_ci_: bool
            d_21_cc_: _dafny.Seq
            out12_: _dafny.Seq
            out13_: bool
            out14_: _dafny.Seq
            out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_closeBudget_)
            d_19_cg_ = out12_
            d_20_ci_ = out13_
            d_21_cc_ = out14_
            generated = d_19_cg_
            insideConstrainedOut = d_20_ci_
            currentConstrainedOut = d_21_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


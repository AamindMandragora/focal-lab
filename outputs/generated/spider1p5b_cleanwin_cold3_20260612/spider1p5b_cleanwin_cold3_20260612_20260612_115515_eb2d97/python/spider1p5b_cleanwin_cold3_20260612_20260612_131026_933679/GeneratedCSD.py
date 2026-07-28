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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query in the format: SQL: <<YOUR QUERY>>. Write a complete SELECT statement using only the provided schema. No explanation, no Markdown, no extra text."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_maxPrefixSteps_: int
        d_3_maxPrefixSteps_ = 8
        d_4_prefixDone_: bool
        d_4_prefixDone_ = insideConstrained
        with _dafny.label("0"):
            while (((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut))) and ((d_2_steps_) < (d_3_maxPrefixSteps_)):
                with _dafny.c_label("0"):
                    d_5_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_5_next_ = out0_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_5_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
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
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_9_constrainedBudget_: int
            d_9_constrainedBudget_ = (maxSteps) - (d_2_steps_)
            if (d_9_constrainedBudget_) > (5):
                d_9_constrainedBudget_ = (d_9_constrainedBudget_) - (5)
            d_10_innerSteps_: int
            d_10_innerSteps_ = 0
            with _dafny.label("2_0"):
                while (((d_10_innerSteps_) < (d_9_constrainedBudget_)) and (insideConstrainedOut)) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                    with _dafny.c_label("2_0"):
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_12_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                        d_12_next_ = out4_
                        d_10_innerSteps_ = (d_10_innerSteps_) + (1)
                        if (d_12_next_) == (eosToken):
                            d_13_rg_: _dafny.Seq
                            d_14_rc_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: _dafny.Seq
                            out5_, out6_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_13_rg_ = out5_
                            d_14_rc_ = out6_
                            generated = d_13_rg_
                            currentConstrainedOut = d_14_rc_
                            raise _dafny.Break("2_0")
                        elif True:
                            d_15_ag_: _dafny.Seq
                            d_16_ai_: bool
                            d_17_ac_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                            d_15_ag_ = out7_
                            d_16_ai_ = out8_
                            d_17_ac_ = out9_
                            generated = d_15_ag_
                            insideConstrainedOut = d_16_ai_
                            currentConstrainedOut = d_17_ac_
                        pass
                pass
            d_2_steps_ = (d_2_steps_) + (d_10_innerSteps_)
            if (insideConstrainedOut) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                d_18_rg_: _dafny.Seq
                d_19_rc_: _dafny.Seq
                out10_: _dafny.Seq
                out11_: _dafny.Seq
                out10_, out11_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                d_18_rg_ = out10_
                d_19_rc_ = out11_
                generated = d_18_rg_
                currentConstrainedOut = d_19_rc_
            if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_2_steps_) < (maxSteps)):
                d_20_cg_: _dafny.Seq
                d_21_ci_: bool
                d_22_cc_: _dafny.Seq
                out12_: _dafny.Seq
                out13_: bool
                out14_: _dafny.Seq
                out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_20_cg_ = out12_
                d_21_ci_ = out13_
                d_22_cc_ = out14_
                generated = d_20_cg_
                insideConstrainedOut = d_21_ci_
                currentConstrainedOut = d_22_cc_
                d_2_steps_ = (d_2_steps_) + (1)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


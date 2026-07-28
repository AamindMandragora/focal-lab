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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer the question with exactly one SQL query. Format your answer as: SQL: <<SELECT ...>> where the SQL query goes between the delimiters. Use only tables and columns from the schema provided. No explanation."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_preambleLimit_: int
        d_3_preambleLimit_ = 10
        d_4_closeReserve_: int
        d_4_closeReserve_ = 25
        with _dafny.label("0"):
            while (((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut))) and ((d_2_steps_) < (d_3_preambleLimit_)):
                with _dafny.c_label("0"):
                    d_5_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_5_next_ = out0_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_5_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_6_eg_: _dafny.Seq
                            d_7_ei_: bool
                            d_8_ec_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_6_eg_ = out1_
                            d_7_ei_ = out2_
                            d_8_ec_ = out3_
                            generated = d_6_eg_
                            insideConstrainedOut = d_7_ei_
                            currentConstrainedOut = d_8_ec_
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
        with _dafny.label("1"):
            while (((d_2_steps_) < (maxSteps)) and (insideConstrainedOut)) and (((d_2_steps_) + (d_4_closeReserve_)) < (maxSteps)):
                with _dafny.c_label("1"):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        raise _dafny.Break("1")
                    d_12_stableLen_: int
                    d_12_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
                    d_13_constrainedPrompt_: _dafny.Seq
                    d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:d_12_stableLen_:]))
                    d_14_next_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                    d_14_next_ = out7_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_14_next_) == (eosToken):
                        raise _dafny.Break("1")
                    elif True:
                        if not((parser).IsCompletePrefix(currentConstrainedOut)):
                            d_15_ag_: _dafny.Seq
                            d_16_ai_: bool
                            d_17_ac_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                            d_15_ag_ = out8_
                            d_16_ai_ = out9_
                            d_17_ac_ = out10_
                            generated = d_15_ag_
                            insideConstrainedOut = d_16_ai_
                            currentConstrainedOut = d_17_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_18_closeBudget_: int
            d_18_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_19_cg_: _dafny.Seq
            d_20_ci_: bool
            d_21_cc_: _dafny.Seq
            out11_: _dafny.Seq
            out12_: bool
            out13_: _dafny.Seq
            out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_closeBudget_)
            d_19_cg_ = out11_
            d_20_ci_ = out12_
            d_21_cc_ = out13_
            generated = d_19_cg_
            insideConstrainedOut = d_20_ci_
            currentConstrainedOut = d_21_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


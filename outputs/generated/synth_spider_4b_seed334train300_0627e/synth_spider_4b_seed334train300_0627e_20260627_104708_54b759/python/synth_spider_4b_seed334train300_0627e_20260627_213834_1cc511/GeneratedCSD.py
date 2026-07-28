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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output format: SQL: <<QUERY>> where QUERY is a minimal valid SQL statement. Use ONLY table and column names from the schema. SELECT only the exact columns the question asks for. Do NOT add IS NOT NULL, LIMIT, or extra conditions unless the question requires them.")))
        d_2_freeCap_: int
        d_2_freeCap_ = 4
        d_3_freeSteps_: int
        d_3_freeSteps_ = 0
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut))) and ((d_3_freeSteps_) < (d_2_freeCap_)):
                with _dafny.c_label("0"):
                    d_4_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_4_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_3_freeSteps_ = (d_3_freeSteps_) + (1)
                    if (d_4_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                    if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_5_eg_: _dafny.Seq
                        d_6_ei_: bool
                        d_7_ec_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_5_eg_ = out1_
                        d_6_ei_ = out2_
                        d_7_ec_ = out3_
                        generated = d_5_eg_
                        insideConstrainedOut = d_6_ei_
                        currentConstrainedOut = d_7_ec_
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_8_og_: _dafny.Seq
            d_9_oi_: bool
            d_10_oc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_8_og_ = out4_
            d_9_oi_ = out5_
            d_10_oc_ = out6_
            generated = d_8_og_
            insideConstrainedOut = d_9_oi_
            currentConstrainedOut = d_10_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        d_11_closeReserve_: int
        d_11_closeReserve_ = 35
        with _dafny.label("1"):
            while ((insideConstrainedOut) and ((d_1_steps_) < (maxSteps))) and (((maxSteps) - (d_1_steps_)) > (d_11_closeReserve_)):
                with _dafny.c_label("1"):
                    d_12_stable_: _dafny.Seq
                    d_12_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_13_constrainedPrompt_: _dafny.Seq
                    d_13_constrainedPrompt_ = (prompt) + (d_12_stable_)
                    d_14_next_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (d_0_helpers_).PenalizedConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('3e0'), eosToken)
                    d_14_next_ = out7_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_14_next_) == (eosToken):
                        raise _dafny.Break("1")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        raise _dafny.Break("1")
                    elif True:
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
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_18_closeBudget_: int
            d_18_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_19_cg2_: _dafny.Seq
            d_20_ci2_: bool
            d_21_cc2_: _dafny.Seq
            out11_: _dafny.Seq
            out12_: bool
            out13_: _dafny.Seq
            out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_closeBudget_)
            d_19_cg2_ = out11_
            d_20_ci2_ = out12_
            d_21_cc2_ = out13_
            generated = d_19_cg2_
            insideConstrainedOut = d_20_ci2_
            currentConstrainedOut = d_21_cc2_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


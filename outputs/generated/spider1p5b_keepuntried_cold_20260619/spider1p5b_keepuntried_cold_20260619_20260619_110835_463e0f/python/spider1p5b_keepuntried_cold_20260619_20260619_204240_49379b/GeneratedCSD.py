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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single SQL SELECT query. Use only the tables and columns from the schema. Keep the SQL simple and correct. Do not add unnecessary JOINs. End the query properly with a semicolon or closing keyword.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            generated = out0_
            insideConstrainedOut = out1_
            currentConstrainedOut = out2_
            d_1_steps_ = (d_1_steps_) + (1)
        d_2_spanTokens_: int
        d_2_spanTokens_ = 0
        d_3_eosHits_: int
        d_3_eosHits_ = 0
        d_4_maxEosHits_: int
        d_4_maxEosHits_ = 3
        with _dafny.label("0"):
            while (((d_1_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_5_cg_: _dafny.Seq
                    d_6_ci_: bool
                    d_7_cc_: _dafny.Seq
                    d_8_closed_: bool
                    out3_: _dafny.Seq
                    out4_: bool
                    out5_: _dafny.Seq
                    out6_: bool
                    out3_, out4_, out5_, out6_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_5_cg_ = out3_
                    d_6_ci_ = out4_
                    d_7_cc_ = out5_
                    d_8_closed_ = out6_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_8_closed_:
                        generated = d_5_cg_
                        insideConstrainedOut = d_6_ci_
                        currentConstrainedOut = d_7_cc_
                        raise _dafny.Break("0")
                    if (d_1_steps_) >= (maxSteps):
                        raise _dafny.Break("0")
                    d_9_constrainedPrompt_: _dafny.Seq
                    d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_10_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                    if (d_2_spanTokens_) >= (100):
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e-1'), eosToken)
                        d_10_next_ = out7_
                    elif True:
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_10_next_ = out8_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_10_next_) == (eosToken):
                        d_3_eosHits_ = (d_3_eosHits_) + (1)
                        if (d_3_eosHits_) >= (d_4_maxEosHits_):
                            raise _dafny.Break("0")
                    elif True:
                        d_3_eosHits_ = 0
                        d_11_ag_: _dafny.Seq
                        d_12_ai_: bool
                        d_13_ac_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: bool
                        out11_: _dafny.Seq
                        out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                        d_11_ag_ = out9_
                        d_12_ai_ = out10_
                        d_13_ac_ = out11_
                        generated = d_11_ag_
                        insideConstrainedOut = d_12_ai_
                        currentConstrainedOut = d_13_ac_
                        d_2_spanTokens_ = (d_2_spanTokens_) + (1)
                    pass
            pass
        if ((insideConstrainedOut) and (((d_1_steps_) + (1)) < (maxSteps))) and ((parser).IsCompletePrefix(currentConstrainedOut)):
            d_14_cg2_: _dafny.Seq
            d_15_ci2_: bool
            d_16_cc2_: _dafny.Seq
            out12_: _dafny.Seq
            out13_: bool
            out14_: _dafny.Seq
            out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_14_cg2_ = out12_
            d_15_ci2_ = out13_
            d_16_cc2_ = out14_
            generated = d_14_cg2_
            insideConstrainedOut = d_15_ci2_
            currentConstrainedOut = d_16_cc2_
            d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


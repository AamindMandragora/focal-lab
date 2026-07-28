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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a COMPLETE SQL SELECT statement that fully answers the question. Include ALL necessary clauses: WHERE for filtering conditions, JOIN for combining tables, GROUP BY/HAVING for aggregation, ORDER BY/LIMIT for ordering and row limits. Use ONLY the exact full table and column names from the schema without abbreviations or aliases. SQL: <<full_complete_query>>")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and ((d_1_steps_) < (5))) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_2_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_2_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_2_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                        if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_3_og_: _dafny.Seq
            d_4_oi_: bool
            d_5_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_3_og_ = out1_
            d_4_oi_ = out2_
            d_5_oc_ = out3_
            generated = d_3_og_
            insideConstrainedOut = d_4_oi_
            currentConstrainedOut = d_5_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        d_6_sqlCap_: int
        if (maxSteps) > ((d_1_steps_) + (250)):
            d_6_sqlCap_ = 250
        elif True:
            d_6_sqlCap_ = (maxSteps) - (d_1_steps_)
        d_7_sqlSteps_: int
        d_7_sqlSteps_ = 0
        with _dafny.label("1"):
            while (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((d_7_sqlSteps_) < (d_6_sqlCap_)):
                with _dafny.c_label("1"):
                    d_8_constrainedPrompt_: _dafny.Seq
                    d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_9_next_: _dafny.Seq
                    out4_: _dafny.Seq
                    out4_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), eosToken)
                    d_9_next_ = out4_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_7_sqlSteps_ = (d_7_sqlSteps_) + (1)
                    if (d_9_next_) == (eosToken):
                        if (((len(currentConstrainedOut)) > (0)) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
                            d_10_cg_: _dafny.Seq
                            d_11_ci_: bool
                            d_12_cc_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_cg_ = out5_
                            d_11_ci_ = out6_
                            d_12_cc_ = out7_
                            generated = d_10_cg_
                            insideConstrainedOut = d_11_ci_
                            currentConstrainedOut = d_12_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("1")
                    elif True:
                        d_13_ag_: _dafny.Seq
                        d_14_ai_: bool
                        d_15_ac_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                        d_13_ag_ = out8_
                        d_14_ai_ = out9_
                        d_15_ac_ = out10_
                        generated = d_13_ag_
                        insideConstrainedOut = d_14_ai_
                        currentConstrainedOut = d_15_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_16_remaining_: int
            d_16_remaining_ = (maxSteps) - (d_1_steps_)
            d_17_closeBudget_: int
            if (d_16_remaining_) <= (150):
                d_17_closeBudget_ = d_16_remaining_
            elif True:
                d_17_closeBudget_ = 150
            d_18_cg_: _dafny.Seq
            d_19_ci_: bool
            d_20_cc_: _dafny.Seq
            out11_: _dafny.Seq
            out12_: bool
            out13_: _dafny.Seq
            out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget_)
            d_18_cg_ = out11_
            d_19_ci_ = out12_
            d_20_cc_ = out13_
            generated = d_18_cg_
            insideConstrainedOut = d_19_ci_
            currentConstrainedOut = d_20_cc_
            d_1_steps_ = (d_1_steps_) + (d_17_closeBudget_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


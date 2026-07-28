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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SQL query. Use only tables and columns from the schema. Write the simplest correct SQL: use SELECT...FROM...WHERE or GROUP BY or JOIN as needed. Do not repeat columns. Do not generate nested subqueries unless required. Stop as soon as the query is complete.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            insideConstrainedOut = True
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_2_cg_: _dafny.Seq
                    d_3_ci_: bool
                    d_4_cc_: _dafny.Seq
                    d_5_closed_: bool
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: _dafny.Seq
                    out3_: bool
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_2_cg_ = out0_
                    d_3_ci_ = out1_
                    d_4_cc_ = out2_
                    d_5_closed_ = out3_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_5_closed_:
                        generated = d_2_cg_
                        insideConstrainedOut = d_3_ci_
                        currentConstrainedOut = d_4_cc_
                        raise _dafny.Break("0")
                    elif True:
                        d_6_constrainedPrompt_: _dafny.Seq
                        d_6_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_7_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_6_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                        d_7_next_ = out4_
                        if (d_7_next_) == (eosToken):
                            d_8_rg_: _dafny.Seq
                            d_9_rc_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: _dafny.Seq
                            out5_, out6_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_8_rg_ = out5_
                            d_9_rc_ = out6_
                            generated = d_8_rg_
                            currentConstrainedOut = d_9_rc_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_10_fg_: _dafny.Seq
                                d_11_fi_: bool
                                d_12_fc_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_10_fg_ = out7_
                                d_11_fi_ = out8_
                                d_12_fc_ = out9_
                                d_1_steps_ = (d_1_steps_) + (1)
                                generated = d_10_fg_
                                insideConstrainedOut = d_11_fi_
                                currentConstrainedOut = d_12_fc_
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            raise _dafny.Break("0")
                        elif True:
                            d_13_ag_: _dafny.Seq
                            d_14_ai_: bool
                            d_15_ac_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_next_)
                            d_13_ag_ = out10_
                            d_14_ai_ = out11_
                            d_15_ac_ = out12_
                            generated = d_13_ag_
                            insideConstrainedOut = d_14_ai_
                            currentConstrainedOut = d_15_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) >= (maxSteps)):
            d_16_rg_: _dafny.Seq
            d_17_rc_: _dafny.Seq
            out13_: _dafny.Seq
            out14_: _dafny.Seq
            out13_, out14_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_16_rg_ = out13_
            d_17_rc_ = out14_
            generated = d_16_rg_
            currentConstrainedOut = d_17_rc_
            if (parser).IsCompletePrefix(currentConstrainedOut):
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


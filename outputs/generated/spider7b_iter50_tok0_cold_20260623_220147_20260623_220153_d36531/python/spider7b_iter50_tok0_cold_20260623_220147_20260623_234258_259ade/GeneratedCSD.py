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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single SQL query. Critical rules: (1) Use MAX(col) or MIN(col) for finding largest/smallest values, NOT ORDER BY...LIMIT 1. (2) Use COUNT(*) to count rows, not COUNT(column). (3) Do NOT use table aliases. (4) Only JOIN tables that are strictly necessary - prefer simple single-table queries when possible. (5) Use lowercase SQL keywords. (6) Match exact table and column names from the schema. No markdown. Output: SQL: <query>")))
        if not(insideConstrainedOut):
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
            generated = out0_
            insideConstrainedOut = out1_
            currentConstrainedOut = out2_
        d_2_reservedClose_: int
        d_2_reservedClose_ = 40
        d_3_genBudget_: int
        if (maxSteps) > (d_2_reservedClose_):
            d_3_genBudget_ = (maxSteps) - (d_2_reservedClose_)
        elif True:
            d_3_genBudget_ = 0
        with _dafny.label("0"):
            while (((d_1_steps_) + (2)) <= (d_3_genBudget_)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_4_cg_: _dafny.Seq
                    d_5_ci_: bool
                    d_6_cc_: _dafny.Seq
                    d_7_closed_: bool
                    out3_: _dafny.Seq
                    out4_: bool
                    out5_: _dafny.Seq
                    out6_: bool
                    out3_, out4_, out5_, out6_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_4_cg_ = out3_
                    d_5_ci_ = out4_
                    d_6_cc_ = out5_
                    d_7_closed_ = out6_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_7_closed_:
                        generated = d_4_cg_
                        insideConstrainedOut = d_5_ci_
                        currentConstrainedOut = d_6_cc_
                    elif True:
                        d_8_constrainedPrompt_: _dafny.Seq
                        d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_9_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                        d_9_next_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_9_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_10_gNew_: _dafny.Seq
                            d_11_iNew_: bool
                            d_12_cNew_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                            d_10_gNew_ = out8_
                            d_11_iNew_ = out9_
                            d_12_cNew_ = out10_
                            generated = d_10_gNew_
                            insideConstrainedOut = d_11_iNew_
                            currentConstrainedOut = d_12_cNew_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_13_closeBudget_: int
            d_13_closeBudget_ = (maxSteps) - (d_1_steps_)
            out11_: _dafny.Seq
            out12_: bool
            out13_: _dafny.Seq
            out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_13_closeBudget_)
            generated = out11_
            insideConstrainedOut = out12_
            currentConstrainedOut = out13_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


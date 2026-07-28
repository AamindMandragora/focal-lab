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
        if True:
            generated = generatedPrefix
            insideConstrainedOut = insideConstrained
            currentConstrainedOut = currentConstrained
            cost = 0
            if (maxSteps) == (0):
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            (d_0_helpers_).AppendTaskGuidance(lm, (((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a complete and syntactically valid SQL query. Include all necessary clauses: "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT with correct columns, FROM with the right table(s), JOIN clauses when multiple ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "tables are needed, WHERE conditions, GROUP BY, HAVING, and ORDER BY as required. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use only table and column names from the schema. Do not truncate the query."))))
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
            d_2_closeReserve_: int = int(0)
            if (maxSteps) >= (80):
                d_2_closeReserve_ = 80
            elif True:
                d_2_closeReserve_ = _dafny.euclidian_division(maxSteps, 2)
            d_3_genBudget_: int = int(0)
            if ((d_1_steps_) + (d_2_closeReserve_)) <= (maxSteps):
                d_3_genBudget_ = (maxSteps) - (d_2_closeReserve_)
            elif True:
                d_3_genBudget_ = d_1_steps_
            with _dafny.label("0"):
                while ((d_1_steps_) < (d_3_genBudget_)) and (insideConstrainedOut):
                    with _dafny.c_label("0"):
                        d_4_constrainedPrompt_: _dafny.Seq
                        d_4_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_5_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_4_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_5_next_ = out3_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        d_6_isComplete_: bool
                        d_6_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_6_isComplete_:
                            raise _dafny.Break("0")
                        d_7_isValid_: bool
                        out4_: bool
                        out4_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_5_next_)
                        d_7_isValid_ = out4_
                        if d_7_isValid_:
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_5_next_)
                            generated = out5_
                            insideConstrainedOut = out6_
                            currentConstrainedOut = out7_
                        pass
                pass
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_8_closeBudget_: int
                d_8_closeBudget_ = (maxSteps) - (d_1_steps_)
                d_9_cg_: _dafny.Seq
                d_10_ci_: bool
                d_11_cc_: _dafny.Seq
                out8_: _dafny.Seq
                out9_: bool
                out10_: _dafny.Seq
                out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_8_closeBudget_)
                d_9_cg_ = out8_
                d_10_ci_ = out9_
                d_11_cc_ = out10_
                generated = d_9_cg_
                insideConstrainedOut = d_10_ci_
                currentConstrainedOut = d_11_cc_
                d_1_steps_ = maxSteps
            cost = d_1_steps_
            if (cost) > (maxSteps):
                cost = maxSteps
        return generated, insideConstrainedOut, currentConstrainedOut, cost


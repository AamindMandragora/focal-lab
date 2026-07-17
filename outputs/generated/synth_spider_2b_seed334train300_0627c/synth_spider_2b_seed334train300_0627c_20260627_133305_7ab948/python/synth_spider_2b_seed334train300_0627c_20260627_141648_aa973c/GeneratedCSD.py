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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a complete and correct SQL query using ONLY the table and column names from the schema. Include all necessary clauses (WHERE, GROUP BY, HAVING, ORDER BY, JOIN) to answer the question precisely. Output format: SQL: <<your complete query here>>")))
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_2_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_2_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_2_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                    if (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_3_rem_: int
            d_3_rem_ = (maxSteps) - (d_1_steps_)
            d_4_closeReserve_: int
            if (d_3_rem_) >= (20):
                d_4_closeReserve_ = 10
            elif True:
                if (d_3_rem_) >= (4):
                    d_4_closeReserve_ = 2
                elif True:
                    d_4_closeReserve_ = 1
            d_5_fillBudget_: int
            if (d_3_rem_) > (d_4_closeReserve_):
                d_5_fillBudget_ = (d_3_rem_) - (d_4_closeReserve_)
            elif True:
                d_5_fillBudget_ = 0
            if (d_5_fillBudget_) >= (1):
                d_6_stable_: _dafny.Seq
                d_6_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_7_constrainedPrompt_: _dafny.Seq
                d_7_constrainedPrompt_ = (prompt) + (d_6_stable_)
                d_8_maxStepsPerUnit_: int
                if (d_5_fillBudget_) >= (20):
                    d_8_maxStepsPerUnit_ = 20
                elif True:
                    d_8_maxStepsPerUnit_ = d_5_fillBudget_
                d_9_filled_: _dafny.Seq
                out1_: _dafny.Seq
                out1_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, eosToken, d_8_maxStepsPerUnit_, 3, d_5_fillBudget_)
                d_9_filled_ = out1_
                generated = (d_6_stable_) + (d_9_filled_)
                currentConstrainedOut = d_9_filled_
                d_1_steps_ = (d_1_steps_) + (d_5_fillBudget_)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_10_closeBudget_: int
            d_10_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_11_cg2_: _dafny.Seq
            d_12_ci2_: bool
            d_13_cc2_: _dafny.Seq
            out2_: _dafny.Seq
            out3_: bool
            out4_: _dafny.Seq
            out2_, out3_, out4_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_10_closeBudget_)
            d_11_cg2_ = out2_
            d_12_ci2_ = out3_
            d_13_cc2_ = out4_
            generated = d_11_cg2_
            insideConstrainedOut = d_12_ci2_
            currentConstrainedOut = d_13_cc2_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


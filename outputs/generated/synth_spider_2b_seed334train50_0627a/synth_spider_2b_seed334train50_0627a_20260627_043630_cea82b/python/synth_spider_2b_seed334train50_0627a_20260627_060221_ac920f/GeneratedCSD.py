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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write a single correct SQL query. Use INTERSECT for 'both X and Y' queries. Use ORDER BY col DESC LIMIT 1 instead of MAX() in WHERE. Use only table/column names from the schema. Keep queries simple and direct.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if not(insideConstrainedOut):
            if (d_1_steps_) < (maxSteps):
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
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_5_rem_: int
            d_5_rem_ = (maxSteps) - (d_1_steps_)
            d_6_closeReserve_: int
            d_6_closeReserve_ = _dafny.euclidian_division(d_5_rem_, 3)
            if ((d_6_closeReserve_) < (10)) and ((d_5_rem_) >= (12)):
                d_6_closeReserve_ = 10
            if (d_6_closeReserve_) > (65):
                d_6_closeReserve_ = 65
            if (d_6_closeReserve_) >= (d_5_rem_):
                if (d_5_rem_) > (1):
                    d_6_closeReserve_ = _dafny.euclidian_division(d_5_rem_, 2)
                elif True:
                    d_6_closeReserve_ = 0
            d_7_fillBudget_: int
            d_7_fillBudget_ = (d_5_rem_) - (d_6_closeReserve_)
            if (d_7_fillBudget_) >= (1):
                d_8_schemaTokens_: _dafny.Seq
                out3_: _dafny.Seq
                out3_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                d_8_schemaTokens_ = out3_
                d_9_promptStr_: _dafny.Seq
                d_9_promptStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(prompt)
                d_10_stable_: _dafny.Seq
                d_10_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_11_constrainedPrompt_: _dafny.Seq
                d_11_constrainedPrompt_ = (prompt) + (d_10_stable_)
                d_12_rollbackBudget_: int
                if (d_7_fillBudget_) >= (4):
                    d_12_rollbackBudget_ = _dafny.euclidian_division(d_7_fillBudget_, 4)
                elif True:
                    d_12_rollbackBudget_ = 1
                d_13_filled_: _dafny.Seq
                out4_: _dafny.Seq
                out4_ = (d_0_helpers_).RegenerateUnitOnCheckFailure(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken, d_7_fillBudget_, 3, d_12_rollbackBudget_, d_8_schemaTokens_)
                d_13_filled_ = out4_
                generated = (d_10_stable_) + (d_13_filled_)
                currentConstrainedOut = d_13_filled_
                d_1_steps_ = (d_1_steps_) + (d_7_fillBudget_)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_14_closeBudget_: int
            d_14_closeBudget_ = (maxSteps) - (d_1_steps_)
            out5_: _dafny.Seq
            out6_: bool
            out7_: _dafny.Seq
            out5_, out6_, out7_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudget_)
            generated = out5_
            insideConstrainedOut = out6_
            currentConstrainedOut = out7_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


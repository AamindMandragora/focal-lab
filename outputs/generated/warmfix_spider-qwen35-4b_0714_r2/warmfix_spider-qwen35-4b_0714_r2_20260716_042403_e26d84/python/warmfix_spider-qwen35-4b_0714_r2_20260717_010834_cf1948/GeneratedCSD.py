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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Spider text-to-SQL benchmark. Output: SQL: <<QUERY>>. QUERY must be COMPLETE. Clause rules: (1) aggregation words (most, least, count, total, average, sum, min, max, number of, more than, fewer than) -> use COUNT/SUM/AVG/MIN/MAX plus GROUP BY column plus HAVING condition; (2) multiple entities or tables -> JOIN each table ON condition; (3) filter or condition words -> WHERE clause; (4) ranked or ordered results -> ORDER BY column ASC/DESC; (5) unique or distinct values -> SELECT DISTINCT. Use ONLY exact table and column names from the provided schema. Do not stop generating after a bare SELECT...FROM without all required clauses.")))
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
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_6_remaining_: int
            d_6_remaining_ = (maxSteps) - (d_1_steps_)
            if (d_6_remaining_) >= (2):
                d_7_rawBudget_: int
                d_7_rawBudget_ = (d_6_remaining_) - (1)
                d_8_sqlBudget_: int = int(0)
                if (d_7_rawBudget_) <= (400):
                    d_8_sqlBudget_ = d_7_rawBudget_
                elif True:
                    d_8_sqlBudget_ = 400
                d_9_constrainedPrompt_: _dafny.Seq
                d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_10_resultConstrained_: _dafny.Seq
                out4_: _dafny.Seq
                out4_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken, d_8_sqlBudget_, 5, 50)
                d_10_resultConstrained_ = out4_
                d_11_stablePrefix_: _dafny.Seq
                d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                generated = (d_11_stablePrefix_) + (d_10_resultConstrained_)
                currentConstrainedOut = d_10_resultConstrained_
                d_1_steps_ = (d_1_steps_) + (d_8_sqlBudget_)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_12_cg_: _dafny.Seq
                d_13_ci_: bool
                d_14_cc_: _dafny.Seq
                out5_: _dafny.Seq
                out6_: bool
                out7_: _dafny.Seq
                out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_12_cg_ = out5_
                d_13_ci_ = out6_
                d_14_cc_ = out7_
                generated = d_12_cg_
                insideConstrainedOut = d_13_ci_
                currentConstrainedOut = d_14_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_15_remaining_: int
                d_15_remaining_ = (maxSteps) - (d_1_steps_)
                d_16_closeBudget_: int = int(0)
                if (d_15_remaining_) <= (80):
                    d_16_closeBudget_ = d_15_remaining_
                elif True:
                    d_16_closeBudget_ = 80
                d_17_cg_: _dafny.Seq
                d_18_ci_: bool
                d_19_cc_: _dafny.Seq
                out8_: _dafny.Seq
                out9_: bool
                out10_: _dafny.Seq
                out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_16_closeBudget_)
                d_17_cg_ = out8_
                d_18_ci_ = out9_
                d_19_cc_ = out10_
                generated = d_17_cg_
                insideConstrainedOut = d_18_ci_
                currentConstrainedOut = d_19_cc_
                d_1_steps_ = (d_1_steps_) + (d_16_closeBudget_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


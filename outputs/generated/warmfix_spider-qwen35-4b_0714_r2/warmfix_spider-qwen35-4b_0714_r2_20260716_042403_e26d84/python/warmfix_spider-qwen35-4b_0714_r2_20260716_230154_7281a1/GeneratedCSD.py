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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write SQL to answer the question. Output format: SQL: <<QUERY>> where QUERY is a syntactically complete SQL statement. For each question: include SELECT with the exact columns asked about; include FROM with the correct table(s) from the schema; add JOIN ... ON ... when the answer requires multiple tables; add WHERE for filter conditions; add GROUP BY + HAVING for aggregation questions (most, average, more than N, count); add ORDER BY for ranking/sorting. Use exact table and column names from the provided schema. Generate all necessary clauses - do not stop after a bare SELECT col FROM table.")))
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
        d_6_minTokens_: int
        d_6_minTokens_ = 25
        d_7_sqlMaxSteps_: int = int(0)
        if (maxSteps) > ((d_1_steps_) + (1)):
            d_8_headroom_: int
            d_8_headroom_ = ((maxSteps) - (d_1_steps_)) - (1)
            if (d_8_headroom_) >= (350):
                d_7_sqlMaxSteps_ = 350
            elif True:
                d_7_sqlMaxSteps_ = d_8_headroom_
        elif True:
            d_7_sqlMaxSteps_ = 0
        d_9_sqlSteps_: int
        d_9_sqlSteps_ = 0
        with _dafny.label("1"):
            while ((d_9_sqlSteps_) < (d_7_sqlMaxSteps_)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    d_10_constrainedPrompt_: _dafny.Seq
                    d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_11_next_: _dafny.Seq
                    out4_: _dafny.Seq
                    out4_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                    d_11_next_ = out4_
                    d_9_sqlSteps_ = (d_9_sqlSteps_) + (1)
                    if (d_11_next_) == (eosToken):
                        if (len(currentConstrainedOut)) >= (d_6_minTokens_):
                            raise _dafny.Break("1")
                        elif True:
                            d_12_narrow_: bool
                            out5_: bool
                            out5_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_12_narrow_ = out5_
                            if d_12_narrow_:
                                raise _dafny.Break("1")
                    elif True:
                        d_13_valid_: bool
                        out6_: bool
                        out6_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_11_next_)
                        d_13_valid_ = out6_
                        if d_13_valid_:
                            d_14_ag_: _dafny.Seq
                            d_15_ai_: bool
                            d_16_ac_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                            d_14_ag_ = out7_
                            d_15_ai_ = out8_
                            d_16_ac_ = out9_
                            generated = d_14_ag_
                            insideConstrainedOut = d_15_ai_
                            currentConstrainedOut = d_16_ac_
                    pass
            pass
        d_1_steps_ = (d_1_steps_) + (d_9_sqlSteps_)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_17_cg_: _dafny.Seq
                d_18_ci_: bool
                d_19_cc_: _dafny.Seq
                out10_: _dafny.Seq
                out11_: bool
                out12_: _dafny.Seq
                out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_17_cg_ = out10_
                d_18_ci_ = out11_
                d_19_cc_ = out12_
                generated = d_17_cg_
                insideConstrainedOut = d_18_ci_
                currentConstrainedOut = d_19_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_20_remaining_: int
                d_20_remaining_ = (maxSteps) - (d_1_steps_)
                d_21_closeBudget_: int = int(0)
                if (d_20_remaining_) <= (80):
                    d_21_closeBudget_ = d_20_remaining_
                elif True:
                    d_21_closeBudget_ = 80
                d_22_cg_: _dafny.Seq
                d_23_ci_: bool
                d_24_cc_: _dafny.Seq
                out13_: _dafny.Seq
                out14_: bool
                out15_: _dafny.Seq
                out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_21_closeBudget_)
                d_22_cg_ = out13_
                d_23_ci_ = out14_
                d_24_cc_ = out15_
                generated = d_22_cg_
                insideConstrainedOut = d_23_ci_
                currentConstrainedOut = d_24_cc_
                d_1_steps_ = (d_1_steps_) + (d_21_closeBudget_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


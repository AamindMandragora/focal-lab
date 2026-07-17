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
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SQLite query using only the provided schema context. Output exactly SQL: <<YOUR QUERY>> with no explanation, no Markdown, and no extra text. The decoder will force SQL: << and close with >>. Inside the span, produce only the SQL query. Use exact table and column names from the schema. Prefer the simplest semantically correct SQLite query. Use COUNT(*) for total counts when appropriate, COUNT(DISTINCT ...) for distinct counts, and explicit JOIN conditions when columns come from different tables. Avoid semicolons."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Treat contextual token groups as schema identifier hints, but only use them when they match the question and schema.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_headerStage_: int
        d_2_headerStage_ = 0
        d_3_steps_: int
        d_3_steps_ = 0
        with _dafny.label("0"):
            while (d_3_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_headerStage_) == (0):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:"))]))
                            d_2_headerStage_ = 1
                            d_3_steps_ = (d_3_steps_) + (1)
                        elif (d_2_headerStage_) == (1):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))]))
                            d_2_headerStage_ = 2
                            d_3_steps_ = (d_3_steps_) + (1)
                        elif True:
                            d_4_openedGenerated_: _dafny.Seq
                            d_5_openedInside_: bool
                            d_6_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openedGenerated_ = out0_
                            d_5_openedInside_ = out1_
                            d_6_openedCurrent_ = out2_
                            generated = d_4_openedGenerated_
                            insideConstrainedOut = d_5_openedInside_
                            currentConstrainedOut = d_6_openedCurrent_
                            d_3_steps_ = (d_3_steps_) + (1)
                    elif True:
                        d_7_constrainedPrompt_: _dafny.Seq
                        d_7_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_8_next_: _dafny.Seq
                            d_9_wasConstrained_: bool
                            out3_: _dafny.Seq
                            out4_: bool
                            out3_, out4_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_8_next_ = out3_
                            d_9_wasConstrained_ = out4_
                            d_3_steps_ = (d_3_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                if (d_3_steps_) < (maxSteps):
                                    d_10_closedGenerated_: _dafny.Seq
                                    d_11_closedInside_: bool
                                    d_12_closedCurrent_: _dafny.Seq
                                    out5_: _dafny.Seq
                                    out6_: bool
                                    out7_: _dafny.Seq
                                    out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_10_closedGenerated_ = out5_
                                    d_11_closedInside_ = out6_
                                    d_12_closedCurrent_ = out7_
                                    generated = d_10_closedGenerated_
                                    insideConstrainedOut = d_11_closedInside_
                                    currentConstrainedOut = d_12_closedCurrent_
                                    d_3_steps_ = (d_3_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                d_13_valid_: bool
                                out8_: bool
                                out8_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_8_next_)
                                d_13_valid_ = out8_
                                if d_13_valid_:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                    currentConstrainedOut = (currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                elif True:
                                    if (d_3_steps_) < (maxSteps):
                                        d_14_closedGenerated2_: _dafny.Seq
                                        d_15_closedInside2_: bool
                                        d_16_closedCurrent2_: _dafny.Seq
                                        out9_: _dafny.Seq
                                        out10_: bool
                                        out11_: _dafny.Seq
                                        out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_14_closedGenerated2_ = out9_
                                        d_15_closedInside2_ = out10_
                                        d_16_closedCurrent2_ = out11_
                                        generated = d_14_closedGenerated2_
                                        insideConstrainedOut = d_15_closedInside2_
                                        currentConstrainedOut = d_16_closedCurrent2_
                                        d_3_steps_ = (d_3_steps_) + (1)
                                    raise _dafny.Break("0")
                        elif True:
                            d_17_nextSql_: _dafny.Seq
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'), 12, eosToken)
                            d_17_nextSql_ = out12_
                            d_3_steps_ = (d_3_steps_) + (1)
                            if (d_17_nextSql_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_18_appendedGenerated_: _dafny.Seq
                                d_19_appendedInside_: bool
                                d_20_appendedCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_nextSql_)
                                d_18_appendedGenerated_ = out13_
                                d_19_appendedInside_ = out14_
                                d_20_appendedCurrent_ = out15_
                                generated = d_18_appendedGenerated_
                                insideConstrainedOut = d_19_appendedInside_
                                currentConstrainedOut = d_20_appendedCurrent_
                    pass
            pass
        if (insideConstrainedOut) and ((d_3_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_21_finalGenerated_: _dafny.Seq
                d_22_finalInside_: bool
                d_23_finalCurrent_: _dafny.Seq
                out16_: _dafny.Seq
                out17_: bool
                out18_: _dafny.Seq
                out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_21_finalGenerated_ = out16_
                d_22_finalInside_ = out17_
                d_23_finalCurrent_ = out18_
                generated = d_21_finalGenerated_
                insideConstrainedOut = d_22_finalInside_
                currentConstrainedOut = d_23_finalCurrent_
                d_3_steps_ = (d_3_steps_) + (1)
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                d_3_steps_ = (d_3_steps_) + (1)
        cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


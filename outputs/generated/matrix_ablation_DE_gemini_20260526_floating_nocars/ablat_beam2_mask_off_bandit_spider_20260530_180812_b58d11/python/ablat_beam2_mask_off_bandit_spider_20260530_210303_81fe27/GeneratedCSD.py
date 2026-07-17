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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SQLite query using only the provided schema context. Output exactly SQL: <<YOUR QUERY>> with no explanation, no Markdown, and no extra text. The decoder will provide SQL: << and the final >>; inside the span produce only the SQL query. Prefer the simplest semantically correct Spider-style query. Use exact table and column names from the schema. Avoid unnecessary table aliases; use explicit table.column names unless aliases are needed for self-joins or disambiguation. For questions about records not appearing in another set, consider EXCEPT or NOT IN only when semantically appropriate. For questions requiring both conditions, consider INTERSECT or GROUP BY/HAVING when appropriate. Include all required joins, filters, grouping, ordering, limits, set operations, and aggregation, and do not invent tables or columns."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Contextual token groups are schema hints; prefer them only when they match the question and schema.")))
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
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_7_closedGenerated_: _dafny.Seq
                        d_8_closedInside_: bool
                        d_9_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_7_closedGenerated_ = out3_
                        d_8_closedInside_ = out4_
                        d_9_closedCurrent_ = out5_
                        generated = d_7_closedGenerated_
                        insideConstrainedOut = d_8_closedInside_
                        currentConstrainedOut = d_9_closedCurrent_
                        d_3_steps_ = (d_3_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_11_next_: _dafny.Seq
                        d_11_next_ = eosToken
                        d_12_gatedNext_: _dafny.Seq
                        d_13_wasConstrained_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out6_, out7_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_12_gatedNext_ = out6_
                        d_13_wasConstrained_ = out7_
                        d_11_next_ = d_12_gatedNext_
                        d_3_steps_ = (d_3_steps_) + (1)
                        if (d_11_next_) == (eosToken):
                            if (d_3_steps_) < (maxSteps):
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e1'), 6, eosToken)
                                d_11_next_ = out8_
                                d_3_steps_ = (d_3_steps_) + (1)
                                if (d_11_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_14_appendedGenerated_: _dafny.Seq
                                    d_15_appendedInside_: bool
                                    d_16_appendedCurrent_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                                    d_14_appendedGenerated_ = out9_
                                    d_15_appendedInside_ = out10_
                                    d_16_appendedCurrent_ = out11_
                                    generated = d_14_appendedGenerated_
                                    insideConstrainedOut = d_15_appendedInside_
                                    currentConstrainedOut = d_16_appendedCurrent_
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_17_appendedGenerated2_: _dafny.Seq
                            d_18_appendedInside2_: bool
                            d_19_appendedCurrent2_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                            d_17_appendedGenerated2_ = out12_
                            d_18_appendedInside2_ = out13_
                            d_19_appendedCurrent2_ = out14_
                            generated = d_17_appendedGenerated2_
                            insideConstrainedOut = d_18_appendedInside2_
                            currentConstrainedOut = d_19_appendedCurrent2_
                    pass
            pass
        cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


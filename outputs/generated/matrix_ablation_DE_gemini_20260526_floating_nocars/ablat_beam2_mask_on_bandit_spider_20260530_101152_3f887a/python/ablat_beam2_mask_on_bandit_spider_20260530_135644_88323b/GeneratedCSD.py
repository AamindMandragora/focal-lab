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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one SQL query using only the provided schema context. The complete answer must be exactly SQL: <<YOUR QUERY>>. Inside the delimiters emit only the SQL body: no explanation, Markdown, comments, or extra prose. Prefer schema table and column names exactly as given. Use joins when selected columns and filter columns come from different tables. For questions requiring both of two values, use INTERSECT or GROUP BY/HAVING rather than simple OR logic. Do not add a trailing semicolon."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Use the supplied schema context when it matches the question, but keep the model's SQL plan faithful to the question.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_sqlHeaderCount_: int
        out0_: int
        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL: ")))
        d_2_sqlHeaderCount_ = out0_
        d_3_openCount_: int
        out1_: int
        out1_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        d_3_openCount_ = out1_
        d_4_headerDone_: bool
        d_4_headerDone_ = (insideConstrainedOut) or ((d_2_sqlHeaderCount_) > (0))
        d_5_spanStarted_: bool
        d_5_spanStarted_ = (insideConstrainedOut) or ((d_3_openCount_) > (0))
        d_6_eosPenaltyTokens_: _dafny.Seq
        d_6_eosPenaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "```")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";"))])
        d_7_steps_: int
        d_7_steps_ = 0
        with _dafny.label("0"):
            while (d_7_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_4_headerDone_):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL: "))]))
                            d_4_headerDone_ = True
                            d_7_steps_ = (d_7_steps_) + (1)
                        elif not(d_5_spanStarted_):
                            d_8_openedGenerated_: _dafny.Seq
                            d_9_openedInside_: bool
                            d_10_openedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_openedGenerated_ = out2_
                            d_9_openedInside_ = out3_
                            d_10_openedCurrent_ = out4_
                            generated = d_8_openedGenerated_
                            insideConstrainedOut = d_9_openedInside_
                            currentConstrainedOut = d_10_openedCurrent_
                            d_5_spanStarted_ = True
                            d_7_steps_ = (d_7_steps_) + (1)
                        elif True:
                            d_11_sink_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_sink_ = out5_
                            d_7_steps_ = (d_7_steps_) + (1)
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_closedGenerated_: _dafny.Seq
                        d_13_closedInside_: bool
                        d_14_closedCurrent_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_closedGenerated_ = out6_
                        d_13_closedInside_ = out7_
                        d_14_closedCurrent_ = out8_
                        generated = d_12_closedGenerated_
                        insideConstrainedOut = d_13_closedInside_
                        currentConstrainedOut = d_14_closedCurrent_
                        d_7_steps_ = (d_7_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_16_gatedNext_: _dafny.Seq
                        d_17_wasConstrained_: bool
                        out9_: _dafny.Seq
                        out10_: bool
                        out9_, out10_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_16_gatedNext_ = out9_
                        d_17_wasConstrained_ = out10_
                        d_18_next_: _dafny.Seq
                        d_18_next_ = d_16_gatedNext_
                        d_7_steps_ = (d_7_steps_) + (1)
                        if ((d_18_next_) == (eosToken)) and ((d_7_steps_) < (maxSteps)):
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([]), _dafny.BigRational('0e0'), d_6_eosPenaltyTokens_, _dafny.BigRational('1e1'), 0, eosToken)
                            d_18_next_ = out11_
                            d_7_steps_ = (d_7_steps_) + (1)
                        if (d_18_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_19_appendedGenerated_: _dafny.Seq
                            d_20_appendedInside_: bool
                            d_21_appendedCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                            d_19_appendedGenerated_ = out12_
                            d_20_appendedInside_ = out13_
                            d_21_appendedCurrent_ = out14_
                            generated = d_19_appendedGenerated_
                            insideConstrainedOut = d_20_appendedInside_
                            currentConstrainedOut = d_21_appendedCurrent_
                            if ((d_7_steps_) < (maxSteps)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_22_closedGenerated2_: _dafny.Seq
                                d_23_closedInside2_: bool
                                d_24_closedCurrent2_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_22_closedGenerated2_ = out15_
                                d_23_closedInside2_ = out16_
                                d_24_closedCurrent2_ = out17_
                                generated = d_22_closedGenerated2_
                                insideConstrainedOut = d_23_closedInside2_
                                currentConstrainedOut = d_24_closedCurrent2_
                                d_7_steps_ = (d_7_steps_) + (1)
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_7_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


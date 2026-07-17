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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one SQL query using only the provided schema context. The complete answer must be exactly SQL: <<YOUR QUERY>>. Inside the delimiters emit only the SQL body: no explanation, Markdown, comments, or extra prose. Prefer standard Spider SQL and schema names exactly as given. Avoid table aliases unless a self-join makes them necessary. For either/or value conditions, prefer explicit OR comparisons over IN lists. For questions requiring both of two values, use INTERSECT or GROUP BY/HAVING rather than simple either-value logic. Do not add a trailing semicolon. Close the query only when it is complete."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Use the supplied schema context naturally, but do not overfit to token groups when the question points elsewhere.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_openCount_: int
        out0_: int
        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        d_2_openCount_ = out0_
        d_3_headerDone_: bool
        d_3_headerDone_ = (insideConstrainedOut) or ((len(generated)) > (0))
        d_4_spanStarted_: bool
        d_4_spanStarted_ = (insideConstrainedOut) or ((d_2_openCount_) > (0))
        d_5_eosPenaltyTokens_: _dafny.Seq
        d_5_eosPenaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "```")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";"))])
        d_6_localLimit_: int
        d_6_localLimit_ = maxSteps
        if (d_6_localLimit_) > (128):
            d_6_localLimit_ = 128
        d_7_steps_: int
        d_7_steps_ = 0
        with _dafny.label("0"):
            while (d_7_steps_) < (d_6_localLimit_):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_3_headerDone_):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL: "))]))
                            d_3_headerDone_ = True
                            d_7_steps_ = (d_7_steps_) + (1)
                        elif not(d_4_spanStarted_):
                            d_8_openedGenerated_: _dafny.Seq
                            d_9_openedInside_: bool
                            d_10_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_openedGenerated_ = out1_
                            d_9_openedInside_ = out2_
                            d_10_openedCurrent_ = out3_
                            generated = d_8_openedGenerated_
                            insideConstrainedOut = d_9_openedInside_
                            currentConstrainedOut = d_10_openedCurrent_
                            d_4_spanStarted_ = True
                            d_7_steps_ = (d_7_steps_) + (1)
                        elif True:
                            d_11_sink_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_sink_ = out4_
                            d_7_steps_ = (d_7_steps_) + (1)
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_closedGenerated_: _dafny.Seq
                        d_13_closedInside_: bool
                        d_14_closedCurrent_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_closedGenerated_ = out5_
                        d_13_closedInside_ = out6_
                        d_14_closedCurrent_ = out7_
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
                        out8_: _dafny.Seq
                        out9_: bool
                        out8_, out9_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_16_gatedNext_ = out8_
                        d_17_wasConstrained_ = out9_
                        d_18_next_: _dafny.Seq
                        d_18_next_ = d_16_gatedNext_
                        d_7_steps_ = (d_7_steps_) + (1)
                        if ((d_18_next_) == (eosToken)) and ((d_7_steps_) < (d_6_localLimit_)):
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), d_5_eosPenaltyTokens_, _dafny.BigRational('1e1'), 8, eosToken)
                            d_18_next_ = out10_
                            d_7_steps_ = (d_7_steps_) + (1)
                        if (d_18_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_19_appendedGenerated_: _dafny.Seq
                            d_20_appendedInside_: bool
                            d_21_appendedCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                            d_19_appendedGenerated_ = out11_
                            d_20_appendedInside_ = out12_
                            d_21_appendedCurrent_ = out13_
                            generated = d_19_appendedGenerated_
                            insideConstrainedOut = d_20_appendedInside_
                            currentConstrainedOut = d_21_appendedCurrent_
                            if ((d_7_steps_) < (d_6_localLimit_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_22_closedGenerated2_: _dafny.Seq
                                d_23_closedInside2_: bool
                                d_24_closedCurrent2_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_22_closedGenerated2_ = out14_
                                d_23_closedInside2_ = out15_
                                d_24_closedCurrent2_ = out16_
                                generated = d_22_closedGenerated2_
                                insideConstrainedOut = d_23_closedInside2_
                                currentConstrainedOut = d_24_closedCurrent2_
                                d_7_steps_ = (d_7_steps_) + (1)
                                raise _dafny.Break("0")
                    pass
            pass
        if (((d_7_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
            d_25_finalClosedGenerated_: _dafny.Seq
            d_26_finalClosedInside_: bool
            d_27_finalClosedCurrent_: _dafny.Seq
            out17_: _dafny.Seq
            out18_: bool
            out19_: _dafny.Seq
            out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_25_finalClosedGenerated_ = out17_
            d_26_finalClosedInside_ = out18_
            d_27_finalClosedCurrent_ = out19_
            generated = d_25_finalClosedGenerated_
            insideConstrainedOut = d_26_finalClosedInside_
            currentConstrainedOut = d_27_finalClosedCurrent_
            d_7_steps_ = (d_7_steps_) + (1)
        cost = d_7_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


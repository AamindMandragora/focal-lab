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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly SQL: <<query>>. Inside << >> put only one SQL query using the given schema; no explanation, Markdown, comments, or semicolon. Prefer exact table and column names. Avoid aliases unless needed for self-joins. Use DISTINCT only if asked. For both-of-two conditions use INTERSECT or GROUP BY/HAVING; use linking tables for many-to-many joins."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Prefer schema tokens from the contextual groups when they fit.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_openCount_: int
        out0_: int
        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        d_2_openCount_ = out0_
        d_3_headerDone_: bool
        d_3_headerDone_ = (insideConstrainedOut) or ((len(generated)) > (0))
        d_4_spanStarted_: bool
        d_4_spanStarted_ = (insideConstrainedOut) or ((d_2_openCount_) > (0))
        d_5_noGroups_: _dafny.Seq
        d_5_noGroups_ = _dafny.SeqWithoutIsStrInference([])
        d_6_eosPenaltyTokens_: _dafny.Seq
        d_6_eosPenaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "```")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";"))])
        d_7_narrowThreshold_: int
        d_7_narrowThreshold_ = 12
        d_8_localLimit_: int
        d_8_localLimit_ = maxSteps
        if (d_8_localLimit_) > (90):
            d_8_localLimit_ = 90
        d_9_steps_: int
        d_9_steps_ = 0
        with _dafny.label("0"):
            while (d_9_steps_) < (d_8_localLimit_):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_3_headerDone_):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL: "))]))
                            d_3_headerDone_ = True
                            d_9_steps_ = (d_9_steps_) + (1)
                        elif not(d_4_spanStarted_):
                            d_10_openedGenerated_: _dafny.Seq
                            d_11_openedInside_: bool
                            d_12_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_openedGenerated_ = out1_
                            d_11_openedInside_ = out2_
                            d_12_openedCurrent_ = out3_
                            generated = d_10_openedGenerated_
                            insideConstrainedOut = d_11_openedInside_
                            currentConstrainedOut = d_12_openedCurrent_
                            d_4_spanStarted_ = True
                            d_9_steps_ = (d_9_steps_) + (1)
                        elif True:
                            d_13_sink_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_13_sink_ = out4_
                            d_9_steps_ = (d_9_steps_) + (1)
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_14_closedGenerated_: _dafny.Seq
                        d_15_closedInside_: bool
                        d_16_closedCurrent_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_closedGenerated_ = out5_
                        d_15_closedInside_ = out6_
                        d_16_closedCurrent_ = out7_
                        generated = d_14_closedGenerated_
                        insideConstrainedOut = d_15_closedInside_
                        currentConstrainedOut = d_16_closedCurrent_
                        d_9_steps_ = (d_9_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_18_gatedNext_: _dafny.Seq
                        d_19_wasConstrained_: bool
                        out8_: _dafny.Seq
                        out9_: bool
                        out8_, out9_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_18_gatedNext_ = out8_
                        d_19_wasConstrained_ = out9_
                        d_20_next_: _dafny.Seq
                        d_20_next_ = d_18_gatedNext_
                        d_9_steps_ = (d_9_steps_) + (1)
                        if ((d_20_next_) == (eosToken)) and ((d_9_steps_) < (d_8_localLimit_)):
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, d_5_noGroups_, _dafny.BigRational('0e0'), d_6_eosPenaltyTokens_, _dafny.BigRational('8e0'), d_7_narrowThreshold_, eosToken)
                            d_20_next_ = out10_
                            d_9_steps_ = (d_9_steps_) + (1)
                        if (d_20_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_21_appendedGenerated_: _dafny.Seq
                            d_22_appendedInside_: bool
                            d_23_appendedCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                            d_21_appendedGenerated_ = out11_
                            d_22_appendedInside_ = out12_
                            d_23_appendedCurrent_ = out13_
                            generated = d_21_appendedGenerated_
                            insideConstrainedOut = d_22_appendedInside_
                            currentConstrainedOut = d_23_appendedCurrent_
                            if ((d_9_steps_) < (d_8_localLimit_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_24_closedGenerated2_: _dafny.Seq
                                d_25_closedInside2_: bool
                                d_26_closedCurrent2_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_24_closedGenerated2_ = out14_
                                d_25_closedInside2_ = out15_
                                d_26_closedCurrent2_ = out16_
                                generated = d_24_closedGenerated2_
                                insideConstrainedOut = d_25_closedInside2_
                                currentConstrainedOut = d_26_closedCurrent2_
                                d_9_steps_ = (d_9_steps_) + (1)
                                raise _dafny.Break("0")
                    pass
            pass
        if (((d_9_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
            d_27_finalClosedGenerated_: _dafny.Seq
            d_28_finalClosedInside_: bool
            d_29_finalClosedCurrent_: _dafny.Seq
            out17_: _dafny.Seq
            out18_: bool
            out19_: _dafny.Seq
            out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_27_finalClosedGenerated_ = out17_
            d_28_finalClosedInside_ = out18_
            d_29_finalClosedCurrent_ = out19_
            generated = d_27_finalClosedGenerated_
            insideConstrainedOut = d_28_finalClosedInside_
            currentConstrainedOut = d_29_finalClosedCurrent_
            d_9_steps_ = (d_9_steps_) + (1)
        cost = d_9_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


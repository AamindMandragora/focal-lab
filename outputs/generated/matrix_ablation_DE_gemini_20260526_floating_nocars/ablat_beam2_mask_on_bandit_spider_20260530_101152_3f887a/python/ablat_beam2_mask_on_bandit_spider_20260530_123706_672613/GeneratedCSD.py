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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Return exactly SQL: <<SQL query>>. Use only the provided schema. Inside << >> emit only the SQL query: no prose, Markdown, comments, or trailing semicolon. Prefer exact table and column names; use JOIN ... ON for multi-table questions; use INTERSECT for conditions asking for both values."))
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
        d_6_emptyGroups_: _dafny.Seq
        d_6_emptyGroups_ = _dafny.SeqWithoutIsStrInference([])
        d_7_localStepCap_: int
        d_7_localStepCap_ = maxSteps
        if (d_7_localStepCap_) > (180):
            d_7_localStepCap_ = 180
        d_8_steps_: int
        d_8_steps_ = 0
        with _dafny.label("0"):
            while (d_8_steps_) < (d_7_localStepCap_):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_3_headerDone_):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL: "))]))
                            d_3_headerDone_ = True
                            d_8_steps_ = (d_8_steps_) + (1)
                        elif not(d_4_spanStarted_):
                            d_9_openedGenerated_: _dafny.Seq
                            d_10_openedInside_: bool
                            d_11_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_9_openedGenerated_ = out1_
                            d_10_openedInside_ = out2_
                            d_11_openedCurrent_ = out3_
                            generated = d_9_openedGenerated_
                            insideConstrainedOut = d_10_openedInside_
                            currentConstrainedOut = d_11_openedCurrent_
                            d_4_spanStarted_ = True
                            d_8_steps_ = (d_8_steps_) + (1)
                        elif True:
                            d_12_sink_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_sink_ = out4_
                            d_8_steps_ = (d_8_steps_) + (1)
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_closedGenerated_: _dafny.Seq
                        d_14_closedInside_: bool
                        d_15_closedCurrent_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_closedGenerated_ = out5_
                        d_14_closedInside_ = out6_
                        d_15_closedCurrent_ = out7_
                        generated = d_13_closedGenerated_
                        insideConstrainedOut = d_14_closedInside_
                        currentConstrainedOut = d_15_closedCurrent_
                        d_8_steps_ = (d_8_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_16_constrainedPrompt_: _dafny.Seq
                        d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_17_gatedNext_: _dafny.Seq
                        d_18_wasConstrained_: bool
                        out8_: _dafny.Seq
                        out9_: bool
                        out8_, out9_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_17_gatedNext_ = out8_
                        d_18_wasConstrained_ = out9_
                        d_19_next_: _dafny.Seq
                        d_19_next_ = d_17_gatedNext_
                        d_8_steps_ = (d_8_steps_) + (1)
                        if ((d_19_next_) == (eosToken)) and ((d_8_steps_) < (d_7_localStepCap_)):
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, d_6_emptyGroups_, _dafny.BigRational('0e0'), d_5_eosPenaltyTokens_, _dafny.BigRational('1e1'), 0, eosToken)
                            d_19_next_ = out10_
                            d_8_steps_ = (d_8_steps_) + (1)
                        if (d_19_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_20_appendedGenerated_: _dafny.Seq
                            d_21_appendedInside_: bool
                            d_22_appendedCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                            d_20_appendedGenerated_ = out11_
                            d_21_appendedInside_ = out12_
                            d_22_appendedCurrent_ = out13_
                            generated = d_20_appendedGenerated_
                            insideConstrainedOut = d_21_appendedInside_
                            currentConstrainedOut = d_22_appendedCurrent_
                            if ((d_8_steps_) < (d_7_localStepCap_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_23_closedGenerated2_: _dafny.Seq
                                d_24_closedInside2_: bool
                                d_25_closedCurrent2_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_23_closedGenerated2_ = out14_
                                d_24_closedInside2_ = out15_
                                d_25_closedCurrent2_ = out16_
                                generated = d_23_closedGenerated2_
                                insideConstrainedOut = d_24_closedInside2_
                                currentConstrainedOut = d_25_closedCurrent2_
                                d_8_steps_ = (d_8_steps_) + (1)
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_8_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


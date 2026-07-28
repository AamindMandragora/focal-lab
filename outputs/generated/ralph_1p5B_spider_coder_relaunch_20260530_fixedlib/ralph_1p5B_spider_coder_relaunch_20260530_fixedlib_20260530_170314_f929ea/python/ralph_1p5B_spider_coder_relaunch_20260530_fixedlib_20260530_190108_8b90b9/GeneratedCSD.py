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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<QUERY>> where QUERY is a valid SQL SELECT statement using the schema. No explanation. Keep the query concise and correct.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanLength_: int
        d_2_spanLength_ = 0
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_3_next_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_3_next_ = out0_
            d_1_steps_ = (d_1_steps_) + (1)
            if (d_3_next_) == (eosToken):
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                insideConstrainedOut = True
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                d_2_spanLength_ = 0
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_4_openedGenerated_: _dafny.Seq
            d_5_openedInside_: bool
            d_6_openedCurrent_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_4_openedGenerated_ = out1_
            d_5_openedInside_ = out2_
            d_6_openedCurrent_ = out3_
            generated = d_4_openedGenerated_
            insideConstrainedOut = d_5_openedInside_
            currentConstrainedOut = d_6_openedCurrent_
            d_1_steps_ = (d_1_steps_) + (1)
            d_2_spanLength_ = 0
        d_7_penaltyTokens_: _dafny.Seq
        d_7_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIKE"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_8_closedGenerated_: _dafny.Seq
                        d_9_closedInside_: bool
                        d_10_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_8_closedGenerated_ = out4_
                        d_9_closedInside_ = out5_
                        d_10_closedCurrent_ = out6_
                        generated = d_8_closedGenerated_
                        insideConstrainedOut = d_9_closedInside_
                        currentConstrainedOut = d_10_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_12_next_: _dafny.Seq
                        d_12_next_ = eosToken
                        if (d_2_spanLength_) < (60):
                            d_13_wasConstrained_: bool = False
                            out7_: _dafny.Seq
                            out8_: bool
                            out7_, out8_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_12_next_ = out7_
                            d_13_wasConstrained_ = out8_
                        elif True:
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), d_7_penaltyTokens_, _dafny.BigRational('8e0'), 8, eosToken)
                            d_12_next_ = out9_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_12_next_) == (eosToken):
                            if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
                                d_14_closedGenerated_: _dafny.Seq
                                d_15_closedInside_: bool
                                d_16_closedCurrent_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_14_closedGenerated_ = out10_
                                d_15_closedInside_ = out11_
                                d_16_closedCurrent_ = out12_
                                generated = d_14_closedGenerated_
                                insideConstrainedOut = d_15_closedInside_
                                currentConstrainedOut = d_16_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_17_appendedGenerated_: _dafny.Seq
                            d_18_appendedInside_: bool
                            d_19_appendedCurrent_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                            d_17_appendedGenerated_ = out13_
                            d_18_appendedInside_ = out14_
                            d_19_appendedCurrent_ = out15_
                            generated = d_17_appendedGenerated_
                            insideConstrainedOut = d_18_appendedInside_
                            currentConstrainedOut = d_19_appendedCurrent_
                            d_2_spanLength_ = (d_2_spanLength_) + (1)
                    pass
            pass
        if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
            d_20_closedGenerated_: _dafny.Seq
            d_21_closedInside_: bool
            d_22_closedCurrent_: _dafny.Seq
            out16_: _dafny.Seq
            out17_: bool
            out18_: _dafny.Seq
            out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_20_closedGenerated_ = out16_
            d_21_closedInside_ = out17_
            d_22_closedCurrent_ = out18_
            generated = d_20_closedGenerated_
            insideConstrainedOut = d_21_closedInside_
            currentConstrainedOut = d_22_closedCurrent_
            d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


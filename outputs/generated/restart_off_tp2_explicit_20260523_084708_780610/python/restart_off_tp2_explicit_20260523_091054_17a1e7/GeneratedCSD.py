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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step. Wrap every arithmetic computation and the final numeric answer inside << and >> delimiters, for example <<2+3=5>>. After the equals sign and numeric result, immediately close with >>. Keep each <<...>> short.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanLenLimit_: int
        d_2_spanLenLimit_ = 18
        d_3_outsideChunkCap_: int
        d_3_outsideChunkCap_ = 40
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_4_remaining_: int
                    d_4_remaining_ = (maxSteps) - (d_1_steps_)
                    if not(insideConstrainedOut):
                        d_5_cb_: int
                        if (d_4_remaining_) < (d_3_outsideChunkCap_):
                            d_5_cb_ = d_4_remaining_
                        elif True:
                            d_5_cb_ = d_3_outsideChunkCap_
                        d_6_chunkedG_: _dafny.Seq
                        d_7_stoppedOpen_: bool
                        d_8_stoppedEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_cb_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_chunkedG_ = out0_
                        d_7_stoppedOpen_ = out1_
                        d_8_stoppedEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        generated = d_6_chunkedG_
                        d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                        if d_8_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_7_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif (d_9_stepsUsed_) == (0):
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_10_closedGenerated_: _dafny.Seq
                        d_11_closedInside_: bool
                        d_12_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_10_closedGenerated_ = out4_
                        d_11_closedInside_ = out5_
                        d_12_closedCurrent_ = out6_
                        generated = d_10_closedGenerated_
                        insideConstrainedOut = d_11_closedInside_
                        currentConstrainedOut = d_12_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_2_spanLenLimit_):
                        d_13_rolledGenerated_: _dafny.Seq
                        d_14_rolledCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: _dafny.Seq
                        out7_, out8_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_13_rolledGenerated_ = out7_
                        d_14_rolledCurrent_ = out8_
                        generated = d_13_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_14_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_16_spanStr_: _dafny.Seq
                        d_16_spanStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                        d_17_equalsCount_: int
                        d_17_equalsCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_16_spanStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                        d_18_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (len(currentConstrainedOut)) < (2):
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!!!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">"))]), _dafny.BigRational('6e0'), eosToken)
                            d_18_next_ = out9_
                        elif True:
                            d_19_groups_: _dafny.Seq
                            d_19_groups_ = validTokenGroups
                            if (d_17_equalsCount_) >= (1):
                                d_19_groups_ = (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">"))])])) + (d_19_groups_)
                            elif (len(currentConstrainedOut)) >= (4):
                                d_19_groups_ = (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))])])) + (d_19_groups_)
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, d_19_groups_, _dafny.BigRational('8e0'), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!!!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('6e0'), 100, eosToken)
                            d_18_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_18_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_20_appendedGenerated_: _dafny.Seq
                            d_21_appendedInside_: bool
                            d_22_appendedCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                            d_20_appendedGenerated_ = out11_
                            d_21_appendedInside_ = out12_
                            d_22_appendedCurrent_ = out13_
                            generated = d_20_appendedGenerated_
                            insideConstrainedOut = d_21_appendedInside_
                            currentConstrainedOut = d_22_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


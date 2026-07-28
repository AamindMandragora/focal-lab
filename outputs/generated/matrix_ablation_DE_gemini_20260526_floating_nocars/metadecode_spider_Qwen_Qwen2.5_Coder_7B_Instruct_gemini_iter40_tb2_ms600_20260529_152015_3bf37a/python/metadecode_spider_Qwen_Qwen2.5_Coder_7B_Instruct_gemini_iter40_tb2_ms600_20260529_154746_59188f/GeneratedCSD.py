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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_seenFrom_: bool
        d_2_seenFrom_ = False
        if (not(insideConstrained)) and ((maxSteps) > (0)):
            d_3_chunkBudget_: int
            if (5) < (maxSteps):
                d_3_chunkBudget_ = 5
            elif True:
                d_3_chunkBudget_ = maxSteps
            d_4_generatedOut_: _dafny.Seq
            d_5_stoppedOnOpen_: bool
            d_6_stoppedOnEos_: bool
            d_7_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_4_generatedOut_ = out0_
            d_5_stoppedOnOpen_ = out1_
            d_6_stoppedOnEos_ = out2_
            d_7_stepsUsed_ = out3_
            d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
            generated = d_4_generatedOut_
            if d_6_stoppedOnEos_:
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            if d_5_stoppedOnOpen_:
                insideConstrainedOut = True
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_8_openedGenerated_: _dafny.Seq
                        d_9_openedInside_: bool
                        d_10_openedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_8_openedGenerated_ = out4_
                        d_9_openedInside_ = out5_
                        d_10_openedCurrent_ = out6_
                        generated = d_8_openedGenerated_
                        insideConstrainedOut = d_9_openedInside_
                        currentConstrainedOut = d_10_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_11_closedGenerated_: _dafny.Seq
                        d_12_closedInside_: bool
                        d_13_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_11_closedGenerated_ = out7_
                        d_12_closedInside_ = out8_
                        d_13_closedCurrent_ = out9_
                        generated = d_11_closedGenerated_
                        insideConstrainedOut = d_12_closedInside_
                        currentConstrainedOut = d_13_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_14_constrainedPrompt_: _dafny.Seq
                        d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_15_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if not(d_2_seenFrom_):
                            d_16_sqlKeywordGroups_: _dafny.Seq
                            d_16_sqlKeywordGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))])])
                            d_17_groupsToBoost_: _dafny.Seq
                            d_17_groupsToBoost_ = (d_16_sqlKeywordGroups_) + (validTokenGroups)
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, d_17_groupsToBoost_, _dafny.BigRational('4e0'), eosToken)
                            d_15_next_ = out10_
                        elif True:
                            d_18_narrowThreshold_: int
                            d_18_narrowThreshold_ = 12
                            d_19_boostAmount_: _dafny.BigRational
                            d_19_boostAmount_ = _dafny.BigRational('4e0')
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, d_19_boostAmount_, d_18_narrowThreshold_, eosToken)
                            d_15_next_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_15_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_20_appendedGenerated_: _dafny.Seq
                            d_21_appendedInside_: bool
                            d_22_appendedCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                            d_20_appendedGenerated_ = out12_
                            d_21_appendedInside_ = out13_
                            d_22_appendedCurrent_ = out14_
                            generated = d_20_appendedGenerated_
                            insideConstrainedOut = d_21_appendedInside_
                            currentConstrainedOut = d_22_appendedCurrent_
                            if (not(d_2_seenFrom_)) and ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))):
                                d_2_seenFrom_ = True
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


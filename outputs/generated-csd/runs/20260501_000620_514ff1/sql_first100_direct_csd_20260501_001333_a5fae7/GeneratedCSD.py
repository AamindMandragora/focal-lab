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
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_3_flatGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens):
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
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_7_isComplete_: bool
                        d_7_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_isComplete_:
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))) in ((lm).Tokens):
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
                            elif True:
                                raise _dafny.Break("0")
                            raise _dafny.Break("0")
                        elif True:
                            d_11_dead_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_11_dead_ = out7_
                            if d_11_dead_:
                                d_12_repaired_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")))
                                d_12_repaired_ = out8_
                                d_13_trim_: int
                                d_13_trim_ = (len(currentConstrainedOut)) - (len(d_12_repaired_))
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (d_13_trim_):])
                                currentConstrainedOut = d_12_repaired_
                                insideConstrainedOut = True
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_14_tokBeforeComma_: _dafny.Seq
                                d_15_foundBeforeComma_: bool
                                out9_: _dafny.Seq
                                out10_: bool
                                out9_, out10_ = (d_0_helpers_).LastTokenBefore(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")))
                                d_14_tokBeforeComma_ = out9_
                                d_15_foundBeforeComma_ = out10_
                                d_16_usePenalty_: bool
                                d_16_usePenalty_ = False
                                if (0) < (len(currentConstrainedOut)):
                                    if (d_15_foundBeforeComma_) and (((currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]) == (d_14_tokBeforeComma_)):
                                        d_16_usePenalty_ = True
                                    elif ((1) < (len(currentConstrainedOut))) and (((currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]) == ((currentConstrainedOut)[(len(currentConstrainedOut)) - (2)])):
                                        d_16_usePenalty_ = True
                                d_17_stablePrefix_: _dafny.Seq
                                d_17_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                def lambda0_(forall_var_0_):
                                    d_18_t_: _dafny.Seq = forall_var_0_
                                    return not ((d_18_t_) in (d_3_flatGroups_)) or ((d_18_t_) in ((lm).Tokens))

                                if ((d_16_usePenalty_) and ((0) < (len(d_3_flatGroups_)))) and (_dafny.quantifier((d_3_flatGroups_).UniqueElements, True, lambda0_)):
                                    d_19_next_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = (d_0_helpers_).PenalizedConstrainedStep(lm, parser, (prompt) + (d_17_stablePrefix_), currentConstrainedOut, d_3_flatGroups_, _dafny.BigRational('5e0'), eosToken)
                                    d_19_next_ = out11_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_19_next_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_20_appendedGenerated_: _dafny.Seq
                                        d_21_appendedInside_: bool
                                        d_22_appendedCurrent_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out13_: bool
                                        out14_: _dafny.Seq
                                        out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                        d_20_appendedGenerated_ = out12_
                                        d_21_appendedInside_ = out13_
                                        d_22_appendedCurrent_ = out14_
                                        generated = d_20_appendedGenerated_
                                        insideConstrainedOut = d_21_appendedInside_
                                        currentConstrainedOut = d_22_appendedCurrent_
                                elif True:
                                    d_23_next_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out15_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_17_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                                    d_23_next_ = out15_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_23_next_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_24_appendedGenerated_: _dafny.Seq
                                        d_25_appendedInside_: bool
                                        d_26_appendedCurrent_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out17_: bool
                                        out18_: _dafny.Seq
                                        out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                        d_24_appendedGenerated_ = out16_
                                        d_25_appendedInside_ = out17_
                                        d_26_appendedCurrent_ = out18_
                                        generated = d_24_appendedGenerated_
                                        insideConstrainedOut = d_25_appendedInside_
                                        currentConstrainedOut = d_26_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


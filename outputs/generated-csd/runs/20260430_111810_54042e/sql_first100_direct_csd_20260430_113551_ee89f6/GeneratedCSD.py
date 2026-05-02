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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_commaTok_: _dafny.Seq
        d_2_commaTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))
        d_3_whereTok_: _dafny.Seq
        d_3_whereTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE"))
        d_4_fromTok_: _dafny.Seq
        d_4_fromTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_5_groupTok_: _dafny.Seq
        d_5_groupTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP"))
        d_6_orderTok_: _dafny.Seq
        d_6_orderTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER"))
        d_7_limitTok_: _dafny.Seq
        d_7_limitTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT"))
        d_8_joinTok_: _dafny.Seq
        d_8_joinTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN"))
        d_9_selectTok_: _dafny.Seq
        d_9_selectTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT"))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens):
                            d_10_openedGenerated_: _dafny.Seq
                            d_11_openedInside_: bool
                            d_12_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_openedGenerated_ = out0_
                            d_11_openedInside_ = out1_
                            d_12_openedCurrent_ = out2_
                            generated = d_10_openedGenerated_
                            insideConstrainedOut = d_11_openedInside_
                            currentConstrainedOut = d_12_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_13_isComplete_: bool
                        d_13_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_13_isComplete_:
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))) in ((lm).Tokens):
                                d_14_closedGenerated_: _dafny.Seq
                                d_15_closedInside_: bool
                                d_16_closedCurrent_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_14_closedGenerated_ = out3_
                                d_15_closedInside_ = out4_
                                d_16_closedCurrent_ = out5_
                                generated = d_14_closedGenerated_
                                insideConstrainedOut = d_15_closedInside_
                                currentConstrainedOut = d_16_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_17_shouldRepair_: bool
                            d_17_shouldRepair_ = False
                            d_18_remainingBudget_: int
                            d_18_remainingBudget_ = (maxSteps) - (d_1_steps_)
                            if (d_18_remainingBudget_) <= (1):
                                d_17_shouldRepair_ = True
                            elif True:
                                d_19_deadEndish_: bool
                                out6_: bool
                                out6_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                                d_19_deadEndish_ = out6_
                                if d_19_deadEndish_:
                                    d_17_shouldRepair_ = True
                                elif True:
                                    d_20_validCount_: int
                                    out7_: int
                                    out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                    d_20_validCount_ = out7_
                                    if ((len(currentConstrainedOut)) > (40)) and ((d_20_validCount_) <= (2)):
                                        d_17_shouldRepair_ = True
                            if d_17_shouldRepair_:
                                d_21_repaired_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_2_commaTok_)
                                d_21_repaired_ = out8_
                                d_22_candidate_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_3_whereTok_)
                                d_22_candidate_ = out9_
                                if (len(d_22_candidate_)) < (len(d_21_repaired_)):
                                    d_21_repaired_ = d_22_candidate_
                                out10_: _dafny.Seq
                                out10_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_4_fromTok_)
                                d_22_candidate_ = out10_
                                if (len(d_22_candidate_)) < (len(d_21_repaired_)):
                                    d_21_repaired_ = d_22_candidate_
                                out11_: _dafny.Seq
                                out11_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_5_groupTok_)
                                d_22_candidate_ = out11_
                                if (len(d_22_candidate_)) < (len(d_21_repaired_)):
                                    d_21_repaired_ = d_22_candidate_
                                out12_: _dafny.Seq
                                out12_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_6_orderTok_)
                                d_22_candidate_ = out12_
                                if (len(d_22_candidate_)) < (len(d_21_repaired_)):
                                    d_21_repaired_ = d_22_candidate_
                                out13_: _dafny.Seq
                                out13_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_7_limitTok_)
                                d_22_candidate_ = out13_
                                if (len(d_22_candidate_)) < (len(d_21_repaired_)):
                                    d_21_repaired_ = d_22_candidate_
                                out14_: _dafny.Seq
                                out14_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_8_joinTok_)
                                d_22_candidate_ = out14_
                                if (len(d_22_candidate_)) < (len(d_21_repaired_)):
                                    d_21_repaired_ = d_22_candidate_
                                out15_: _dafny.Seq
                                out15_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_9_selectTok_)
                                d_22_candidate_ = out15_
                                if (len(d_22_candidate_)) < (len(d_21_repaired_)):
                                    d_21_repaired_ = d_22_candidate_
                                if (len(d_21_repaired_)) < (len(currentConstrainedOut)):
                                    generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(currentConstrainedOut)) - (len(d_21_repaired_))):])
                                    currentConstrainedOut = d_21_repaired_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_23_next_: _dafny.Seq
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_23_next_ = out16_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_23_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_24_appendedGenerated_: _dafny.Seq
                                    d_25_appendedInside_: bool
                                    d_26_appendedCurrent_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                    d_24_appendedGenerated_ = out17_
                                    d_25_appendedInside_ = out18_
                                    d_26_appendedCurrent_ = out19_
                                    generated = d_24_appendedGenerated_
                                    insideConstrainedOut = d_25_appendedInside_
                                    currentConstrainedOut = d_26_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


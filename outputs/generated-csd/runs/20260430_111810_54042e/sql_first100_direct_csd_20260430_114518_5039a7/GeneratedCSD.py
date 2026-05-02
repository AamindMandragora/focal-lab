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
                        d_13_isCompleteNow_: bool
                        d_13_isCompleteNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_13_isCompleteNow_:
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
                            d_17_validCount_: int
                            out6_: int
                            out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_17_validCount_ = out6_
                            if ((d_17_validCount_) <= (1)) and ((len(currentConstrainedOut)) > (0)):
                                d_18_repaired_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_2_commaTok_)
                                d_18_repaired_ = out7_
                                d_19_candidate_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_3_whereTok_)
                                d_19_candidate_ = out8_
                                if (len(d_19_candidate_)) > (len(d_18_repaired_)):
                                    d_18_repaired_ = d_19_candidate_
                                out9_: _dafny.Seq
                                out9_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_4_fromTok_)
                                d_19_candidate_ = out9_
                                if (len(d_19_candidate_)) > (len(d_18_repaired_)):
                                    d_18_repaired_ = d_19_candidate_
                                out10_: _dafny.Seq
                                out10_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_5_groupTok_)
                                d_19_candidate_ = out10_
                                if (len(d_19_candidate_)) > (len(d_18_repaired_)):
                                    d_18_repaired_ = d_19_candidate_
                                out11_: _dafny.Seq
                                out11_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_6_orderTok_)
                                d_19_candidate_ = out11_
                                if (len(d_19_candidate_)) > (len(d_18_repaired_)):
                                    d_18_repaired_ = d_19_candidate_
                                out12_: _dafny.Seq
                                out12_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_7_limitTok_)
                                d_19_candidate_ = out12_
                                if (len(d_19_candidate_)) > (len(d_18_repaired_)):
                                    d_18_repaired_ = d_19_candidate_
                                out13_: _dafny.Seq
                                out13_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_8_joinTok_)
                                d_19_candidate_ = out13_
                                if (len(d_19_candidate_)) > (len(d_18_repaired_)):
                                    d_18_repaired_ = d_19_candidate_
                                out14_: _dafny.Seq
                                out14_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_9_selectTok_)
                                d_19_candidate_ = out14_
                                if (len(d_19_candidate_)) > (len(d_18_repaired_)):
                                    d_18_repaired_ = d_19_candidate_
                                if (len(d_18_repaired_)) < (len(currentConstrainedOut)):
                                    generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(currentConstrainedOut)) - (len(d_18_repaired_))):])
                                    currentConstrainedOut = d_18_repaired_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                (lm).GenerateLogits((prompt) + (generated))
                                (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'))
                                if (d_17_validCount_) <= (3):
                                    (d_0_helpers_).ScaleAllLogits(lm, _dafny.BigRational('8e-1'))
                                d_20_next_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).ConstrainedSample(lm, parser, currentConstrainedOut, eosToken)
                                d_20_next_ = out15_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_20_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_21_appendedGenerated_: _dafny.Seq
                                    d_22_appendedInside_: bool
                                    d_23_appendedCurrent_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                    d_21_appendedGenerated_ = out16_
                                    d_22_appendedInside_ = out17_
                                    d_23_appendedCurrent_ = out18_
                                    generated = d_21_appendedGenerated_
                                    insideConstrainedOut = d_22_appendedInside_
                                    currentConstrainedOut = d_23_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


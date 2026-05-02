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
        d_2_fromTok_: _dafny.Seq
        d_2_fromTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_3_whereTok_: _dafny.Seq
        d_3_whereTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE"))
        d_4_groupTok_: _dafny.Seq
        d_4_groupTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP"))
        d_5_orderTok_: _dafny.Seq
        d_5_orderTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER"))
        d_6_limitTok_: _dafny.Seq
        d_6_limitTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT"))
        d_7_joinTok_: _dafny.Seq
        d_7_joinTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN"))
        d_8_commaTok_: _dafny.Seq
        d_8_commaTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))
        d_9_selectTok_: _dafny.Seq
        d_9_selectTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT"))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_10_chunkBudget_: int
                        d_10_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_11_chunkedGenerated_: _dafny.Seq
                        d_12_stoppedOnOpenSpan_: bool
                        d_13_stoppedOnEos_: bool
                        d_14_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_10_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_11_chunkedGenerated_ = out0_
                        d_12_stoppedOnOpenSpan_ = out1_
                        d_13_stoppedOnEos_ = out2_
                        d_14_stepsUsed_ = out3_
                        generated = d_11_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_14_stepsUsed_)
                        if d_13_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_12_stoppedOnOpenSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_15_isComplete_: bool
                        d_15_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_15_isComplete_:
                            d_16_closedGenerated_: _dafny.Seq
                            d_17_closedInside_: bool
                            d_18_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_16_closedGenerated_ = out4_
                            d_17_closedInside_ = out5_
                            d_18_closedCurrent_ = out6_
                            generated = d_16_closedGenerated_
                            insideConstrainedOut = d_17_closedInside_
                            currentConstrainedOut = d_18_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_19_remainingBudget_: int
                            d_19_remainingBudget_ = (maxSteps) - (d_1_steps_)
                            d_20_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_20_validCount_ = out7_
                            d_21_shouldRepair_: bool
                            d_21_shouldRepair_ = False
                            if (d_19_remainingBudget_) <= (1):
                                d_21_shouldRepair_ = True
                            elif (len(currentConstrainedOut)) >= (d_19_remainingBudget_):
                                d_21_shouldRepair_ = True
                            elif True:
                                d_22_deadEndish_: bool
                                out8_: bool
                                out8_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                                d_22_deadEndish_ = out8_
                                if d_22_deadEndish_:
                                    d_21_shouldRepair_ = True
                                elif ((len(currentConstrainedOut)) > (80)) and ((d_20_validCount_) <= (2)):
                                    d_21_shouldRepair_ = True
                            if d_21_shouldRepair_:
                                d_23_repaired_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_8_commaTok_)
                                d_23_repaired_ = out9_
                                if (len(d_23_repaired_)) == (len(currentConstrainedOut)):
                                    d_24_repaired2_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_3_whereTok_)
                                    d_24_repaired2_ = out10_
                                    if (len(d_24_repaired2_)) < (len(d_23_repaired_)):
                                        d_23_repaired_ = d_24_repaired2_
                                    d_25_repaired3_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_2_fromTok_)
                                    d_25_repaired3_ = out11_
                                    if (len(d_25_repaired3_)) < (len(d_23_repaired_)):
                                        d_23_repaired_ = d_25_repaired3_
                                    d_26_repaired4_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_4_groupTok_)
                                    d_26_repaired4_ = out12_
                                    if (len(d_26_repaired4_)) < (len(d_23_repaired_)):
                                        d_23_repaired_ = d_26_repaired4_
                                    d_27_repaired5_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out13_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_5_orderTok_)
                                    d_27_repaired5_ = out13_
                                    if (len(d_27_repaired5_)) < (len(d_23_repaired_)):
                                        d_23_repaired_ = d_27_repaired5_
                                    d_28_repaired6_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out14_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_6_limitTok_)
                                    d_28_repaired6_ = out14_
                                    if (len(d_28_repaired6_)) < (len(d_23_repaired_)):
                                        d_23_repaired_ = d_28_repaired6_
                                    d_29_repaired7_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out15_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_7_joinTok_)
                                    d_29_repaired7_ = out15_
                                    if (len(d_29_repaired7_)) < (len(d_23_repaired_)):
                                        d_23_repaired_ = d_29_repaired7_
                                    d_30_repaired8_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out16_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_9_selectTok_)
                                    d_30_repaired8_ = out16_
                                    if (len(d_30_repaired8_)) < (len(d_23_repaired_)):
                                        d_23_repaired_ = d_30_repaired8_
                                if (len(d_23_repaired_)) < (len(currentConstrainedOut)):
                                    generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(currentConstrainedOut)) - (len(d_23_repaired_))):])
                                    currentConstrainedOut = d_23_repaired_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_31_stablePrefix_: _dafny.Seq
                                d_31_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_32_constrainedPrompt_: _dafny.Seq
                                d_32_constrainedPrompt_ = (prompt) + (d_31_stablePrefix_)
                                (lm).GenerateLogits((d_32_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(validTokenGroups)) > (0):
                                    (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'))
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_2_fromTok_, d_3_whereTok_, d_4_groupTok_, d_5_orderTok_, d_6_limitTok_, d_7_joinTok_, d_8_commaTok_]), _dafny.BigRational('15e-1'))
                                if (len(currentConstrainedOut)) > (0):
                                    (d_0_helpers_).PenalizeTokenLogits(lm, currentConstrainedOut, _dafny.BigRational('3e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_33_next_: _dafny.Seq
                                out17_: _dafny.Seq
                                out17_ = (lm).ChooseNextToken()
                                d_33_next_ = out17_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_33_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_34_appendedGenerated_: _dafny.Seq
                                    d_35_appendedInside_: bool
                                    d_36_appendedCurrent_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_33_next_)
                                    d_34_appendedGenerated_ = out18_
                                    d_35_appendedInside_ = out19_
                                    d_36_appendedCurrent_ = out20_
                                    generated = d_34_appendedGenerated_
                                    insideConstrainedOut = d_35_appendedInside_
                                    currentConstrainedOut = d_36_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


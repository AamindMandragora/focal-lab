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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, eosToken):
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
        d_2_minSqlTokens_: int
        d_2_minSqlTokens_ = 6
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        (lm).GenerateLogits((prompt) + (generated))
                        (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (lm).ChooseNextTokenUnconstrained()
                        d_3_next_ = out0_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_4_complete_: bool
                        d_4_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        d_5_narrowForClose_: bool
                        out1_: bool
                        out1_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_5_narrowForClose_ = out1_
                        if ((d_4_complete_) and (not(d_5_narrowForClose_))) and ((d_2_minSqlTokens_) <= (len(currentConstrainedOut))):
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out2_
                            d_7_closedInside_ = out3_
                            d_8_closedCurrent_ = out4_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_narrow_: bool
                            out5_: bool
                            out5_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_9_narrow_ = out5_
                            if ((d_9_narrow_) and ((0) < (len(currentConstrainedOut)))) and (not(d_4_complete_)):
                                d_10_stablePrefix_: _dafny.Seq
                                d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_11_rolled_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_11_rolled_ = out6_
                                d_12_rolledGenerated_: _dafny.Seq
                                d_13_rolledCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: _dafny.Seq
                                out7_, out8_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_10_stablePrefix_, generated, currentConstrainedOut)
                                d_12_rolledGenerated_ = out7_
                                d_13_rolledCurrent_ = out8_
                                generated = d_12_rolledGenerated_
                                currentConstrainedOut = d_13_rolledCurrent_
                                insideConstrainedOut = True
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_14_validCount_: int
                                out9_: int
                                out9_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_14_validCount_ = out9_
                                if (d_14_validCount_) <= (4):
                                    d_15_stablePrefix2_: _dafny.Seq
                                    d_15_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    (lm).GenerateLogits(((prompt) + (d_15_stablePrefix2_)) + (currentConstrainedOut))
                                    d_16_cands_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 4, eosToken)
                                    d_16_cands_ = out10_
                                    (d_0_helpers_).BoostTokenLogits(lm, d_16_cands_, _dafny.BigRational('8e0'))
                                    d_17_next2_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = (lm).ChooseNextToken()
                                    d_17_next2_ = out11_
                                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_17_next2_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_18_validNext2_: bool
                                        d_18_validNext2_ = (parser).IsValidPrefix((currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([d_17_next2_])))
                                        if (not(d_4_complete_)) and (d_18_validNext2_):
                                            d_19_appendedGenerated_: _dafny.Seq
                                            d_20_appendedInside_: bool
                                            d_21_appendedCurrent_: _dafny.Seq
                                            out12_: _dafny.Seq
                                            out13_: bool
                                            out14_: _dafny.Seq
                                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next2_)
                                            d_19_appendedGenerated_ = out12_
                                            d_20_appendedInside_ = out13_
                                            d_21_appendedCurrent_ = out14_
                                            generated = d_19_appendedGenerated_
                                            insideConstrainedOut = d_20_appendedInside_
                                            currentConstrainedOut = d_21_appendedCurrent_
                                elif True:
                                    d_22_stablePrefix3_: _dafny.Seq
                                    d_22_stablePrefix3_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_23_constrainedPrompt_: _dafny.Seq
                                    d_23_constrainedPrompt_ = (prompt) + (d_22_stablePrefix3_)
                                    d_24_next3_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_24_next3_ = out15_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_24_next3_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_25_validNext3_: bool
                                        d_25_validNext3_ = (parser).IsValidPrefix((currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([d_24_next3_])))
                                        if (not(d_4_complete_)) and (d_25_validNext3_):
                                            d_26_appendedGenerated2_: _dafny.Seq
                                            d_27_appendedInside2_: bool
                                            d_28_appendedCurrent2_: _dafny.Seq
                                            out16_: _dafny.Seq
                                            out17_: bool
                                            out18_: _dafny.Seq
                                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next3_)
                                            d_26_appendedGenerated2_ = out16_
                                            d_27_appendedInside2_ = out17_
                                            d_28_appendedCurrent2_ = out18_
                                            generated = d_26_appendedGenerated2_
                                            insideConstrainedOut = d_27_appendedInside2_
                                            currentConstrainedOut = d_28_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


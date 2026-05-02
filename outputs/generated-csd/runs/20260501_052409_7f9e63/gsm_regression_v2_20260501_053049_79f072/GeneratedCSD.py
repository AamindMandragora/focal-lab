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
        d_2_spanBudget_: int
        if (stepTokenBudget) == (0):
            d_2_spanBudget_ = 6
        elif True:
            d_2_spanBudget_ = stepTokenBudget
        d_3_seenOpen_: bool
        d_3_seenOpen_ = False
        d_4_i_: int
        d_4_i_ = 0
        while (d_4_i_) < (len(generated)):
            if ((generated)[d_4_i_]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                d_3_seenOpen_ = True
            d_4_i_ = (d_4_i_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_3_seenOpen_)) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)):
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                            d_5_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (lm).ChooseNextTokenUnconstrained()
                            d_5_next_ = out0_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_5_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                                if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_3_seenOpen_ = True
                        elif True:
                            d_6_next2_: _dafny.Seq
                            out1_: _dafny.Seq
                            out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_6_next2_ = out1_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_next2_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next2_]))
                                if (d_6_next2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_3_seenOpen_ = True
                    elif True:
                        d_7_complete_: bool
                        d_7_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_complete_:
                            d_8_closedGenerated_: _dafny.Seq
                            d_9_closedInside_: bool
                            d_10_closedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_8_closedGenerated_ = out2_
                            d_9_closedInside_ = out3_
                            d_10_closedCurrent_ = out4_
                            generated = d_8_closedGenerated_
                            insideConstrainedOut = d_9_closedInside_
                            currentConstrainedOut = d_10_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if (len(currentConstrainedOut)) >= (d_2_spanBudget_):
                                d_11_repaired_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_11_repaired_ = out5_
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(currentConstrainedOut)) - (len(d_11_repaired_))):])
                                currentConstrainedOut = d_11_repaired_
                                d_12_repairedComplete_: bool
                                d_12_repairedComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_12_repairedComplete_:
                                    d_13_closedGenerated2_: _dafny.Seq
                                    d_14_closedInside2_: bool
                                    d_15_closedCurrent2_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_13_closedGenerated2_ = out6_
                                    d_14_closedInside2_ = out7_
                                    d_15_closedCurrent2_ = out8_
                                    generated = d_13_closedGenerated2_
                                    insideConstrainedOut = d_14_closedInside2_
                                    currentConstrainedOut = d_15_closedCurrent2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_16_stablePrefix0_: _dafny.Seq
                                    d_16_stablePrefix0_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_17_next3_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out9_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_16_stablePrefix0_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('8e0'), 10, eosToken)
                                    d_17_next3_ = out9_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_17_next3_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_18_appendedGenerated0_: _dafny.Seq
                                        d_19_appendedInside0_: bool
                                        d_20_appendedCurrent0_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next3_)
                                        d_18_appendedGenerated0_ = out10_
                                        d_19_appendedInside0_ = out11_
                                        d_20_appendedCurrent0_ = out12_
                                        generated = d_18_appendedGenerated0_
                                        insideConstrainedOut = d_19_appendedInside0_
                                        currentConstrainedOut = d_20_appendedCurrent0_
                            elif True:
                                d_21_stablePrefix_: _dafny.Seq
                                d_21_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_22_validCount_: int
                                out13_: int
                                out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_22_validCount_ = out13_
                                if (d_22_validCount_) <= (10):
                                    d_23_next4_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_21_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), 10, eosToken)
                                    d_23_next4_ = out14_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_23_next4_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_24_appendedGenerated1_: _dafny.Seq
                                        d_25_appendedInside1_: bool
                                        d_26_appendedCurrent1_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out17_: _dafny.Seq
                                        out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next4_)
                                        d_24_appendedGenerated1_ = out15_
                                        d_25_appendedInside1_ = out16_
                                        d_26_appendedCurrent1_ = out17_
                                        generated = d_24_appendedGenerated1_
                                        insideConstrainedOut = d_25_appendedInside1_
                                        currentConstrainedOut = d_26_appendedCurrent1_
                                elif True:
                                    d_27_next5_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out18_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_21_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), 10, eosToken)
                                    d_27_next5_ = out18_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_27_next5_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_28_appendedGenerated2_: _dafny.Seq
                                        d_29_appendedInside2_: bool
                                        d_30_appendedCurrent2_: _dafny.Seq
                                        out19_: _dafny.Seq
                                        out20_: bool
                                        out21_: _dafny.Seq
                                        out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next5_)
                                        d_28_appendedGenerated2_ = out19_
                                        d_29_appendedInside2_ = out20_
                                        d_30_appendedCurrent2_ = out21_
                                        generated = d_28_appendedGenerated2_
                                        insideConstrainedOut = d_29_appendedInside2_
                                        currentConstrainedOut = d_30_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


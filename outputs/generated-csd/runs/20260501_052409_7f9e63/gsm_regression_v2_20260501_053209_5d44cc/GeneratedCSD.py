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
        d_2_seenOpen_: bool
        d_2_seenOpen_ = insideConstrained
        d_3_j_: int
        d_3_j_ = 0
        while (d_3_j_) < (len(generated)):
            if ((generated)[d_3_j_]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                d_2_seenOpen_ = True
            d_3_j_ = (d_3_j_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((not(d_2_seenOpen_)) and ((len(generated)) >= (8))) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)):
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'))
                            d_4_next0_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (lm).ChooseNextTokenUnconstrained()
                            d_4_next0_ = out0_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_4_next0_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if (d_4_next0_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_5_openedGenerated_: _dafny.Seq
                                    d_6_openedInside_: bool
                                    d_7_openedCurrent_: _dafny.Seq
                                    out1_: _dafny.Seq
                                    out2_: bool
                                    out3_: _dafny.Seq
                                    out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_5_openedGenerated_ = out1_
                                    d_6_openedInside_ = out2_
                                    d_7_openedCurrent_ = out3_
                                    generated = d_5_openedGenerated_
                                    insideConstrainedOut = d_6_openedInside_
                                    currentConstrainedOut = d_7_openedCurrent_
                                    d_2_seenOpen_ = True
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next0_]))
                        elif True:
                            d_8_next1_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next1_ = out4_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next1_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if ((d_8_next1_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)):
                                    d_9_openedGenerated2_: _dafny.Seq
                                    d_10_openedInside2_: bool
                                    d_11_openedCurrent2_: _dafny.Seq
                                    out5_: _dafny.Seq
                                    out6_: bool
                                    out7_: _dafny.Seq
                                    out5_, out6_, out7_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_9_openedGenerated2_ = out5_
                                    d_10_openedInside2_ = out6_
                                    d_11_openedCurrent2_ = out7_
                                    generated = d_9_openedGenerated2_
                                    insideConstrainedOut = d_10_openedInside2_
                                    currentConstrainedOut = d_11_openedCurrent2_
                                    d_2_seenOpen_ = True
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next1_]))
                    elif True:
                        d_12_complete_: bool
                        d_12_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_12_complete_:
                            d_13_closedGenerated_: _dafny.Seq
                            d_14_closedInside_: bool
                            d_15_closedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_13_closedGenerated_ = out8_
                            d_14_closedInside_ = out9_
                            d_15_closedCurrent_ = out10_
                            generated = d_13_closedGenerated_
                            insideConstrainedOut = d_14_closedInside_
                            currentConstrainedOut = d_15_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if ((stepTokenBudget) > (0)) and ((len(currentConstrainedOut)) >= (stepTokenBudget)):
                                d_16_repaired_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_16_repaired_ = out11_
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(currentConstrainedOut)) - (len(d_16_repaired_))):])
                                currentConstrainedOut = d_16_repaired_
                                d_17_repairedComplete_: bool
                                d_17_repairedComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_17_repairedComplete_:
                                    d_18_closedGenerated2_: _dafny.Seq
                                    d_19_closedInside2_: bool
                                    d_20_closedCurrent2_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_18_closedGenerated2_ = out12_
                                    d_19_closedInside2_ = out13_
                                    d_20_closedCurrent2_ = out14_
                                    generated = d_18_closedGenerated2_
                                    insideConstrainedOut = d_19_closedInside2_
                                    currentConstrainedOut = d_20_closedCurrent2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    if (len(d_16_repaired_)) == (len(currentConstrainedOut)):
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        raise _dafny.Break("0")
                            elif True:
                                d_21_stablePrefix_: _dafny.Seq
                                d_21_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_22_next2_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_21_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_22_next2_ = out15_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_22_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_23_appendedGenerated_: _dafny.Seq
                                    d_24_appendedInside_: bool
                                    d_25_appendedCurrent_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next2_)
                                    d_23_appendedGenerated_ = out16_
                                    d_24_appendedInside_ = out17_
                                    d_25_appendedCurrent_ = out18_
                                    generated = d_23_appendedGenerated_
                                    insideConstrainedOut = d_24_appendedInside_
                                    currentConstrainedOut = d_25_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


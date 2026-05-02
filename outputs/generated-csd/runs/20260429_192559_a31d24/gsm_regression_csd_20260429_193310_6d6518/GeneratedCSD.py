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
        d_2_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_flatGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_1_steps_) + (1)) < (maxSteps):
                            d_3_lastBefore_: _dafny.Seq
                            d_4_foundBefore_: bool
                            out1_: _dafny.Seq
                            out2_: bool
                            out1_, out2_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                            d_3_lastBefore_ = out1_
                            d_4_foundBefore_ = out2_
                            if (not(d_4_foundBefore_)) or ((d_3_lastBefore_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))):
                                d_5_openedGenerated_: _dafny.Seq
                                d_6_openedInside_: bool
                                d_7_openedCurrent_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_5_openedGenerated_ = out3_
                                d_6_openedInside_ = out4_
                                d_7_openedCurrent_ = out5_
                                generated = d_5_openedGenerated_
                                insideConstrainedOut = d_6_openedInside_
                                currentConstrainedOut = d_7_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                (lm).GenerateLogits((prompt) + (generated))
                                if (len(d_2_flatGroups_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_2_flatGroups_, _dafny.BigRational('3e0'))
                                d_8_next_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = (lm).ChooseNextTokenUnconstrained()
                                d_8_next_ = out6_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_8_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            if (len(d_2_flatGroups_)) > (0):
                                (d_0_helpers_).BoostTokenLogits(lm, d_2_flatGroups_, _dafny.BigRational('3e0'))
                            d_9_next2_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (lm).ChooseNextTokenUnconstrained()
                            d_9_next2_ = out7_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next2_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next2_]))
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_10_closedGenerated_: _dafny.Seq
                            d_11_closedInside_: bool
                            d_12_closedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_closedGenerated_ = out8_
                            d_11_closedInside_ = out9_
                            d_12_closedCurrent_ = out10_
                            generated = d_10_closedGenerated_
                            insideConstrainedOut = d_11_closedInside_
                            currentConstrainedOut = d_12_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_13_dead_: bool
                            out11_: bool
                            out11_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_13_dead_ = out11_
                            if d_13_dead_:
                                d_14_repaired_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_14_repaired_ = out12_
                                d_15_stablePrefix_: _dafny.Seq
                                d_15_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_16_repairedGenerated_: _dafny.Seq
                                d_17_repairedCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: _dafny.Seq
                                out13_, out14_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_15_stablePrefix_, generated, currentConstrainedOut)
                                d_16_repairedGenerated_ = out13_
                                d_17_repairedCurrent_ = out14_
                                generated = d_16_repairedGenerated_
                                currentConstrainedOut = d_17_repairedCurrent_
                                insideConstrainedOut = True
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_18_stablePrefix2_: _dafny.Seq
                                d_18_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_19_constrainedPrompt_: _dafny.Seq
                                d_19_constrainedPrompt_ = (prompt) + (d_18_stablePrefix2_)
                                d_20_next3_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_20_next3_ = out15_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_20_next3_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_21_appendedGenerated_: _dafny.Seq
                                    d_22_appendedInside_: bool
                                    d_23_appendedCurrent_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next3_)
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


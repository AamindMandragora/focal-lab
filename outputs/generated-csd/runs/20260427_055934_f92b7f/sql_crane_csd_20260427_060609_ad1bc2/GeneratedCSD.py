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
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 6
        d_3_deadEndThreshold_: int
        d_3_deadEndThreshold_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_openedGenerated_: _dafny.Seq
                        d_5_openedInside_: bool
                        d_6_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_4_openedGenerated_ = out0_
                        d_5_openedInside_ = out1_
                        d_6_openedCurrent_ = out2_
                        generated = d_4_openedGenerated_
                        insideConstrainedOut = d_5_openedInside_
                        currentConstrainedOut = d_6_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_7_completeNow_: bool
                        d_7_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_completeNow_:
                            d_8_closedGenerated_: _dafny.Seq
                            d_9_closedInside_: bool
                            d_10_closedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_8_closedGenerated_ = out3_
                            d_9_closedInside_ = out4_
                            d_10_closedCurrent_ = out5_
                            generated = d_8_closedGenerated_
                            insideConstrainedOut = d_9_closedInside_
                            currentConstrainedOut = d_10_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_11_narrow_: bool
                            out6_: bool
                            out6_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, d_3_deadEndThreshold_)
                            d_11_narrow_ = out6_
                            if d_11_narrow_:
                                d_12_stablePrefix_: _dafny.Seq
                                d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_13_repairedGenerated_: _dafny.Seq
                                d_14_repairedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: _dafny.Seq
                                out7_, out8_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_12_stablePrefix_, generated, currentConstrainedOut)
                                d_13_repairedGenerated_ = out7_
                                d_14_repairedCurrent_ = out8_
                                generated = d_13_repairedGenerated_
                                currentConstrainedOut = d_14_repairedCurrent_
                            elif True:
                                d_15_stablePrefix2_: _dafny.Seq
                                d_15_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_16_constrainedPrompt_: _dafny.Seq
                                d_16_constrainedPrompt_ = (prompt) + (d_15_stablePrefix2_)
                                d_17_validCount_: int
                                out9_: int
                                out9_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_17_validCount_ = out9_
                                if (d_17_validCount_) <= (d_2_narrowThreshold_):
                                    (lm).GenerateLogits((d_16_constrainedPrompt_) + (currentConstrainedOut))
                                    d_18_candidates_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                                    d_18_candidates_ = out10_
                                    (d_0_helpers_).BoostTokenLogits(lm, d_18_candidates_, _dafny.BigRational('8e0'))
                                d_19_next_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, eosToken)
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
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


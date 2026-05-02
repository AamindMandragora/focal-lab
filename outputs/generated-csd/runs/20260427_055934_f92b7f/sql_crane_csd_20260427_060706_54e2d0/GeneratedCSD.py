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
        d_2_minCloseLen_: int
        d_2_minCloseLen_ = 12
        d_3_deadEndThreshold_: int
        d_3_deadEndThreshold_ = 0
        d_4_tinyBranchThreshold_: int
        d_4_tinyBranchThreshold_ = 2
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_openedGenerated_: _dafny.Seq
                        d_6_openedInside_: bool
                        d_7_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_5_openedGenerated_ = out0_
                        d_6_openedInside_ = out1_
                        d_7_openedCurrent_ = out2_
                        generated = d_5_openedGenerated_
                        insideConstrainedOut = d_6_openedInside_
                        currentConstrainedOut = d_7_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_8_completeNow_: bool
                        d_8_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if (d_8_completeNow_) and ((d_2_minCloseLen_) <= (len(currentConstrainedOut))):
                            d_9_closedGenerated_: _dafny.Seq
                            d_10_closedInside_: bool
                            d_11_closedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated_ = out3_
                            d_10_closedInside_ = out4_
                            d_11_closedCurrent_ = out5_
                            generated = d_9_closedGenerated_
                            insideConstrainedOut = d_10_closedInside_
                            currentConstrainedOut = d_11_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_12_deadEnd_: bool
                            out6_: bool
                            out6_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, d_3_deadEndThreshold_)
                            d_12_deadEnd_ = out6_
                            if d_12_deadEnd_:
                                d_13_stablePrefix_: _dafny.Seq
                                d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_14_repairedGenerated_: _dafny.Seq
                                d_15_repairedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: _dafny.Seq
                                out7_, out8_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_13_stablePrefix_, generated, currentConstrainedOut)
                                d_14_repairedGenerated_ = out7_
                                d_15_repairedCurrent_ = out8_
                                generated = d_14_repairedGenerated_
                                currentConstrainedOut = d_15_repairedCurrent_
                            elif True:
                                d_16_notCompleteNow_: bool
                                d_16_notCompleteNow_ = not(d_8_completeNow_)
                                d_17_stablePrefix2_: _dafny.Seq
                                d_17_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_18_constrainedPrompt_: _dafny.Seq
                                d_18_constrainedPrompt_ = (prompt) + (d_17_stablePrefix2_)
                                (lm).GenerateLogits((d_18_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(currentConstrainedOut)) < (d_2_minCloseLen_):
                                    (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                                    d_19_validCount_: int
                                    out9_: int
                                    out9_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                    d_19_validCount_ = out9_
                                    if (d_19_validCount_) <= (d_4_tinyBranchThreshold_):
                                        d_20_candidates_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out10_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, 3, eosToken)
                                        d_20_candidates_ = out10_
                                        (d_0_helpers_).PenalizeTokenLogits(lm, d_20_candidates_, _dafny.BigRational('3e0'))
                                d_21_next_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_21_next_ = out11_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_21_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    if d_16_notCompleteNow_:
                                        d_22_appendedGenerated_: _dafny.Seq
                                        d_23_appendedInside_: bool
                                        d_24_appendedCurrent_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out13_: bool
                                        out14_: _dafny.Seq
                                        out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                        d_22_appendedGenerated_ = out12_
                                        d_23_appendedInside_ = out13_
                                        d_24_appendedCurrent_ = out14_
                                        generated = d_22_appendedGenerated_
                                        insideConstrainedOut = d_23_appendedInside_
                                        currentConstrainedOut = d_24_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


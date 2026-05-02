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
        d_2_deadEndThreshold_: int
        d_2_deadEndThreshold_ = 1
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_4_completeNow_: bool
                        d_4_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_4_completeNow_:
                            d_5_closedGenerated_: _dafny.Seq
                            d_6_closedInside_: bool
                            d_7_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_5_closedGenerated_ = out1_
                            d_6_closedInside_ = out2_
                            d_7_closedCurrent_ = out3_
                            generated = d_5_closedGenerated_
                            insideConstrainedOut = d_6_closedInside_
                            currentConstrainedOut = d_7_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_narrow_: bool
                            out4_: bool
                            out4_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, d_2_deadEndThreshold_)
                            d_8_narrow_ = out4_
                            if d_8_narrow_:
                                d_9_repaired_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_9_repaired_ = out5_
                                d_10_stablePrefix_: _dafny.Seq
                                d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_11_beforeSep_: _dafny.Seq
                                d_12_foundSep_: bool
                                out6_: _dafny.Seq
                                out7_: bool
                                out6_, out7_ = (d_0_helpers_).LastTokenBefore(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")))
                                d_11_beforeSep_ = out6_
                                d_12_foundSep_ = out7_
                                if d_12_foundSep_:
                                    d_13_groupIdx_: int
                                    out8_: int
                                    out8_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_11_beforeSep_)
                                    d_13_groupIdx_ = out8_
                                    if (0) <= (d_13_groupIdx_):
                                        d_14_boundaryRepair_: _dafny.Seq
                                        out9_: _dafny.Seq
                                        out9_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_11_beforeSep_)
                                        d_14_boundaryRepair_ = out9_
                                        d_9_repaired_ = d_14_boundaryRepair_
                                generated = (d_10_stablePrefix_) + (d_9_repaired_)
                                currentConstrainedOut = d_9_repaired_
                            elif True:
                                d_15_stablePrefix_: _dafny.Seq
                                d_15_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_16_constrainedPrompt_: _dafny.Seq
                                d_16_constrainedPrompt_ = (prompt) + (d_15_stablePrefix_)
                                d_17_remaining_: int
                                d_17_remaining_ = (maxSteps) - (d_1_steps_)
                                d_18_localBudget_: int
                                d_18_localBudget_ = stepTokenBudget
                                if (d_17_remaining_) < (d_18_localBudget_):
                                    d_18_localBudget_ = d_17_remaining_
                                if (d_18_localBudget_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_19_currentOut_: _dafny.Seq
                                    d_20_hitEos_: bool
                                    d_21_stepsUsed_: int
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: int
                                    out10_, out11_, out12_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, d_18_localBudget_, eosToken)
                                    d_19_currentOut_ = out10_
                                    d_20_hitEos_ = out11_
                                    d_21_stepsUsed_ = out12_
                                    if (d_21_stepsUsed_) > (0):
                                        generated = (d_15_stablePrefix_) + (d_19_currentOut_)
                                        currentConstrainedOut = d_19_currentOut_
                                        d_1_steps_ = (d_1_steps_) + (d_21_stepsUsed_)
                                        if d_20_hitEos_:
                                            raise _dafny.Break("0")
                                    elif True:
                                        d_22_next_: _dafny.Seq
                                        out13_: _dafny.Seq
                                        out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, eosToken)
                                        d_22_next_ = out13_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        if (d_22_next_) == (eosToken):
                                            raise _dafny.Break("0")
                                        elif True:
                                            d_23_appendedGenerated_: _dafny.Seq
                                            d_24_appendedInside_: bool
                                            d_25_appendedCurrent_: _dafny.Seq
                                            out14_: _dafny.Seq
                                            out15_: bool
                                            out16_: _dafny.Seq
                                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                            d_23_appendedGenerated_ = out14_
                                            d_24_appendedInside_ = out15_
                                            d_25_appendedCurrent_ = out16_
                                            generated = d_23_appendedGenerated_
                                            insideConstrainedOut = d_24_appendedInside_
                                            currentConstrainedOut = d_25_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


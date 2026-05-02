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
                        d_3_remainingOutside_: int
                        d_3_remainingOutside_ = (maxSteps) - (d_1_steps_)
                        d_4_chunkedGenerated_: _dafny.Seq
                        d_5_stoppedOnOpenSpan_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_stepsUsedOutside_: int
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: bool
                        out4_: int
                        out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_remainingOutside_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_chunkedGenerated_ = out1_
                        d_5_stoppedOnOpenSpan_ = out2_
                        d_6_stoppedOnEos_ = out3_
                        d_7_stepsUsedOutside_ = out4_
                        generated = d_4_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsedOutside_)
                        if d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_5_stoppedOnOpenSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_8_closedGenerated_: _dafny.Seq
                            d_9_closedInside_: bool
                            d_10_closedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_8_closedGenerated_ = out5_
                            d_9_closedInside_ = out6_
                            d_10_closedCurrent_ = out7_
                            generated = d_8_closedGenerated_
                            insideConstrainedOut = d_9_closedInside_
                            currentConstrainedOut = d_10_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_remainingInside_: int
                            d_11_remainingInside_ = (maxSteps) - (d_1_steps_)
                            d_12_narrow_: bool
                            out8_: bool
                            out8_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_12_narrow_ = out8_
                            if d_12_narrow_:
                                d_13_repaired_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_13_repaired_ = out9_
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(currentConstrainedOut)) - (len(d_13_repaired_))):])
                                currentConstrainedOut = d_13_repaired_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_14_symbolBudget_: int
                                d_14_symbolBudget_ = stepTokenBudget
                                if (d_11_remainingInside_) < (d_14_symbolBudget_):
                                    d_14_symbolBudget_ = d_11_remainingInside_
                                if (d_14_symbolBudget_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_15_stablePrefix_: _dafny.Seq
                                    d_15_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_16_constrainedPrompt_: _dafny.Seq
                                    d_16_constrainedPrompt_ = (prompt) + (d_15_stablePrefix_)
                                    d_17_currentOut_: _dafny.Seq
                                    d_18_hitEos_: bool
                                    d_19_stepsUsedInside_: int
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: int
                                    out10_, out11_, out12_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, d_14_symbolBudget_, eosToken)
                                    d_17_currentOut_ = out10_
                                    d_18_hitEos_ = out11_
                                    d_19_stepsUsedInside_ = out12_
                                    if (d_19_stepsUsedInside_) == (0):
                                        d_20_repaired2_: _dafny.Seq
                                        out13_: _dafny.Seq
                                        out13_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                        d_20_repaired2_ = out13_
                                        generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(currentConstrainedOut)) - (len(d_20_repaired2_))):])
                                        currentConstrainedOut = d_20_repaired2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        generated = (d_15_stablePrefix_) + (d_17_currentOut_)
                                        currentConstrainedOut = d_17_currentOut_
                                        d_1_steps_ = (d_1_steps_) + (d_19_stepsUsedInside_)
                                        if d_18_hitEos_:
                                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


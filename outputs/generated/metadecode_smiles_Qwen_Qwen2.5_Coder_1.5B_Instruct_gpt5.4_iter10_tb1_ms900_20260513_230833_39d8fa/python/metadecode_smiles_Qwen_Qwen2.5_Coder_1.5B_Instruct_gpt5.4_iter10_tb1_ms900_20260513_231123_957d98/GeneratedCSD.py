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
        d_2_rollbackLimit_: int
        d_2_rollbackLimit_ = 64
        d_3_pendingForcedOpen_: bool
        d_3_pendingForcedOpen_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_3_pendingForcedOpen_:
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
                            d_3_pendingForcedOpen_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_remainingOutside_: int
                            d_7_remainingOutside_ = (maxSteps) - (d_1_steps_)
                            d_8_outsideBudget_: int
                            if (d_7_remainingOutside_) <= (2):
                                d_8_outsideBudget_ = d_7_remainingOutside_
                            elif True:
                                d_8_outsideBudget_ = 2
                            d_9_chunkedGenerated_: _dafny.Seq
                            d_10_stoppedOnOpenSpan_: bool
                            d_11_stoppedOnEos_: bool
                            d_12_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_outsideBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_9_chunkedGenerated_ = out3_
                            d_10_stoppedOnOpenSpan_ = out4_
                            d_11_stoppedOnEos_ = out5_
                            d_12_stepsUsed_ = out6_
                            generated = d_9_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                            if d_11_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_10_stoppedOnOpenSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_3_pendingForcedOpen_ = False
                            elif True:
                                d_3_pendingForcedOpen_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_closedGenerated_: _dafny.Seq
                        d_14_closedInside_: bool
                        d_15_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_closedGenerated_ = out7_
                        d_14_closedInside_ = out8_
                        d_15_closedCurrent_ = out9_
                        generated = d_13_closedGenerated_
                        insideConstrainedOut = d_14_closedInside_
                        currentConstrainedOut = d_15_closedCurrent_
                        d_3_pendingForcedOpen_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_2_rollbackLimit_):
                        d_16_rolledGenerated_: _dafny.Seq
                        d_17_rolledCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_16_rolledGenerated_ = out10_
                        d_17_rolledCurrent_ = out11_
                        generated = d_16_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_17_rolledCurrent_
                        d_3_pendingForcedOpen_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_18_stablePrefix_: _dafny.Seq
                        d_18_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (d_18_stablePrefix_)
                        d_20_remainingInside_: int
                        d_20_remainingInside_ = (maxSteps) - (d_1_steps_)
                        d_21_symbolBudget_: int
                        if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_20_remainingInside_)):
                            d_21_symbolBudget_ = d_20_remainingInside_
                        elif True:
                            d_21_symbolBudget_ = stepTokenBudget
                        d_22_symbolGenerated_: _dafny.Seq
                        d_23_symbolCurrent_: _dafny.Seq
                        d_24_hitEos_: bool
                        d_25_stepsUsed_: int
                        out12_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: int
                        out12_, out13_, out14_, out15_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_19_constrainedPrompt_, generated, currentConstrainedOut, d_21_symbolBudget_, eosToken)
                        d_22_symbolGenerated_ = out12_
                        d_23_symbolCurrent_ = out13_
                        d_24_hitEos_ = out14_
                        d_25_stepsUsed_ = out15_
                        generated = d_22_symbolGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_23_symbolCurrent_
                        d_3_pendingForcedOpen_ = False
                        d_1_steps_ = (d_1_steps_) + (d_25_stepsUsed_)
                        if d_24_hitEos_:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


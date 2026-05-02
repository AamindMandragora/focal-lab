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
        d_2_shouldStop_: bool
        d_2_shouldStop_ = False
        while ((d_1_steps_) < (maxSteps)) and (not(d_2_shouldStop_)):
            if not(insideConstrainedOut):
                d_3_rem0_: int
                d_3_rem0_ = (maxSteps) - (d_1_steps_)
                if (d_3_rem0_) == (0):
                    d_2_shouldStop_ = True
                elif True:
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
                d_7_rem1_: int
                d_7_rem1_ = (maxSteps) - (d_1_steps_)
                if (d_7_rem1_) == (0):
                    d_2_shouldStop_ = True
                elif True:
                    d_8_completeNow_: bool
                    d_8_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if d_8_completeNow_:
                        d_9_validCount_: int
                        out3_: int
                        out3_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_9_validCount_ = out3_
                        if ((d_9_validCount_) <= (1)) or ((d_7_rem1_) == (1)):
                            d_10_closedGenerated_: _dafny.Seq
                            d_11_closedInside_: bool
                            d_12_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_closedGenerated_ = out4_
                            d_11_closedInside_ = out5_
                            d_12_closedCurrent_ = out6_
                            generated = d_10_closedGenerated_
                            insideConstrainedOut = d_11_closedInside_
                            currentConstrainedOut = d_12_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_shouldStop_ = True
                        elif True:
                            d_13_stablePrefix1_: _dafny.Seq
                            d_13_stablePrefix1_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_14_constrainedPrompt1_: _dafny.Seq
                            d_14_constrainedPrompt1_ = (prompt) + (d_13_stablePrefix1_)
                            d_15_currentOut1_: _dafny.Seq
                            d_16_hitEos1_: bool
                            d_17_stepsUsed1_: int
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: int
                            out7_, out8_, out9_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_14_constrainedPrompt1_, currentConstrainedOut, d_7_rem1_, eosToken)
                            d_15_currentOut1_ = out7_
                            d_16_hitEos1_ = out8_
                            d_17_stepsUsed1_ = out9_
                            generated = (d_13_stablePrefix1_) + (d_15_currentOut1_)
                            insideConstrainedOut = True
                            currentConstrainedOut = d_15_currentOut1_
                            d_1_steps_ = (d_1_steps_) + (d_17_stepsUsed1_)
                            if d_16_hitEos1_:
                                d_2_shouldStop_ = True
                    elif True:
                        d_18_stablePrefix2_: _dafny.Seq
                        d_18_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_19_constrainedPrompt2_: _dafny.Seq
                        d_19_constrainedPrompt2_ = (prompt) + (d_18_stablePrefix2_)
                        d_20_currentOut2_: _dafny.Seq
                        d_21_hitEos2_: bool
                        d_22_stepsUsed2_: int
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: int
                        out10_, out11_, out12_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_19_constrainedPrompt2_, currentConstrainedOut, d_7_rem1_, eosToken)
                        d_20_currentOut2_ = out10_
                        d_21_hitEos2_ = out11_
                        d_22_stepsUsed2_ = out12_
                        generated = (d_18_stablePrefix2_) + (d_20_currentOut2_)
                        insideConstrainedOut = True
                        currentConstrainedOut = d_20_currentOut2_
                        d_1_steps_ = (d_1_steps_) + (d_22_stepsUsed2_)
                        if d_21_hitEos2_:
                            d_2_shouldStop_ = True
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


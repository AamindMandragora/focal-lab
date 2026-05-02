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
        if True:
            generated = generatedPrefix
            insideConstrainedOut = insideConstrained
            currentConstrainedOut = currentConstrained
            cost = 0
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_narrowThreshold_: int
            d_2_narrowThreshold_ = 8
            d_3_done_: bool
            d_3_done_ = False
            while ((d_1_steps_) < (maxSteps)) and (not(d_3_done_)):
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
                    d_7_isComplete_: bool
                    d_7_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if d_7_isComplete_:
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
                        d_3_done_ = True
                    elif True:
                        d_11_stablePrefix_: _dafny.Seq
                        d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_12_constrainedPrompt_: _dafny.Seq
                        d_12_constrainedPrompt_ = (prompt) + (d_11_stablePrefix_)
                        d_13_validCount_: int
                        out6_: int
                        out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_13_validCount_ = out6_
                        if (d_13_validCount_) <= (d_2_narrowThreshold_):
                            d_14_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_14_next_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_14_next_) == (eosToken):
                                d_3_done_ = True
                            elif True:
                                d_15_appendedGenerated_: _dafny.Seq
                                d_16_appendedInside_: bool
                                d_17_appendedCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                d_15_appendedGenerated_ = out8_
                                d_16_appendedInside_ = out9_
                                d_17_appendedCurrent_ = out10_
                                generated = d_15_appendedGenerated_
                                insideConstrainedOut = d_16_appendedInside_
                                currentConstrainedOut = d_17_appendedCurrent_
                        elif True:
                            d_18_symbolBudget_: int
                            d_18_symbolBudget_ = (maxSteps) - (d_1_steps_)
                            d_19_currentOut_: _dafny.Seq
                            d_20_hitEos_: bool
                            d_21_stepsUsed_: int
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: int
                            out11_, out12_, out13_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, d_18_symbolBudget_, eosToken)
                            d_19_currentOut_ = out11_
                            d_20_hitEos_ = out12_
                            d_21_stepsUsed_ = out13_
                            generated = (d_11_stablePrefix_) + (d_19_currentOut_)
                            currentConstrainedOut = d_19_currentOut_
                            d_1_steps_ = (d_1_steps_) + (d_21_stepsUsed_)
                            if d_20_hitEos_:
                                d_3_done_ = True
            cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


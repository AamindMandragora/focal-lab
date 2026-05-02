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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokens, eosToken):
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
        d_1_narrowThreshold_: int
        d_1_narrowThreshold_ = 10
        d_2_deadEndMinCount_: int
        d_2_deadEndMinCount_ = 2
        d_3_steps_: int
        d_3_steps_ = 0
        d_4_done_: bool
        d_4_done_ = False
        while ((d_3_steps_) < (maxSteps)) and (not(d_4_done_)):
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
                d_3_steps_ = (d_3_steps_) + (1)
            elif True:
                d_8_complete_: bool
                d_8_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_8_complete_:
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
                    d_3_steps_ = (d_3_steps_) + (1)
                    d_4_done_ = True
                elif True:
                    d_12_stablePrefix_: _dafny.Seq
                    d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_13_constrainedPrompt_: _dafny.Seq
                    d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                    d_14_validCount_: int
                    out6_: int
                    out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                    d_14_validCount_ = out6_
                    d_15_narrow_: bool
                    out7_: bool
                    out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, d_2_deadEndMinCount_)
                    d_15_narrow_ = out7_
                    if (d_15_narrow_) or ((d_14_validCount_) <= (d_1_narrowThreshold_)):
                        d_16_next_: _dafny.Seq
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_16_next_ = out8_
                        d_3_steps_ = (d_3_steps_) + (1)
                        if (d_16_next_) == (eosToken):
                            d_4_done_ = True
                        elif True:
                            d_17_appendedGenerated_: _dafny.Seq
                            d_18_appendedInside_: bool
                            d_19_appendedCurrent_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                            d_17_appendedGenerated_ = out9_
                            d_18_appendedInside_ = out10_
                            d_19_appendedCurrent_ = out11_
                            generated = d_17_appendedGenerated_
                            insideConstrainedOut = d_18_appendedInside_
                            currentConstrainedOut = d_19_appendedCurrent_
                    elif True:
                        d_20_symbolBudget_: int
                        d_20_symbolBudget_ = (maxSteps) - (d_3_steps_)
                        d_21_symbolOut_: _dafny.Seq
                        d_22_hitEos_: bool
                        d_23_stepsUsed_: int
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: int
                        out12_, out13_, out14_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, d_20_symbolBudget_, eosToken)
                        d_21_symbolOut_ = out12_
                        d_22_hitEos_ = out13_
                        d_23_stepsUsed_ = out14_
                        generated = (d_12_stablePrefix_) + (d_21_symbolOut_)
                        currentConstrainedOut = d_21_symbolOut_
                        d_3_steps_ = (d_3_steps_) + (d_23_stepsUsed_)
                        if d_22_hitEos_:
                            d_4_done_ = True
        cost = d_3_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


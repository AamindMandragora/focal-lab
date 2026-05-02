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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, eosToken):
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
        d_2_done_: bool
        d_2_done_ = False
        while ((d_1_steps_) < (maxSteps)) and (not(d_2_done_)):
            if not(insideConstrainedOut):
                d_3_openedGenerated_: _dafny.Seq
                d_4_openedInside_: bool
                d_5_openedCurrent_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_3_openedGenerated_ = out0_
                d_4_openedInside_ = out1_
                d_5_openedCurrent_ = out2_
                generated = d_3_openedGenerated_
                insideConstrainedOut = d_4_openedInside_
                currentConstrainedOut = d_5_openedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_6_complete_: bool
                d_6_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_6_complete_:
                    if ((d_1_steps_) + (1)) <= (maxSteps):
                        d_7_closedGenerated_: _dafny.Seq
                        d_8_closedInside_: bool
                        d_9_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_7_closedGenerated_ = out3_
                        d_8_closedInside_ = out4_
                        d_9_closedCurrent_ = out5_
                        generated = d_7_closedGenerated_
                        insideConstrainedOut = d_8_closedInside_
                        currentConstrainedOut = d_9_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    d_2_done_ = True
                elif True:
                    d_10_validCount_: int
                    out6_: int
                    out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                    d_10_validCount_ = out6_
                    d_11_narrow_: bool
                    out7_: bool
                    out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                    d_11_narrow_ = out7_
                    if ((not(d_11_narrow_)) and ((d_10_validCount_) > (2))) and ((len(currentConstrainedOut)) > (0)):
                        d_12_rolled_: _dafny.Seq
                        out8_: _dafny.Seq
                        out8_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                        d_12_rolled_ = out8_
                        if (len(d_12_rolled_)) < (len(currentConstrainedOut)):
                            d_13_stablePrefix_: _dafny.Seq
                            d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_14_rolledGenerated_: _dafny.Seq
                            d_15_rolledCurrent_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: _dafny.Seq
                            out9_, out10_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_13_stablePrefix_, generated, currentConstrainedOut)
                            d_14_rolledGenerated_ = out9_
                            d_15_rolledCurrent_ = out10_
                            generated = d_14_rolledGenerated_
                            currentConstrainedOut = d_15_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_16_constrainedPrompt_: _dafny.Seq
                        d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_17_next_: _dafny.Seq
                        out11_: _dafny.Seq
                        out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_17_next_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_17_next_) == (eosToken):
                            d_2_done_ = True
                        elif True:
                            d_18_appendedGenerated_: _dafny.Seq
                            d_19_appendedInside_: bool
                            d_20_appendedCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                            d_18_appendedGenerated_ = out12_
                            d_19_appendedInside_ = out13_
                            d_20_appendedCurrent_ = out14_
                            generated = d_18_appendedGenerated_
                            insideConstrainedOut = d_19_appendedInside_
                            currentConstrainedOut = d_20_appendedCurrent_
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


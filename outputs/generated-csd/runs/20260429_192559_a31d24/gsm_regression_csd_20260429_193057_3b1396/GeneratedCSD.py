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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_2_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_2_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                            if VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_3_openedGenerated_: _dafny.Seq
                                d_4_openedInside_: bool
                                d_5_openedCurrent_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (1):]))
                                d_3_openedGenerated_ = out1_
                                d_4_openedInside_ = out2_
                                d_5_openedCurrent_ = out3_
                                generated = d_3_openedGenerated_
                                insideConstrainedOut = d_4_openedInside_
                                currentConstrainedOut = d_5_openedCurrent_
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out4_
                            d_7_closedInside_ = out5_
                            d_8_closedCurrent_ = out6_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_dead_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_9_dead_ = out7_
                            if d_9_dead_:
                                d_10_repaired_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_10_repaired_ = out8_
                                d_11_stablePrefix_: _dafny.Seq
                                d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_12_repairedGenerated_: _dafny.Seq
                                d_13_repairedCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: _dafny.Seq
                                out9_, out10_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_11_stablePrefix_, generated, currentConstrainedOut)
                                d_12_repairedGenerated_ = out9_
                                d_13_repairedCurrent_ = out10_
                                generated = d_12_repairedGenerated_
                                currentConstrainedOut = d_13_repairedCurrent_
                                insideConstrainedOut = True
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_14_stablePrefix2_: _dafny.Seq
                                d_14_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_15_constrainedPrompt_: _dafny.Seq
                                d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix2_)
                                d_16_next2_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_16_next2_ = out11_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_16_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_17_appendedGenerated_: _dafny.Seq
                                    d_18_appendedInside_: bool
                                    d_19_appendedCurrent_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next2_)
                                    d_17_appendedGenerated_ = out12_
                                    d_18_appendedInside_ = out13_
                                    d_19_appendedCurrent_ = out14_
                                    generated = d_17_appendedGenerated_
                                    insideConstrainedOut = d_18_appendedInside_
                                    currentConstrainedOut = d_19_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


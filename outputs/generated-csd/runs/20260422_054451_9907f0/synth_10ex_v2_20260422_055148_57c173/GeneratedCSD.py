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
        (d_0_helpers_).cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (lm).ChooseNextTokenUnconstrained()
                        d_2_next_ = out0_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_2_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                            if (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_3_isComplete_: bool
                        d_3_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_isComplete_:
                            d_4_closedGenerated_: _dafny.Seq
                            d_5_closedInside_: bool
                            d_6_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_4_closedGenerated_ = out1_
                            d_5_closedInside_ = out2_
                            d_6_closedCurrent_ = out3_
                            generated = d_4_closedGenerated_
                            insideConstrainedOut = d_5_closedInside_
                            currentConstrainedOut = d_6_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_validCount_: int
                            out4_: int
                            out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_7_validCount_ = out4_
                            d_8_narrow_: bool
                            out5_: bool
                            out5_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_8_narrow_ = out5_
                            if ((d_7_validCount_) == (0)) and (d_8_narrow_):
                                d_9_rolled_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_9_rolled_ = out6_
                                currentConstrainedOut = d_9_rolled_
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(generated)) - ((len(generated)) - ((len(currentConstrainedOut)) - (len(d_9_rolled_))))):])
                                d_10_completeAfterRollback_: bool
                                d_10_completeAfterRollback_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_10_completeAfterRollback_:
                                    d_11_closedGenerated2_: _dafny.Seq
                                    d_12_closedInside2_: bool
                                    d_13_closedCurrent2_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_11_closedGenerated2_ = out7_
                                    d_12_closedInside2_ = out8_
                                    d_13_closedCurrent2_ = out9_
                                    generated = d_11_closedGenerated2_
                                    insideConstrainedOut = d_12_closedInside2_
                                    currentConstrainedOut = d_13_closedCurrent2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_14_constrainedPrompt_: _dafny.Seq
                                d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_15_next2_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_15_next2_ = out10_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_15_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_16_appendedGenerated_: _dafny.Seq
                                    d_17_appendedInside_: bool
                                    d_18_appendedCurrent_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next2_)
                                    d_16_appendedGenerated_ = out11_
                                    d_17_appendedInside_ = out12_
                                    d_18_appendedCurrent_ = out13_
                                    generated = d_16_appendedGenerated_
                                    insideConstrainedOut = d_17_appendedInside_
                                    currentConstrainedOut = d_18_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


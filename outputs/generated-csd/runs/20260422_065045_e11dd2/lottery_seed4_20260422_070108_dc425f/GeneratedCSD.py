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
                    if insideConstrainedOut:
                        d_2_completeNow_: bool
                        d_2_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_2_completeNow_:
                            d_3_closedGenerated_: _dafny.Seq
                            d_4_closedInside_: bool
                            d_5_closedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_3_closedGenerated_ = out0_
                            d_4_closedInside_ = out1_
                            d_5_closedCurrent_ = out2_
                            generated = d_3_closedGenerated_
                            insideConstrainedOut = d_4_closedInside_
                            currentConstrainedOut = d_5_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_remainingInside_: int
                            d_6_remainingInside_ = (maxSteps) - (d_1_steps_)
                            if (d_6_remainingInside_) <= (1):
                                raise _dafny.Break("0")
                            elif True:
                                d_7_validCount_: int
                                out3_: int
                                out3_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_7_validCount_ = out3_
                                d_8_narrow_: bool
                                out4_: bool
                                out4_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                                d_8_narrow_ = out4_
                                d_9_constrainedPrompt_: _dafny.Seq
                                d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_10_nextIn_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_10_nextIn_ = out5_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_10_nextIn_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_11_appendedGenerated_: _dafny.Seq
                                    d_12_appendedInside_: bool
                                    d_13_appendedCurrent_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_nextIn_)
                                    d_11_appendedGenerated_ = out6_
                                    d_12_appendedInside_ = out7_
                                    d_13_appendedCurrent_ = out8_
                                    generated = d_11_appendedGenerated_
                                    insideConstrainedOut = d_12_appendedInside_
                                    currentConstrainedOut = d_13_appendedCurrent_
                    elif True:
                        d_14_remaining_: int
                        d_14_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_14_remaining_) >= (2):
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                            d_15_top_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_15_top_ = out9_
                            if (d_15_top_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_16_openedGenerated_: _dafny.Seq
                                d_17_openedInside_: bool
                                d_18_openedCurrent_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_16_openedGenerated_ = out10_
                                d_17_openedInside_ = out11_
                                d_18_openedCurrent_ = out12_
                                generated = d_16_openedGenerated_
                                insideConstrainedOut = d_17_openedInside_
                                currentConstrainedOut = d_18_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_19_nextOut_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_19_nextOut_ = out13_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_19_nextOut_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    if (d_19_nextOut_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_19_nextOut_]))
                                        insideConstrainedOut = True
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    elif True:
                                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_19_nextOut_]))
                        elif True:
                            d_20_nextLast_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_20_nextLast_ = out14_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_20_nextLast_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if (d_20_nextLast_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_20_nextLast_]))
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


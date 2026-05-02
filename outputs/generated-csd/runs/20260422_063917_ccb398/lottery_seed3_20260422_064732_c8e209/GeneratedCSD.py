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
        d_2_constrainedTokensUsed_: int
        d_2_constrainedTokensUsed_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_3_isComplete_: bool
                        d_3_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_isComplete_:
                            d_4_closedGenerated_: _dafny.Seq
                            d_5_closedInside_: bool
                            d_6_closedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_4_closedGenerated_ = out0_
                            d_5_closedInside_ = out1_
                            d_6_closedCurrent_ = out2_
                            generated = d_4_closedGenerated_
                            insideConstrainedOut = d_5_closedInside_
                            currentConstrainedOut = d_6_closedCurrent_
                            d_2_constrainedTokensUsed_ = 0
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_narrow_: bool
                            out3_: bool
                            out3_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_7_narrow_ = out3_
                            if ((d_1_steps_) + (2)) > (maxSteps):
                                raise _dafny.Break("0")
                            elif True:
                                if (d_2_constrainedTokensUsed_) >= (2):
                                    raise _dafny.Break("0")
                                elif True:
                                    if (d_7_narrow_) and ((d_2_constrainedTokensUsed_) > (0)):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_8_constrainedPrompt_: _dafny.Seq
                                        d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                        d_9_nextConstrained_: _dafny.Seq
                                        out4_: _dafny.Seq
                                        out4_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, eosToken)
                                        d_9_nextConstrained_ = out4_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        if (d_9_nextConstrained_) == (eosToken):
                                            raise _dafny.Break("0")
                                        elif True:
                                            d_10_appendedGenerated_: _dafny.Seq
                                            d_11_appendedInside_: bool
                                            d_12_appendedCurrent_: _dafny.Seq
                                            out5_: _dafny.Seq
                                            out6_: bool
                                            out7_: _dafny.Seq
                                            out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_nextConstrained_)
                                            d_10_appendedGenerated_ = out5_
                                            d_11_appendedInside_ = out6_
                                            d_12_appendedCurrent_ = out7_
                                            generated = d_10_appendedGenerated_
                                            insideConstrainedOut = d_11_appendedInside_
                                            currentConstrainedOut = d_12_appendedCurrent_
                                            d_2_constrainedTokensUsed_ = (d_2_constrainedTokensUsed_) + (1)
                    elif True:
                        d_13_shouldOpen_: bool
                        d_13_shouldOpen_ = False
                        if ((d_1_steps_) + (3)) <= (maxSteps):
                            if (0) < (len(generated)):
                                d_14_prev_: _dafny.Seq
                                d_14_prev_ = (generated)[(len(generated)) - (1)]
                                if VerifiedDecoderAgent.default__.Contains(d_14_prev_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0"))):
                                    d_13_shouldOpen_ = True
                                elif True:
                                    if VerifiedDecoderAgent.default__.Contains(d_14_prev_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1"))):
                                        d_13_shouldOpen_ = True
                                    elif True:
                                        if VerifiedDecoderAgent.default__.Contains(d_14_prev_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2"))):
                                            d_13_shouldOpen_ = True
                                        elif True:
                                            if VerifiedDecoderAgent.default__.Contains(d_14_prev_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3"))):
                                                d_13_shouldOpen_ = True
                                            elif True:
                                                if VerifiedDecoderAgent.default__.Contains(d_14_prev_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4"))):
                                                    d_13_shouldOpen_ = True
                                                elif True:
                                                    if VerifiedDecoderAgent.default__.Contains(d_14_prev_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5"))):
                                                        d_13_shouldOpen_ = True
                                                    elif True:
                                                        if VerifiedDecoderAgent.default__.Contains(d_14_prev_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6"))):
                                                            d_13_shouldOpen_ = True
                                                        elif True:
                                                            if VerifiedDecoderAgent.default__.Contains(d_14_prev_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7"))):
                                                                d_13_shouldOpen_ = True
                                                            elif True:
                                                                if VerifiedDecoderAgent.default__.Contains(d_14_prev_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8"))):
                                                                    d_13_shouldOpen_ = True
                                                                elif True:
                                                                    if VerifiedDecoderAgent.default__.Contains(d_14_prev_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))):
                                                                        d_13_shouldOpen_ = True
                        if d_13_shouldOpen_:
                            d_15_openedGenerated_: _dafny.Seq
                            d_16_openedInside_: bool
                            d_17_openedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_15_openedGenerated_ = out8_
                            d_16_openedInside_ = out9_
                            d_17_openedCurrent_ = out10_
                            generated = d_15_openedGenerated_
                            insideConstrainedOut = d_16_openedInside_
                            currentConstrainedOut = d_17_openedCurrent_
                            d_2_constrainedTokensUsed_ = 0
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                            d_18_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (lm).ChooseNextTokenUnconstrained()
                            d_18_next_ = out11_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_18_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if (d_18_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    raise _dafny.Break("0")
                                elif True:
                                    if (d_18_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                        raise _dafny.Break("0")
                                    elif True:
                                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_18_next_]))
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


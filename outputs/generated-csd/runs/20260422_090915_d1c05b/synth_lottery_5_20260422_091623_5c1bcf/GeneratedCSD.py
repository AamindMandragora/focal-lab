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
                        (lm).GenerateLogits((prompt) + (generated))
                        d_2_argmax_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).GetHighestLogitToken(lm)
                        d_2_argmax_ = out0_
                        d_3_argmaxLogit_: _dafny.BigRational
                        out1_: _dafny.BigRational
                        out1_ = (d_0_helpers_).GetTokenLogit(lm, d_2_argmax_)
                        d_3_argmaxLogit_ = out1_
                        d_4_openLogit_: _dafny.BigRational
                        out2_: _dafny.BigRational
                        out2_ = (d_0_helpers_).GetTokenLogit(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        d_4_openLogit_ = out2_
                        d_5_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out3_
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif (((d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_2_argmax_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))))) or ((d_4_openLogit_) >= ((d_3_argmaxLogit_) - (_dafny.BigRational('5e-1')))):
                            generated = (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (1):])) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_6_complete_: bool
                        d_6_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_6_complete_:
                            d_7_closedGenerated_: _dafny.Seq
                            d_8_closedInside_: bool
                            d_9_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_closedGenerated_ = out4_
                            d_8_closedInside_ = out5_
                            d_9_closedCurrent_ = out6_
                            generated = d_7_closedGenerated_
                            insideConstrainedOut = d_8_closedInside_
                            currentConstrainedOut = d_9_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_constrainedPrompt_: _dafny.Seq
                            d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_11_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_11_next_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_12_appendedGenerated_: _dafny.Seq
                                d_13_appendedInside_: bool
                                d_14_appendedCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                                d_12_appendedGenerated_ = out8_
                                d_13_appendedInside_ = out9_
                                d_14_appendedCurrent_ = out10_
                                generated = d_12_appendedGenerated_
                                insideConstrainedOut = d_13_appendedInside_
                                currentConstrainedOut = d_14_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


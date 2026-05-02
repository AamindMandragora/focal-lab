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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        (lm).GenerateLogits((prompt) + (generated))
                        d_2_argmax_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).GetHighestLogitToken(lm)
                        d_2_argmax_ = out0_
                        d_3_hintedOpen_: bool
                        d_3_hintedOpen_ = VerifiedDecoderAgent.default__.Contains(d_2_argmax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        if (d_2_argmax_) == (eosToken):
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif (d_2_argmax_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif d_3_hintedOpen_:
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            d_4_next_: _dafny.Seq
                            out1_: _dafny.Seq
                            out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_4_next_ = out1_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_4_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                                if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_5_complete_: bool
                        d_5_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_complete_:
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out2_
                            d_7_closedInside_ = out3_
                            d_8_closedCurrent_ = out4_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_constrainedPrompt_: _dafny.Seq
                            d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            (lm).GenerateLogits((d_9_constrainedPrompt_) + (currentConstrainedOut))
                            d_10_argmax_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_10_argmax_ = out5_
                            d_11_argmaxValid_: bool
                            out6_: bool
                            out6_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_10_argmax_)
                            d_11_argmaxValid_ = out6_
                            if (d_10_argmax_) == (eosToken):
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif d_11_argmaxValid_:
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_12_appendedGenerated_: _dafny.Seq
                                d_13_appendedInside_: bool
                                d_14_appendedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_argmax_)
                                d_12_appendedGenerated_ = out7_
                                d_13_appendedInside_ = out8_
                                d_14_appendedCurrent_ = out9_
                                generated = d_12_appendedGenerated_
                                insideConstrainedOut = d_13_appendedInside_
                                currentConstrainedOut = d_14_appendedCurrent_
                            elif True:
                                d_15_next_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_15_next_ = out10_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_15_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_16_appendedGenerated2_: _dafny.Seq
                                    d_17_appendedInside2_: bool
                                    d_18_appendedCurrent2_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                    d_16_appendedGenerated2_ = out11_
                                    d_17_appendedInside2_ = out12_
                                    d_18_appendedCurrent2_ = out13_
                                    generated = d_16_appendedGenerated2_
                                    insideConstrainedOut = d_17_appendedInside2_
                                    currentConstrainedOut = d_18_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


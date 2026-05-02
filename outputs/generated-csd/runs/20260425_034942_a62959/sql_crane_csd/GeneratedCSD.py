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
                        (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                        (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
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
                            d_7_constrainedPrompt_: _dafny.Seq
                            d_7_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            (lm).GenerateLogits((d_7_constrainedPrompt_) + (currentConstrainedOut))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                            d_8_argmax_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_8_argmax_ = out4_
                            d_9_argmaxValid_: bool
                            out5_: bool
                            out5_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_8_argmax_)
                            d_9_argmaxValid_ = out5_
                            if ((d_8_argmax_) != (eosToken)) and (d_9_argmaxValid_):
                                d_10_appendedGenerated_: _dafny.Seq
                                d_11_appendedInside_: bool
                                d_12_appendedCurrent_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_argmax_)
                                d_10_appendedGenerated_ = out6_
                                d_11_appendedInside_ = out7_
                                d_12_appendedCurrent_ = out8_
                                generated = d_10_appendedGenerated_
                                insideConstrainedOut = d_11_appendedInside_
                                currentConstrainedOut = d_12_appendedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_13_next_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_13_next_ = out9_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_13_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_14_appendedGenerated_: _dafny.Seq
                                    d_15_appendedInside_: bool
                                    d_16_appendedCurrent_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                    d_14_appendedGenerated_ = out10_
                                    d_15_appendedInside_ = out11_
                                    d_16_appendedCurrent_ = out12_
                                    generated = d_14_appendedGenerated_
                                    insideConstrainedOut = d_15_appendedInside_
                                    currentConstrainedOut = d_16_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


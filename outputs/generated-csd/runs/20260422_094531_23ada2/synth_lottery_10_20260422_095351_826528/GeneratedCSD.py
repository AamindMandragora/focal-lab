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
        d_2_openedHere_: bool
        d_2_openedHere_ = insideConstrained
        d_3_leadSteps_: int
        d_3_leadSteps_ = 0
        d_4_constrainedLen_: int
        d_4_constrainedLen_ = len(currentConstrainedOut)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_4_constrainedLen_ = len(currentConstrainedOut)
                        d_5_completeNow_: bool
                        d_5_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_completeNow_:
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out0_
                            d_7_closedInside_ = out1_
                            d_8_closedCurrent_ = out2_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_4_constrainedLen_ = 0
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if ((d_1_steps_) + (1)) >= (maxSteps):
                                raise _dafny.Break("0")
                            elif True:
                                if (d_4_constrainedLen_) >= (8):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_9_constrainedPrompt_: _dafny.Seq
                                    d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_10_nextConstrained_: _dafny.Seq
                                    out3_: _dafny.Seq
                                    out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_10_nextConstrained_ = out3_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_10_nextConstrained_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_11_appendedGenerated_: _dafny.Seq
                                        d_12_appendedInside_: bool
                                        d_13_appendedCurrent_: _dafny.Seq
                                        out4_: _dafny.Seq
                                        out5_: bool
                                        out6_: _dafny.Seq
                                        out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_nextConstrained_)
                                        d_11_appendedGenerated_ = out4_
                                        d_12_appendedInside_ = out5_
                                        d_13_appendedCurrent_ = out6_
                                        generated = d_11_appendedGenerated_
                                        insideConstrainedOut = d_12_appendedInside_
                                        currentConstrainedOut = d_13_appendedCurrent_
                                        d_4_constrainedLen_ = len(currentConstrainedOut)
                    elif True:
                        if ((not(d_2_openedHere_)) and ((d_3_leadSteps_) >= (6))) and (((d_1_steps_) + (3)) <= (maxSteps)):
                            d_14_openedGenerated_: _dafny.Seq
                            d_15_openedInside_: bool
                            d_16_openedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_14_openedGenerated_ = out7_
                            d_15_openedInside_ = out8_
                            d_16_openedCurrent_ = out9_
                            generated = d_14_openedGenerated_
                            insideConstrainedOut = d_15_openedInside_
                            currentConstrainedOut = d_16_openedCurrent_
                            d_2_openedHere_ = True
                            d_4_constrainedLen_ = 0
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                            d_17_next_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (lm).ChooseNextToken()
                            d_17_next_ = out10_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_17_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_17_next_]))
                                d_3_leadSteps_ = (d_3_leadSteps_) + (1)
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


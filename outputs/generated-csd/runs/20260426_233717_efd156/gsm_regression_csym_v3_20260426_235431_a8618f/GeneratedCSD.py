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
                d_3_produced_: int
                d_3_produced_ = (len(generated)) - (len(generatedPrefix))
                d_4_remaining_: int
                d_4_remaining_ = (maxSteps) - (d_1_steps_)
                if ((d_3_produced_) < (2)) or ((d_4_remaining_) < (3)):
                    d_5_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_5_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_5_next_) == (eosToken):
                        d_2_done_ = True
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                elif True:
                    (lm).GenerateLogits((prompt) + (generated))
                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'))
                    d_6_nextOpen_: _dafny.Seq
                    out1_: _dafny.Seq
                    out1_ = (lm).ChooseNextToken()
                    d_6_nextOpen_ = out1_
                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_6_nextOpen_) == (eosToken):
                        d_2_done_ = True
                    elif True:
                        if (d_6_nextOpen_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_7_openedGenerated_: _dafny.Seq
                            d_8_openedInside_: bool
                            d_9_openedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_openedGenerated_ = out2_
                            d_8_openedInside_ = out3_
                            d_9_openedCurrent_ = out4_
                            generated = d_7_openedGenerated_
                            insideConstrainedOut = d_8_openedInside_
                            currentConstrainedOut = d_9_openedCurrent_
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_nextOpen_]))
            elif True:
                d_10_complete_: bool
                d_10_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_10_complete_:
                    d_11_closedGenerated_: _dafny.Seq
                    d_12_closedInside_: bool
                    d_13_closedCurrent_: _dafny.Seq
                    out5_: _dafny.Seq
                    out6_: bool
                    out7_: _dafny.Seq
                    out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_11_closedGenerated_ = out5_
                    d_12_closedInside_ = out6_
                    d_13_closedCurrent_ = out7_
                    generated = d_11_closedGenerated_
                    insideConstrainedOut = d_12_closedInside_
                    currentConstrainedOut = d_13_closedCurrent_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    if ((maxSteps) - (d_1_steps_)) == (1):
                        d_2_done_ = True
                    elif True:
                        d_14_next2_: _dafny.Seq
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                        d_14_next2_ = out8_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_14_next2_) == (eosToken):
                            d_2_done_ = True
                        elif True:
                            d_15_appendedGenerated_: _dafny.Seq
                            d_16_appendedInside_: bool
                            d_17_appendedCurrent_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next2_)
                            d_15_appendedGenerated_ = out9_
                            d_16_appendedInside_ = out10_
                            d_17_appendedCurrent_ = out11_
                            generated = d_15_appendedGenerated_
                            insideConstrainedOut = d_16_appendedInside_
                            currentConstrainedOut = d_17_appendedCurrent_
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


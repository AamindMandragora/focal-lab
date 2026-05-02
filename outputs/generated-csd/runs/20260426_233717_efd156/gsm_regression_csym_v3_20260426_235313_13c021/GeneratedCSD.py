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
                (lm).GenerateLogits((prompt) + (generated))
                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                d_3_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (lm).ChooseNextToken()
                d_3_next_ = out0_
                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_3_next_) == (eosToken):
                    d_2_done_ = True
                elif True:
                    if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_4_openedGenerated_: _dafny.Seq
                        d_5_openedInside_: bool
                        d_6_openedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_4_openedGenerated_ = out1_
                        d_5_openedInside_ = out2_
                        d_6_openedCurrent_ = out3_
                        generated = d_4_openedGenerated_
                        insideConstrainedOut = d_5_openedInside_
                        currentConstrainedOut = d_6_openedCurrent_
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
            elif True:
                d_7_complete_: bool
                d_7_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_7_complete_:
                    d_8_closedGenerated_: _dafny.Seq
                    d_9_closedInside_: bool
                    d_10_closedCurrent_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_8_closedGenerated_ = out4_
                    d_9_closedInside_ = out5_
                    d_10_closedCurrent_ = out6_
                    generated = d_8_closedGenerated_
                    insideConstrainedOut = d_9_closedInside_
                    currentConstrainedOut = d_10_closedCurrent_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    if ((maxSteps) - (d_1_steps_)) == (1):
                        d_2_done_ = True
                    elif True:
                        d_11_next2_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                        d_11_next2_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_11_next2_) == (eosToken):
                            d_2_done_ = True
                        elif True:
                            d_12_appendedGenerated_: _dafny.Seq
                            d_13_appendedInside_: bool
                            d_14_appendedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next2_)
                            d_12_appendedGenerated_ = out8_
                            d_13_appendedInside_ = out9_
                            d_14_appendedCurrent_ = out10_
                            generated = d_12_appendedGenerated_
                            insideConstrainedOut = d_13_appendedInside_
                            currentConstrainedOut = d_14_appendedCurrent_
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


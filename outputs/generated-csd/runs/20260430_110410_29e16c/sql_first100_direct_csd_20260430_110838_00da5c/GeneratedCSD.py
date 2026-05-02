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
                            if VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens):
                                    d_3_openedGenerated_: _dafny.Seq
                                    d_4_openedInside_: bool
                                    d_5_openedCurrent_: _dafny.Seq
                                    out1_: _dafny.Seq
                                    out2_: bool
                                    out3_: _dafny.Seq
                                    out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_3_openedGenerated_ = out1_
                                    d_4_openedInside_ = out2_
                                    d_5_openedCurrent_ = out3_
                                    generated = d_3_openedGenerated_
                                    insideConstrainedOut = d_4_openedInside_
                                    currentConstrainedOut = d_5_openedCurrent_
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))) in ((lm).Tokens):
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
                                raise _dafny.Break("0")
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            if (len(validTokenGroups)) > (0):
                                (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_9_nextConstrained_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (lm).ChooseNextToken()
                            d_9_nextConstrained_ = out7_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_nextConstrained_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_10_appendedGenerated_: _dafny.Seq
                                d_11_appendedInside_: bool
                                d_12_appendedCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_nextConstrained_)
                                d_10_appendedGenerated_ = out8_
                                d_11_appendedInside_ = out9_
                                d_12_appendedCurrent_ = out10_
                                generated = d_10_appendedGenerated_
                                insideConstrainedOut = d_11_appendedInside_
                                currentConstrainedOut = d_12_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


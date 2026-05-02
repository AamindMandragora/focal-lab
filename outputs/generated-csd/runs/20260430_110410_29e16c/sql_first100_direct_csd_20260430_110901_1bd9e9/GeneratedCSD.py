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
                        if ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)) and ((len(validTokenGroups)) > (0)):
                            d_2_flat_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                            d_2_flat_ = out0_
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in (d_2_flat_):
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
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_6_next_: _dafny.Seq
                                out4_: _dafny.Seq
                                out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_6_next_ = out4_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_6_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                        elif True:
                            d_7_next2_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next2_ = out5_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next2_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next2_]))
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))) in ((lm).Tokens):
                                d_8_closedGenerated_: _dafny.Seq
                                d_9_closedInside_: bool
                                d_10_closedCurrent_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_8_closedGenerated_ = out6_
                                d_9_closedInside_ = out7_
                                d_10_closedCurrent_ = out8_
                                generated = d_8_closedGenerated_
                                insideConstrainedOut = d_9_closedInside_
                                currentConstrainedOut = d_10_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            if (len(validTokenGroups)) > (0):
                                (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_11_nextConstrained_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (lm).ChooseNextToken()
                            d_11_nextConstrained_ = out9_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_11_nextConstrained_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_12_appendedGenerated_: _dafny.Seq
                                d_13_appendedInside_: bool
                                d_14_appendedCurrent_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_nextConstrained_)
                                d_12_appendedGenerated_ = out10_
                                d_13_appendedInside_ = out11_
                                d_14_appendedCurrent_ = out12_
                                generated = d_12_appendedGenerated_
                                insideConstrainedOut = d_13_appendedInside_
                                currentConstrainedOut = d_14_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


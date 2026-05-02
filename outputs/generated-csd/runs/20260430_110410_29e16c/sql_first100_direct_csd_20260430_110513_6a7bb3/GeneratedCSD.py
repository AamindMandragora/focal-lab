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
        d_2_activeGroup_: int
        d_2_activeGroup_ = -1
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif VerifiedDecoderAgent.default__.Contains(d_3_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens):
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
                                d_2_activeGroup_ = -1
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                    elif True:
                        d_7_completeNow_: bool
                        d_7_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_completeNow_:
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
                            d_2_activeGroup_ = -1
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            if (len(validTokenGroups)) > (0):
                                (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'))
                                if ((d_2_activeGroup_) >= (0)) and ((d_2_activeGroup_) < (len(validTokenGroups))):
                                    d_11_activeHasValid_: bool
                                    out7_: bool
                                    out7_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, (validTokenGroups)[d_2_activeGroup_])
                                    d_11_activeHasValid_ = out7_
                                    if d_11_activeHasValid_:
                                        (d_0_helpers_).BoostTokenLogits(lm, (validTokenGroups)[d_2_activeGroup_], _dafny.BigRational('8e0'))
                            d_12_constrainedNext_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).ConstrainedSample(lm, parser, currentConstrainedOut, eosToken)
                            d_12_constrainedNext_ = out8_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_12_constrainedNext_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_13_appendedGenerated_: _dafny.Seq
                                d_14_appendedInside_: bool
                                d_15_appendedCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_constrainedNext_)
                                d_13_appendedGenerated_ = out9_
                                d_14_appendedInside_ = out10_
                                d_15_appendedCurrent_ = out11_
                                generated = d_13_appendedGenerated_
                                insideConstrainedOut = d_14_appendedInside_
                                currentConstrainedOut = d_15_appendedCurrent_
                                d_16_groupIdx_: int
                                out12_: int
                                out12_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_12_constrainedNext_)
                                d_16_groupIdx_ = out12_
                                if (d_16_groupIdx_) >= (0):
                                    d_2_activeGroup_ = d_16_groupIdx_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


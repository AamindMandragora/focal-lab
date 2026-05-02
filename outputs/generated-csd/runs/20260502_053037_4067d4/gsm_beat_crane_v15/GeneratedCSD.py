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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_lastBeforeClose_: _dafny.Seq
        d_2_lastBeforeClose_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        d_3_sawCloseContext_: bool
        d_3_sawCloseContext_ = False
        d_4_narrowThreshold_: int
        d_4_narrowThreshold_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_prevTok_: _dafny.Seq
                        d_6_foundPrev_: bool
                        out0_: _dafny.Seq
                        out1_: bool
                        out0_, out1_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                        d_5_prevTok_ = out0_
                        d_6_foundPrev_ = out1_
                        if d_6_foundPrev_:
                            d_2_lastBeforeClose_ = d_5_prevTok_
                            d_3_sawCloseContext_ = True
                        d_7_shouldOpen_: bool
                        d_7_shouldOpen_ = False
                        if d_3_sawCloseContext_:
                            if (((((d_2_lastBeforeClose_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))) or ((d_2_lastBeforeClose_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+"))))) or ((d_2_lastBeforeClose_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-"))))) or ((d_2_lastBeforeClose_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*"))))) or ((d_2_lastBeforeClose_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")))):
                                d_7_shouldOpen_ = True
                        if (d_7_shouldOpen_) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)):
                            d_8_openedGenerated_: _dafny.Seq
                            d_9_openedInside_: bool
                            d_10_openedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_openedGenerated_ = out2_
                            d_9_openedInside_ = out3_
                            d_10_openedCurrent_ = out4_
                            generated = d_8_openedGenerated_
                            insideConstrainedOut = d_9_openedInside_
                            currentConstrainedOut = d_10_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_sawCloseContext_ = False
                        elif True:
                            d_11_next_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_next_ = out5_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                if (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_3_sawCloseContext_ = False
                    elif True:
                        d_12_complete_: bool
                        d_12_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_12_complete_:
                            d_13_closedGenerated_: _dafny.Seq
                            d_14_closedInside_: bool
                            d_15_closedCurrent_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_13_closedGenerated_ = out6_
                            d_14_closedInside_ = out7_
                            d_15_closedCurrent_ = out8_
                            generated = d_13_closedGenerated_
                            insideConstrainedOut = d_14_closedInside_
                            currentConstrainedOut = d_15_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_16_stablePrefix_: _dafny.Seq
                            d_16_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_17_next_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_16_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_4_narrowThreshold_, eosToken)
                            d_17_next_ = out9_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_17_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_18_appendedGenerated_: _dafny.Seq
                                d_19_appendedInside_: bool
                                d_20_appendedCurrent_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                d_18_appendedGenerated_ = out10_
                                d_19_appendedInside_ = out11_
                                d_20_appendedCurrent_ = out12_
                                generated = d_18_appendedGenerated_
                                insideConstrainedOut = d_19_appendedInside_
                                currentConstrainedOut = d_20_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


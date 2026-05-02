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
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_1_steps_) >= (24)) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)):
                            d_3_openedGenerated_: _dafny.Seq
                            d_4_openedInside_: bool
                            d_5_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_3_openedGenerated_ = out0_
                            d_4_openedInside_ = out1_
                            d_5_openedCurrent_ = out2_
                            generated = d_3_openedGenerated_
                            insideConstrainedOut = d_4_openedInside_
                            currentConstrainedOut = d_5_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_6_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                                if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
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
                            d_11_stablePrefix_: _dafny.Seq
                            d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_12_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_12_validCount_ = out7_
                            if (d_12_validCount_) <= (d_2_narrowThreshold_):
                                d_13_g_: _dafny.Seq
                                d_14_i_: bool
                                d_15_c_: _dafny.Seq
                                d_16_hitEos_: bool
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out11_: bool
                                out8_, out9_, out10_, out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (d_11_stablePrefix_), generated, currentConstrainedOut, eosToken)
                                d_13_g_ = out8_
                                d_14_i_ = out9_
                                d_15_c_ = out10_
                                d_16_hitEos_ = out11_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if d_16_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = d_13_g_
                                    insideConstrainedOut = d_14_i_
                                    currentConstrainedOut = d_15_c_
                            elif True:
                                d_17_next_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_11_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_17_next_ = out12_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_17_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_18_appendedGenerated_: _dafny.Seq
                                    d_19_appendedInside_: bool
                                    d_20_appendedCurrent_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                    d_18_appendedGenerated_ = out13_
                                    d_19_appendedInside_ = out14_
                                    d_20_appendedCurrent_ = out15_
                                    generated = d_18_appendedGenerated_
                                    insideConstrainedOut = d_19_appendedInside_
                                    currentConstrainedOut = d_20_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


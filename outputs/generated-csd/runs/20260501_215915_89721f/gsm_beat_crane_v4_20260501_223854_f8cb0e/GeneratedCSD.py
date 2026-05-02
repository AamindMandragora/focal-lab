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
        d_2_openedAnySpan_: bool
        d_2_openedAnySpan_ = insideConstrained
        d_3_forceAfter_: int
        d_3_forceAfter_ = 8
        d_4_narrowThreshold_: int
        d_4_narrowThreshold_ = 8
        d_5_maxBoostedSpanLen_: int
        d_5_maxBoostedSpanLen_ = 6
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_2_openedAnySpan_)) and ((d_1_steps_) >= (d_3_forceAfter_)):
                            d_6_openedGenerated_: _dafny.Seq
                            d_7_openedInside_: bool
                            d_8_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openedGenerated_ = out0_
                            d_7_openedInside_ = out1_
                            d_8_openedCurrent_ = out2_
                            generated = d_6_openedGenerated_
                            insideConstrainedOut = d_7_openedInside_
                            currentConstrainedOut = d_8_openedCurrent_
                            d_2_openedAnySpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_openedAnySpan_ = True
                    elif True:
                        d_10_complete_: bool
                        d_10_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_10_complete_:
                            d_11_closedGenerated_: _dafny.Seq
                            d_12_closedInside_: bool
                            d_13_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_closedGenerated_ = out4_
                            d_12_closedInside_ = out5_
                            d_13_closedCurrent_ = out6_
                            generated = d_11_closedGenerated_
                            insideConstrainedOut = d_12_closedInside_
                            currentConstrainedOut = d_13_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_14_stablePrefix_: _dafny.Seq
                            d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            if ((len(validTokenGroups)) > (0)) and ((len(currentConstrainedOut)) < (d_5_maxBoostedSpanLen_)):
                                d_15_next_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_14_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_4_narrowThreshold_, eosToken)
                                d_15_next_ = out7_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_15_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_16_appendedGenerated_: _dafny.Seq
                                    d_17_appendedInside_: bool
                                    d_18_appendedCurrent_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                    d_16_appendedGenerated_ = out8_
                                    d_17_appendedInside_ = out9_
                                    d_18_appendedCurrent_ = out10_
                                    generated = d_16_appendedGenerated_
                                    insideConstrainedOut = d_17_appendedInside_
                                    currentConstrainedOut = d_18_appendedCurrent_
                            elif True:
                                d_19_constrainedPrompt_: _dafny.Seq
                                d_19_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                                d_20_g_: _dafny.Seq
                                d_21_i_: bool
                                d_22_c_: _dafny.Seq
                                d_23_hitEos_: bool
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out14_: bool
                                out11_, out12_, out13_, out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_19_constrainedPrompt_, generated, currentConstrainedOut, eosToken)
                                d_20_g_ = out11_
                                d_21_i_ = out12_
                                d_22_c_ = out13_
                                d_23_hitEos_ = out14_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if d_23_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = d_20_g_
                                    insideConstrainedOut = d_21_i_
                                    currentConstrainedOut = d_22_c_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


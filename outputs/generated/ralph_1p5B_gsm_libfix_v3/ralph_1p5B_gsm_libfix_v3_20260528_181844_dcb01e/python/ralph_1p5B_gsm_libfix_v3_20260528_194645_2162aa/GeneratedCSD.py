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
        d_2_spansCompleted_: int
        d_2_spansCompleted_ = 0
        d_3_freeTokensSinceSpan_: int
        d_3_freeTokensSinceSpan_ = 0
        d_4_maxFreeBeforeSpan_: int
        d_4_maxFreeBeforeSpan_ = 150
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_budgetLeft_: int
                        d_5_budgetLeft_ = (maxSteps) - (d_1_steps_)
                        d_6_shouldForceSpan_: bool
                        d_6_shouldForceSpan_ = False
                        if ((d_2_spansCompleted_) == (0)) and ((d_3_freeTokensSinceSpan_) >= (d_4_maxFreeBeforeSpan_)):
                            d_6_shouldForceSpan_ = True
                        if (((d_2_spansCompleted_) == (0)) and ((d_5_budgetLeft_) <= (5))) and ((d_5_budgetLeft_) >= (3)):
                            d_6_shouldForceSpan_ = True
                        if ((d_2_spansCompleted_) >= (1)) and ((d_3_freeTokensSinceSpan_) >= (30)):
                            d_6_shouldForceSpan_ = True
                        if (((d_2_spansCompleted_) == (0)) and ((d_5_budgetLeft_) <= (60))) and ((d_3_freeTokensSinceSpan_) >= (10)):
                            d_6_shouldForceSpan_ = True
                        if (d_6_shouldForceSpan_) and (((d_1_steps_) + (1)) <= (maxSteps)):
                            d_7_g2_: _dafny.Seq
                            d_8_ins2_: bool
                            d_9_cur2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_g2_ = out0_
                            d_8_ins2_ = out1_
                            d_9_cur2_ = out2_
                            generated = d_7_g2_
                            insideConstrainedOut = d_8_ins2_
                            currentConstrainedOut = d_9_cur2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_freeTokensSinceSpan_ = 0
                        elif True:
                            d_10_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                if ((d_2_spansCompleted_) == (0)) and (((d_1_steps_) + (3)) <= (maxSteps)):
                                    d_11_g2_: _dafny.Seq
                                    d_12_ins2_: bool
                                    d_13_cur2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_11_g2_ = out4_
                                    d_12_ins2_ = out5_
                                    d_13_cur2_ = out6_
                                    generated = d_11_g2_
                                    insideConstrainedOut = d_12_ins2_
                                    currentConstrainedOut = d_13_cur2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_freeTokensSinceSpan_ = 0
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                                d_3_freeTokensSinceSpan_ = (d_3_freeTokensSinceSpan_) + (1)
                                if (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_14_g2_: _dafny.Seq
                                    d_15_ins2_: bool
                                    d_16_cur2_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_14_g2_ = out7_
                                    d_15_ins2_ = out8_
                                    d_16_cur2_ = out9_
                                    generated = d_14_g2_
                                    insideConstrainedOut = d_15_ins2_
                                    currentConstrainedOut = d_16_cur2_
                                    d_3_freeTokensSinceSpan_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_17_closedGenerated_: _dafny.Seq
                        d_18_closedInside_: bool
                        d_19_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_closedGenerated_ = out10_
                        d_18_closedInside_ = out11_
                        d_19_closedCurrent_ = out12_
                        generated = d_17_closedGenerated_
                        insideConstrainedOut = d_18_closedInside_
                        currentConstrainedOut = d_19_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spansCompleted_ = (d_2_spansCompleted_) + (1)
                        d_3_freeTokensSinceSpan_ = 0
                    elif True:
                        d_20_next_: _dafny.Seq
                        out13_: _dafny.Seq
                        out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                        d_20_next_ = out13_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_20_next_) == (eosToken):
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                d_21_closedGenerated_: _dafny.Seq
                                d_22_closedInside_: bool
                                d_23_closedCurrent_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_21_closedGenerated_ = out14_
                                d_22_closedInside_ = out15_
                                d_23_closedCurrent_ = out16_
                                generated = d_21_closedGenerated_
                                insideConstrainedOut = d_22_closedInside_
                                currentConstrainedOut = d_23_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spansCompleted_ = (d_2_spansCompleted_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_24_appendedGenerated_: _dafny.Seq
                            d_25_appendedInside_: bool
                            d_26_appendedCurrent_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: _dafny.Seq
                            out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                            d_24_appendedGenerated_ = out17_
                            d_25_appendedInside_ = out18_
                            d_26_appendedCurrent_ = out19_
                            generated = d_24_appendedGenerated_
                            insideConstrainedOut = d_25_appendedInside_
                            currentConstrainedOut = d_26_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


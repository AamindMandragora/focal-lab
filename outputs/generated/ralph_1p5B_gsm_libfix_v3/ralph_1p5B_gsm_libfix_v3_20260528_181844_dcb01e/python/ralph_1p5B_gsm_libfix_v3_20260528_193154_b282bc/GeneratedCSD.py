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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Show each calculation and the final answer inside << >> delimiters, for example <<3*4=12>> or <<42>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeTokensSinceSpan_: int
        d_2_freeTokensSinceSpan_ = 0
        d_3_spansOpened_: int
        d_3_spansOpened_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_shouldForceSpan_: bool
                        d_4_shouldForceSpan_ = False
                        if ((d_3_spansOpened_) == (0)) and ((d_2_freeTokensSinceSpan_) >= (40)):
                            d_4_shouldForceSpan_ = True
                        if ((d_3_spansOpened_) > (0)) and ((d_2_freeTokensSinceSpan_) >= (20)):
                            d_4_shouldForceSpan_ = True
                        if (((maxSteps) - (d_1_steps_)) <= (5)) and ((d_3_spansOpened_) == (0)):
                            d_4_shouldForceSpan_ = True
                        if (d_4_shouldForceSpan_) and (((d_1_steps_) + (1)) <= (maxSteps)):
                            d_5_g2_: _dafny.Seq
                            d_6_ins2_: bool
                            d_7_cur2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_g2_ = out0_
                            d_6_ins2_ = out1_
                            d_7_cur2_ = out2_
                            generated = d_5_g2_
                            insideConstrainedOut = d_6_ins2_
                            currentConstrainedOut = d_7_cur2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spansOpened_ = (d_3_spansOpened_) + (1)
                            d_2_freeTokensSinceSpan_ = 0
                        elif True:
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                if ((d_3_spansOpened_) == (0)) and (((d_1_steps_) + (3)) <= (maxSteps)):
                                    d_9_g2_: _dafny.Seq
                                    d_10_ins2_: bool
                                    d_11_cur2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_9_g2_ = out4_
                                    d_10_ins2_ = out5_
                                    d_11_cur2_ = out6_
                                    generated = d_9_g2_
                                    insideConstrainedOut = d_10_ins2_
                                    currentConstrainedOut = d_11_cur2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spansOpened_ = (d_3_spansOpened_) + (1)
                                    d_2_freeTokensSinceSpan_ = 0
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                d_2_freeTokensSinceSpan_ = (d_2_freeTokensSinceSpan_) + (1)
                                if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_12_g2_: _dafny.Seq
                                    d_13_ins2_: bool
                                    d_14_cur2_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_12_g2_ = out7_
                                    d_13_ins2_ = out8_
                                    d_14_cur2_ = out9_
                                    generated = d_12_g2_
                                    insideConstrainedOut = d_13_ins2_
                                    currentConstrainedOut = d_14_cur2_
                                    d_3_spansOpened_ = (d_3_spansOpened_) + (1)
                                    d_2_freeTokensSinceSpan_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_closedGenerated_: _dafny.Seq
                        d_16_closedInside_: bool
                        d_17_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_closedGenerated_ = out10_
                        d_16_closedInside_ = out11_
                        d_17_closedCurrent_ = out12_
                        generated = d_15_closedGenerated_
                        insideConstrainedOut = d_16_closedInside_
                        currentConstrainedOut = d_17_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_18_next_: _dafny.Seq
                        out13_: _dafny.Seq
                        out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                        d_18_next_ = out13_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_18_next_) == (eosToken):
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                d_19_closedGenerated_: _dafny.Seq
                                d_20_closedInside_: bool
                                d_21_closedCurrent_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_19_closedGenerated_ = out14_
                                d_20_closedInside_ = out15_
                                d_21_closedCurrent_ = out16_
                                generated = d_19_closedGenerated_
                                insideConstrainedOut = d_20_closedInside_
                                currentConstrainedOut = d_21_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_22_appendedGenerated_: _dafny.Seq
                            d_23_appendedInside_: bool
                            d_24_appendedCurrent_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: _dafny.Seq
                            out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                            d_22_appendedGenerated_ = out17_
                            d_23_appendedInside_ = out18_
                            d_24_appendedCurrent_ = out19_
                            generated = d_22_appendedGenerated_
                            insideConstrainedOut = d_23_appendedInside_
                            currentConstrainedOut = d_24_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


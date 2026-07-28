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
        d_2_spansOpened_: int
        d_2_spansOpened_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_2_spansOpened_) < (3)) and (((d_1_steps_) + (3)) <= (maxSteps)):
                            d_3_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_3_next_ = out0_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_3_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                                d_4_g2_: _dafny.Seq
                                d_5_ins2_: bool
                                d_6_cur2_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_4_g2_ = out1_
                                d_5_ins2_ = out2_
                                d_6_cur2_ = out3_
                                generated = d_4_g2_
                                insideConstrainedOut = d_5_ins2_
                                currentConstrainedOut = d_6_cur2_
                                d_2_spansOpened_ = (d_2_spansOpened_) + (1)
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                                if ((_dafny.euclidian_modulus(d_1_steps_, 60)) == (0)) and (((d_1_steps_) + (2)) <= (maxSteps)):
                                    d_7_g2_: _dafny.Seq
                                    d_8_ins2_: bool
                                    d_9_cur2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_7_g2_ = out4_
                                    d_8_ins2_ = out5_
                                    d_9_cur2_ = out6_
                                    generated = d_7_g2_
                                    insideConstrainedOut = d_8_ins2_
                                    currentConstrainedOut = d_9_cur2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_2_spansOpened_ = (d_2_spansOpened_) + (1)
                        elif (d_2_spansOpened_) >= (3):
                            d_10_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_next_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                        elif True:
                            d_11_next_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_next_ = out8_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_closedGenerated_: _dafny.Seq
                        d_13_closedInside_: bool
                        d_14_closedCurrent_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: bool
                        out11_: _dafny.Seq
                        out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_closedGenerated_ = out9_
                        d_13_closedInside_ = out10_
                        d_14_closedCurrent_ = out11_
                        generated = d_12_closedGenerated_
                        insideConstrainedOut = d_13_closedInside_
                        currentConstrainedOut = d_14_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_15_next_: _dafny.Seq
                        out12_: _dafny.Seq
                        out12_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                        d_15_next_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_15_next_) == (eosToken):
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                d_16_closedGenerated_: _dafny.Seq
                                d_17_closedInside_: bool
                                d_18_closedCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_16_closedGenerated_ = out13_
                                d_17_closedInside_ = out14_
                                d_18_closedCurrent_ = out15_
                                generated = d_16_closedGenerated_
                                insideConstrainedOut = d_17_closedInside_
                                currentConstrainedOut = d_18_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_19_valid_: bool
                            out16_: bool
                            out16_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_15_next_)
                            d_19_valid_ = out16_
                            if d_19_valid_:
                                d_20_appendedGenerated_: _dafny.Seq
                                d_21_appendedInside_: bool
                                d_22_appendedCurrent_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                d_20_appendedGenerated_ = out17_
                                d_21_appendedInside_ = out18_
                                d_22_appendedCurrent_ = out19_
                                generated = d_20_appendedGenerated_
                                insideConstrainedOut = d_21_appendedInside_
                                currentConstrainedOut = d_22_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


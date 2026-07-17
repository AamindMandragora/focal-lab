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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Put each calculation and the final numeric answer inside << >> delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeTokensSinceSpan_: int
        d_2_freeTokensSinceSpan_ = 0
        d_3_forceOpenNext_: bool
        d_3_forceOpenNext_ = not(insideConstrained)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_3_forceOpenNext_) and (((d_1_steps_) + (1)) <= (maxSteps)):
                            d_4_g2_: _dafny.Seq
                            d_5_ins2_: bool
                            d_6_cur2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_g2_ = out0_
                            d_5_ins2_ = out1_
                            d_6_cur2_ = out2_
                            generated = d_4_g2_
                            insideConstrainedOut = d_5_ins2_
                            currentConstrainedOut = d_6_cur2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_forceOpenNext_ = False
                            d_2_freeTokensSinceSpan_ = 0
                        elif True:
                            d_7_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                if (len(generated)) > (len(generatedPrefix)):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_3_forceOpenNext_ = True
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                d_2_freeTokensSinceSpan_ = (d_2_freeTokensSinceSpan_) + (1)
                                if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_8_g2_: _dafny.Seq
                                    d_9_ins2_: bool
                                    d_10_cur2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_8_g2_ = out4_
                                    d_9_ins2_ = out5_
                                    d_10_cur2_ = out6_
                                    generated = d_8_g2_
                                    insideConstrainedOut = d_9_ins2_
                                    currentConstrainedOut = d_10_cur2_
                                    d_2_freeTokensSinceSpan_ = 0
                                    d_3_forceOpenNext_ = False
                                elif (d_2_freeTokensSinceSpan_) >= (15):
                                    d_3_forceOpenNext_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_11_closedGenerated_: _dafny.Seq
                        d_12_closedInside_: bool
                        d_13_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_11_closedGenerated_ = out7_
                        d_12_closedInside_ = out8_
                        d_13_closedCurrent_ = out9_
                        generated = d_11_closedGenerated_
                        insideConstrainedOut = d_12_closedInside_
                        currentConstrainedOut = d_13_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_freeTokensSinceSpan_ = 0
                        d_3_forceOpenNext_ = False
                    elif True:
                        d_14_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                        d_14_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_14_next_) == (eosToken):
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_15_closedGenerated_: _dafny.Seq
                                d_16_closedInside_: bool
                                d_17_closedCurrent_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_15_closedGenerated_ = out11_
                                d_16_closedInside_ = out12_
                                d_17_closedCurrent_ = out13_
                                generated = d_15_closedGenerated_
                                insideConstrainedOut = d_16_closedInside_
                                currentConstrainedOut = d_17_closedCurrent_
                                if (d_1_steps_) < (maxSteps):
                                    d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_18_appendedGenerated_: _dafny.Seq
                            d_19_appendedInside_: bool
                            d_20_appendedCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                            d_18_appendedGenerated_ = out14_
                            d_19_appendedInside_ = out15_
                            d_20_appendedCurrent_ = out16_
                            generated = d_18_appendedGenerated_
                            insideConstrainedOut = d_19_appendedInside_
                            currentConstrainedOut = d_20_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


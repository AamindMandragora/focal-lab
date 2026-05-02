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
        d_2_preferFinishLen_: int
        d_2_preferFinishLen_ = 18
        d_3_hardFinishLen_: int
        d_3_hardFinishLen_ = 36
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_5_openedGenerated_: _dafny.Seq
                                d_6_openedInside_: bool
                                d_7_openedCurrent_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_5_openedGenerated_ = out1_
                                d_6_openedInside_ = out2_
                                d_7_openedCurrent_ = out3_
                                generated = d_5_openedGenerated_
                                insideConstrainedOut = d_6_openedInside_
                                currentConstrainedOut = d_7_openedCurrent_
                            elif True:
                                if VerifiedDecoderAgent.default__.Contains(d_4_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_8_openedGenerated2_: _dafny.Seq
                                    d_9_openedInside2_: bool
                                    d_10_openedCurrent2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_8_openedGenerated2_ = out4_
                                    d_9_openedInside2_ = out5_
                                    d_10_openedCurrent2_ = out6_
                                    generated = d_8_openedGenerated2_
                                    insideConstrainedOut = d_9_openedInside2_
                                    currentConstrainedOut = d_10_openedCurrent2_
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                    elif True:
                        d_11_completeNow_: bool
                        d_11_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_11_completeNow_:
                            d_12_closedGenerated_: _dafny.Seq
                            d_13_closedInside_: bool
                            d_14_closedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_12_closedGenerated_ = out7_
                            d_13_closedInside_ = out8_
                            d_14_closedCurrent_ = out9_
                            generated = d_12_closedGenerated_
                            insideConstrainedOut = d_13_closedInside_
                            currentConstrainedOut = d_14_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_15_stablePrefix_: _dafny.Seq
                            d_15_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_16_constrainedPrompt_: _dafny.Seq
                            d_16_constrainedPrompt_ = (prompt) + (d_15_stablePrefix_)
                            d_17_remaining_: int
                            d_17_remaining_ = (maxSteps) - (d_1_steps_)
                            d_18_count_: int
                            out10_: int
                            out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_18_count_ = out10_
                            if ((len(currentConstrainedOut)) >= (d_3_hardFinishLen_)) or (((len(currentConstrainedOut)) >= (d_2_preferFinishLen_)) and ((d_18_count_) <= (6))):
                                (lm).GenerateLogits((d_16_constrainedPrompt_) + (currentConstrainedOut))
                                d_19_penalize_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                d_19_penalize_ = out11_
                                (d_0_helpers_).PenalizeTokenLogits(lm, d_19_penalize_, _dafny.BigRational('8e0'))
                                d_20_nextLate_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = (d_0_helpers_).ConstrainedSample(lm, parser, currentConstrainedOut, eosToken)
                                d_20_nextLate_ = out12_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_20_nextLate_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_21_appendedGenerated_: _dafny.Seq
                                    d_22_appendedInside_: bool
                                    d_23_appendedCurrent_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_nextLate_)
                                    d_21_appendedGenerated_ = out13_
                                    d_22_appendedInside_ = out14_
                                    d_23_appendedCurrent_ = out15_
                                    generated = d_21_appendedGenerated_
                                    insideConstrainedOut = d_22_appendedInside_
                                    currentConstrainedOut = d_23_appendedCurrent_
                            elif True:
                                d_24_generated2_: _dafny.Seq
                                d_25_inside2_: bool
                                d_26_current2_: _dafny.Seq
                                d_27_hitEos_: bool
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out19_: bool
                                out16_, out17_, out18_, out19_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_16_constrainedPrompt_, generated, currentConstrainedOut, eosToken)
                                d_24_generated2_ = out16_
                                d_25_inside2_ = out17_
                                d_26_current2_ = out18_
                                d_27_hitEos_ = out19_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if d_27_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = d_24_generated2_
                                    insideConstrainedOut = d_25_inside2_
                                    currentConstrainedOut = d_26_current2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


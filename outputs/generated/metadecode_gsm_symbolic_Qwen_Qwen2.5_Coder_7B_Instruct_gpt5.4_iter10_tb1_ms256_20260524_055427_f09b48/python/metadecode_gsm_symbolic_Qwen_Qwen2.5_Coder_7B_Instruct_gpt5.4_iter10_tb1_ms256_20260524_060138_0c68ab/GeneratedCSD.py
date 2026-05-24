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
        if True:
            generated = generatedPrefix
            insideConstrainedOut = insideConstrained
            currentConstrainedOut = currentConstrained
            cost = 0
            if (maxSteps) == (0):
                cost = 0
            elif True:
                d_1_steps_: int
                d_1_steps_ = 0
                d_2_outsideSinceSpan_: int
                d_2_outsideSinceSpan_ = 0
                d_3_openThreshold_: int
                d_3_openThreshold_ = 10
                d_4_narrowThreshold_: int
                d_4_narrowThreshold_ = 8
                with _dafny.label("1_0"):
                    while (d_1_steps_) < (maxSteps):
                        with _dafny.c_label("1_0"):
                            if not(insideConstrainedOut):
                                if ((d_2_outsideSinceSpan_) >= (d_3_openThreshold_)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                    d_5_openedGenerated_: _dafny.Seq
                                    d_6_openedInside_: bool
                                    d_7_openedCurrent_: _dafny.Seq
                                    out0_: _dafny.Seq
                                    out1_: bool
                                    out2_: _dafny.Seq
                                    out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_5_openedGenerated_ = out0_
                                    d_6_openedInside_ = out1_
                                    d_7_openedCurrent_ = out2_
                                    generated = d_5_openedGenerated_
                                    insideConstrainedOut = d_6_openedInside_
                                    currentConstrainedOut = d_7_openedCurrent_
                                    d_2_outsideSinceSpan_ = 0
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_8_next_: _dafny.Seq
                                    out3_: _dafny.Seq
                                    out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                    d_8_next_ = out3_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_8_next_) == (eosToken):
                                        raise _dafny.Break("1_0")
                                    elif True:
                                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                        if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                            d_9_enteredGenerated_: _dafny.Seq
                                            d_10_enteredInside_: bool
                                            d_11_enteredCurrent_: _dafny.Seq
                                            out4_: _dafny.Seq
                                            out5_: bool
                                            out6_: _dafny.Seq
                                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                            d_9_enteredGenerated_ = out4_
                                            d_10_enteredInside_ = out5_
                                            d_11_enteredCurrent_ = out6_
                                            generated = d_9_enteredGenerated_
                                            insideConstrainedOut = d_10_enteredInside_
                                            currentConstrainedOut = d_11_enteredCurrent_
                                            d_2_outsideSinceSpan_ = 0
                                        elif True:
                                            d_2_outsideSinceSpan_ = (d_2_outsideSinceSpan_) + (1)
                            elif True:
                                if (parser).IsCompletePrefix(currentConstrainedOut):
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
                                    d_2_outsideSinceSpan_ = 0
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif ((maxSteps) - (d_1_steps_)) == (1):
                                    d_15_rolledGenerated_: _dafny.Seq
                                    d_16_rolledCurrent_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_15_rolledGenerated_ = out10_
                                    d_16_rolledCurrent_ = out11_
                                    generated = d_15_rolledGenerated_
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    raise _dafny.Break("1_0")
                                elif True:
                                    d_17_stablePrefix_: _dafny.Seq
                                    d_17_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_18_constrainedPrompt_: _dafny.Seq
                                    d_18_constrainedPrompt_ = (prompt) + (d_17_stablePrefix_)
                                    d_19_validCount_: int
                                    out12_: int
                                    out12_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                    d_19_validCount_ = out12_
                                    d_20_nextIn_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                                    if (d_19_validCount_) <= (d_4_narrowThreshold_):
                                        out13_: _dafny.Seq
                                        out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                                        d_20_nextIn_ = out13_
                                    elif True:
                                        out14_: _dafny.Seq
                                        out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                        d_20_nextIn_ = out14_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_20_nextIn_) == (eosToken):
                                        d_21_rolledGenerated2_: _dafny.Seq
                                        d_22_rolledCurrent2_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out15_, out16_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_21_rolledGenerated2_ = out15_
                                        d_22_rolledCurrent2_ = out16_
                                        generated = d_21_rolledGenerated2_
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                        raise _dafny.Break("1_0")
                                    elif True:
                                        d_23_appendedGenerated_: _dafny.Seq
                                        d_24_appendedInside_: bool
                                        d_25_appendedCurrent_: _dafny.Seq
                                        out17_: _dafny.Seq
                                        out18_: bool
                                        out19_: _dafny.Seq
                                        out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_nextIn_)
                                        d_23_appendedGenerated_ = out17_
                                        d_24_appendedInside_ = out18_
                                        d_25_appendedCurrent_ = out19_
                                        generated = d_23_appendedGenerated_
                                        insideConstrainedOut = d_24_appendedInside_
                                        currentConstrainedOut = d_25_appendedCurrent_
                            pass
                    pass
                if (d_1_steps_) == (0):
                    d_26_fallback_: _dafny.Seq
                    out20_: _dafny.Seq
                    out20_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_26_fallback_ = out20_
                    d_1_steps_ = 1
                    if (d_26_fallback_) != (eosToken):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_26_fallback_]))
                        if (d_26_fallback_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_27_enteredGenerated2_: _dafny.Seq
                            d_28_enteredInside2_: bool
                            d_29_enteredCurrent2_: _dafny.Seq
                            out21_: _dafny.Seq
                            out22_: bool
                            out23_: _dafny.Seq
                            out21_, out22_, out23_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_27_enteredGenerated2_ = out21_
                            d_28_enteredInside2_ = out22_
                            d_29_enteredCurrent2_ = out23_
                            generated = d_27_enteredGenerated2_
                            insideConstrainedOut = d_28_enteredInside2_
                            currentConstrainedOut = d_29_enteredCurrent2_
                cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


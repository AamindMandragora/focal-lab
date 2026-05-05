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
        d_2_narrowThreshold_ = 12
        d_3_preSpanToken_: _dafny.Seq
        d_3_preSpanToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        d_4_sawPreSpanToken_: bool
        d_4_sawPreSpanToken_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_6_prevTok_: _dafny.Seq
                                d_7_foundPrev_: bool
                                out1_: _dafny.Seq
                                out2_: bool
                                out1_, out2_ = (d_0_helpers_).LastTokenBefore((generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_])), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                                d_6_prevTok_ = out1_
                                d_7_foundPrev_ = out2_
                                d_3_preSpanToken_ = d_6_prevTok_
                                d_4_sawPreSpanToken_ = d_7_foundPrev_
                                d_8_enteredGenerated_: _dafny.Seq
                                d_9_enteredInside_: bool
                                d_10_enteredCurrent_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_])))
                                d_8_enteredGenerated_ = out3_
                                d_9_enteredInside_ = out4_
                                d_10_enteredCurrent_ = out5_
                                generated = d_8_enteredGenerated_
                                insideConstrainedOut = d_9_enteredInside_
                                currentConstrainedOut = d_10_enteredCurrent_
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_11_closedGenerated_: _dafny.Seq
                        d_12_closedInside_: bool
                        d_13_closedCurrent_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_11_closedGenerated_ = out6_
                        d_12_closedInside_ = out7_
                        d_13_closedCurrent_ = out8_
                        generated = d_11_closedGenerated_
                        insideConstrainedOut = d_12_closedInside_
                        currentConstrainedOut = d_13_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_14_stablePrefix_: _dafny.Seq
                        d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                        d_16_validCount_: int
                        out9_: int
                        out9_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_16_validCount_ = out9_
                        if ((d_16_validCount_) <= (d_2_narrowThreshold_)) or ((stepTokenBudget) <= (1)):
                            d_17_groups_: _dafny.Seq
                            d_17_groups_ = validTokenGroups
                            if d_4_sawPreSpanToken_:
                                d_17_groups_ = (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([d_3_preSpanToken_])])) + (validTokenGroups)
                            d_18_next_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, d_17_groups_, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                            d_18_next_ = out10_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_18_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_19_appendedGenerated_: _dafny.Seq
                                d_20_appendedInside_: bool
                                d_21_appendedCurrent_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                d_19_appendedGenerated_ = out11_
                                d_20_appendedInside_ = out12_
                                d_21_appendedCurrent_ = out13_
                                generated = d_19_appendedGenerated_
                                insideConstrainedOut = d_20_appendedInside_
                                currentConstrainedOut = d_21_appendedCurrent_
                        elif True:
                            d_22_remaining_: int
                            d_22_remaining_ = (maxSteps) - (d_1_steps_)
                            d_23_symbolBudget_: int
                            if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_22_remaining_)):
                                d_23_symbolBudget_ = d_22_remaining_
                            elif True:
                                d_23_symbolBudget_ = stepTokenBudget
                            d_24_symbolGenerated_: _dafny.Seq
                            d_25_symbolOut_: _dafny.Seq
                            d_26_hitEos_: bool
                            d_27_stepsUsed_: int
                            out14_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: int
                            out14_, out15_, out16_, out17_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_15_constrainedPrompt_, generated, currentConstrainedOut, d_23_symbolBudget_, eosToken)
                            d_24_symbolGenerated_ = out14_
                            d_25_symbolOut_ = out15_
                            d_26_hitEos_ = out16_
                            d_27_stepsUsed_ = out17_
                            generated = d_24_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_25_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_27_stepsUsed_)
                            if d_26_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Every arithmetic computation must appear inside visible << >> delimiters, and the final computation should also be inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_outsideSinceSpan_: int
        d_2_outsideSinceSpan_ = 0
        d_3_openAfter_: int
        d_3_openAfter_ = 12
        d_4_narrowThreshold_: int
        d_4_narrowThreshold_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_outsideSinceSpan_) >= (d_3_openAfter_):
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
                            d_8_next_ = eosToken
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).SafePenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('12e0'))
                            out3_: _dafny.Seq
                            out3_ = (lm).ChooseNextTokenUnconstrained()
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
                                d_2_outsideSinceSpan_ = (d_2_outsideSinceSpan_) + (1)
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                d_2_outsideSinceSpan_ = (d_2_outsideSinceSpan_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_9_closedGenerated_: _dafny.Seq
                        d_10_closedInside_: bool
                        d_11_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_9_closedGenerated_ = out4_
                        d_10_closedInside_ = out5_
                        d_11_closedCurrent_ = out6_
                        generated = d_9_closedGenerated_
                        insideConstrainedOut = d_10_closedInside_
                        currentConstrainedOut = d_11_closedCurrent_
                        d_2_outsideSinceSpan_ = 0
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_12_stablePrefix_: _dafny.Seq
                        d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                        d_14_validCount_: int
                        out7_: int
                        out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_14_validCount_ = out7_
                        if (d_14_validCount_) <= (d_4_narrowThreshold_):
                            d_15_nextTok_: _dafny.Seq
                            d_16_wasConstrained_: bool
                            out8_: _dafny.Seq
                            out9_: bool
                            out8_, out9_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_15_nextTok_ = out8_
                            d_16_wasConstrained_ = out9_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_15_nextTok_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_17_appendedGenerated_: _dafny.Seq
                                d_18_appendedInside_: bool
                                d_19_appendedCurrent_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_nextTok_)
                                d_17_appendedGenerated_ = out10_
                                d_18_appendedInside_ = out11_
                                d_19_appendedCurrent_ = out12_
                                generated = d_17_appendedGenerated_
                                insideConstrainedOut = d_18_appendedInside_
                                currentConstrainedOut = d_19_appendedCurrent_
                                d_2_outsideSinceSpan_ = 0
                        elif True:
                            d_20_remainingInside_: int
                            d_20_remainingInside_ = (maxSteps) - (d_1_steps_)
                            d_21_symbolBudget_: int
                            if (stepTokenBudget) == (0):
                                d_21_symbolBudget_ = 1
                            elif True:
                                d_21_symbolBudget_ = stepTokenBudget
                            if (d_21_symbolBudget_) > (d_20_remainingInside_):
                                d_21_symbolBudget_ = d_20_remainingInside_
                            if (d_21_symbolBudget_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_22_symbolGenerated_: _dafny.Seq
                                d_23_symbolCurrent_: _dafny.Seq
                                d_24_hitEos_: bool
                                d_25_stepsUsed_: int
                                out13_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: int
                                out13_, out14_, out15_, out16_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_13_constrainedPrompt_, generated, currentConstrainedOut, d_21_symbolBudget_, eosToken)
                                d_22_symbolGenerated_ = out13_
                                d_23_symbolCurrent_ = out14_
                                d_24_hitEos_ = out15_
                                d_25_stepsUsed_ = out16_
                                generated = d_22_symbolGenerated_
                                currentConstrainedOut = d_23_symbolCurrent_
                                insideConstrainedOut = True
                                d_2_outsideSinceSpan_ = 0
                                d_1_steps_ = (d_1_steps_) + (d_25_stepsUsed_)
                                if d_24_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


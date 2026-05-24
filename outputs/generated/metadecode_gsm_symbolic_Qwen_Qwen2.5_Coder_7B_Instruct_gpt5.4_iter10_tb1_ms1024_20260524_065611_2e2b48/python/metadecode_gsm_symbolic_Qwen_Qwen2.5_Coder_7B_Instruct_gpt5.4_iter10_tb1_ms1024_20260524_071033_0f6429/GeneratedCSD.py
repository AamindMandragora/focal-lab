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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Write every arithmetic computation inside visible << >> delimiters, and ensure the final computation is also inside << >>.")))
        (d_0_helpers_).SetNonDeterministic(lm, False)
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_outsideSinceSpan_: int
        d_2_outsideSinceSpan_ = 0
        d_3_openAfter_: int
        d_3_openAfter_ = 10
        d_4_arithmeticCueSeen_: bool
        d_4_arithmeticCueSeen_ = False
        d_5_narrowThreshold_: int
        d_5_narrowThreshold_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                        d_6_closedGenerated_: _dafny.Seq
                        d_7_closedInside_: bool
                        d_8_closedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_6_closedGenerated_ = out0_
                        d_7_closedInside_ = out1_
                        d_8_closedCurrent_ = out2_
                        generated = d_6_closedGenerated_
                        insideConstrainedOut = d_7_closedInside_
                        currentConstrainedOut = d_8_closedCurrent_
                        d_2_outsideSinceSpan_ = 0
                        d_4_arithmeticCueSeen_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif not(insideConstrainedOut):
                        d_9_shouldOpen_: bool
                        d_9_shouldOpen_ = False
                        if d_4_arithmeticCueSeen_:
                            d_9_shouldOpen_ = True
                        elif (d_2_outsideSinceSpan_) >= (d_3_openAfter_):
                            d_9_shouldOpen_ = True
                        if d_9_shouldOpen_:
                            d_10_openedGenerated_: _dafny.Seq
                            d_11_openedInside_: bool
                            d_12_openedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_openedGenerated_ = out3_
                            d_11_openedInside_ = out4_
                            d_12_openedCurrent_ = out5_
                            generated = d_10_openedGenerated_
                            insideConstrainedOut = d_11_openedInside_
                            currentConstrainedOut = d_12_openedCurrent_
                            d_2_outsideSinceSpan_ = 0
                            d_4_arithmeticCueSeen_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).MaskTokensInPrefix(lm, _dafny.SeqWithoutIsStrInference([]))
                            (d_0_helpers_).SafePenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                            d_13_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_13_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_13_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                                d_2_outsideSinceSpan_ = (d_2_outsideSinceSpan_) + (1)
                                if (((((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")))):
                                    d_4_arithmeticCueSeen_ = True
                    elif True:
                        d_14_stablePrefix_: _dafny.Seq
                        d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                        d_16_remainingInside_: int
                        d_16_remainingInside_ = (maxSteps) - (d_1_steps_)
                        d_17_validCount_: int
                        out7_: int
                        out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_17_validCount_ = out7_
                        if (d_17_validCount_) <= (d_5_narrowThreshold_):
                            d_18_nextConstrained_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_5_narrowThreshold_, eosToken)
                            d_18_nextConstrained_ = out8_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_18_nextConstrained_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_19_appendedGenerated_: _dafny.Seq
                                d_20_appendedInside_: bool
                                d_21_appendedCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_nextConstrained_)
                                d_19_appendedGenerated_ = out9_
                                d_20_appendedInside_ = out10_
                                d_21_appendedCurrent_ = out11_
                                generated = d_19_appendedGenerated_
                                insideConstrainedOut = d_20_appendedInside_
                                currentConstrainedOut = d_21_appendedCurrent_
                                d_2_outsideSinceSpan_ = 0
                        elif True:
                            d_22_symbolBudget_: int
                            if (stepTokenBudget) == (0):
                                d_22_symbolBudget_ = 1
                            elif True:
                                d_22_symbolBudget_ = stepTokenBudget
                            if (d_22_symbolBudget_) > (d_16_remainingInside_):
                                d_22_symbolBudget_ = d_16_remainingInside_
                            if (d_22_symbolBudget_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_23_symbolGenerated_: _dafny.Seq
                                d_24_symbolCurrent_: _dafny.Seq
                                d_25_hitEos_: bool
                                d_26_stepsUsed_: int
                                out12_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: int
                                out12_, out13_, out14_, out15_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_15_constrainedPrompt_, generated, currentConstrainedOut, d_22_symbolBudget_, eosToken)
                                d_23_symbolGenerated_ = out12_
                                d_24_symbolCurrent_ = out13_
                                d_25_hitEos_ = out14_
                                d_26_stepsUsed_ = out15_
                                generated = d_23_symbolGenerated_
                                currentConstrainedOut = d_24_symbolCurrent_
                                insideConstrainedOut = True
                                d_2_outsideSinceSpan_ = 0
                                d_1_steps_ = (d_1_steps_) + (d_26_stepsUsed_)
                                if d_25_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


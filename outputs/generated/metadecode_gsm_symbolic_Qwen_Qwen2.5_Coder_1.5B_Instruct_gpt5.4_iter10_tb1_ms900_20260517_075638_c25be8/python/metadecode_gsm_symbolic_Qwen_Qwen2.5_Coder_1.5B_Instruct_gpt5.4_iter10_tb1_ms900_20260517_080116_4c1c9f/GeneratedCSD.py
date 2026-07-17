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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Put every arithmetic computation inside visible << and >> delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_triggerOpen_: bool
        d_2_triggerOpen_ = False
        d_3_rollbackLimit_: int
        d_3_rollbackLimit_ = 32
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_triggerOpen_:
                            d_4_openedGenerated_: _dafny.Seq
                            d_5_openedInside_: bool
                            d_6_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openedGenerated_ = out0_
                            d_5_openedInside_ = out1_
                            d_6_openedCurrent_ = out2_
                            generated = d_4_openedGenerated_
                            insideConstrainedOut = d_5_openedInside_
                            currentConstrainedOut = d_6_openedCurrent_
                            d_2_triggerOpen_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_8_observedGenerated_: _dafny.Seq
                                    d_9_observedInside_: bool
                                    d_10_observedCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_8_observedGenerated_ = out4_
                                    d_9_observedInside_ = out5_
                                    d_10_observedCurrent_ = out6_
                                    generated = d_8_observedGenerated_
                                    insideConstrainedOut = d_9_observedInside_
                                    currentConstrainedOut = d_10_observedCurrent_
                                    d_2_triggerOpen_ = False
                                elif (((((d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))) or ((d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+"))))) or ((d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-"))))) or ((d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*"))))) or ((d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")))):
                                    d_2_triggerOpen_ = True
                                elif True:
                                    d_2_triggerOpen_ = False
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
                        d_2_triggerOpen_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_3_rollbackLimit_):
                        d_14_rolledGenerated_: _dafny.Seq
                        d_15_rolledCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_14_rolledGenerated_ = out10_
                        d_15_rolledCurrent_ = out11_
                        generated = d_14_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_15_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_16_stablePrefix_: _dafny.Seq
                        d_16_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (d_16_stablePrefix_)
                        d_18_remaining_: int
                        d_18_remaining_ = (maxSteps) - (d_1_steps_)
                        d_19_validCount_: int
                        out12_: int
                        out12_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_19_validCount_ = out12_
                        if (((d_19_validCount_) <= (8)) or ((stepTokenBudget) <= (1))) or ((d_18_remaining_) <= (1)):
                            d_20_nextIn_: _dafny.Seq
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_20_nextIn_ = out13_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_20_nextIn_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_21_appendedGenerated_: _dafny.Seq
                                d_22_appendedInside_: bool
                                d_23_appendedCurrent_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_nextIn_)
                                d_21_appendedGenerated_ = out14_
                                d_22_appendedInside_ = out15_
                                d_23_appendedCurrent_ = out16_
                                generated = d_21_appendedGenerated_
                                insideConstrainedOut = d_22_appendedInside_
                                currentConstrainedOut = d_23_appendedCurrent_
                        elif True:
                            d_24_symbolBudget_: int
                            if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_18_remaining_)):
                                d_24_symbolBudget_ = d_18_remaining_
                            elif True:
                                d_24_symbolBudget_ = stepTokenBudget
                            d_25_symbolGenerated_: _dafny.Seq
                            d_26_symbolCurrent_: _dafny.Seq
                            d_27_hitEos_: bool
                            d_28_stepsUsed_: int
                            out17_: _dafny.Seq
                            out18_: _dafny.Seq
                            out19_: bool
                            out20_: int
                            out17_, out18_, out19_, out20_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_17_constrainedPrompt_, generated, currentConstrainedOut, d_24_symbolBudget_, eosToken)
                            d_25_symbolGenerated_ = out17_
                            d_26_symbolCurrent_ = out18_
                            d_27_hitEos_ = out19_
                            d_28_stepsUsed_ = out20_
                            generated = d_25_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_26_symbolCurrent_
                            d_1_steps_ = (d_1_steps_) + (d_28_stepsUsed_)
                            if d_27_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        if ((insideConstrainedOut) and ((d_1_steps_) < (maxSteps))) and ((parser).IsCompletePrefix(currentConstrainedOut)):
            d_29_finalGenerated_: _dafny.Seq
            d_30_finalInside_: bool
            d_31_finalCurrent_: _dafny.Seq
            out21_: _dafny.Seq
            out22_: bool
            out23_: _dafny.Seq
            out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_29_finalGenerated_ = out21_
            d_30_finalInside_ = out22_
            d_31_finalCurrent_ = out23_
            generated = d_29_finalGenerated_
            insideConstrainedOut = d_30_finalInside_
            currentConstrainedOut = d_31_finalCurrent_
            d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


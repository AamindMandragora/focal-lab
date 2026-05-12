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
        d_2_openedFallback_: bool
        d_2_openedFallback_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_2_openedFallback_):
                            d_3_remainingOutside_: int
                            d_3_remainingOutside_ = (maxSteps) - (d_1_steps_)
                            d_4_chunkBudget_: int
                            if (d_3_remainingOutside_) > (3):
                                d_4_chunkBudget_ = 3
                            elif True:
                                d_4_chunkBudget_ = d_3_remainingOutside_
                            d_5_chunkedGenerated_: _dafny.Seq
                            d_6_stoppedOnOpenSpan_: bool
                            d_7_stoppedOnEos_: bool
                            d_8_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_5_chunkedGenerated_ = out0_
                            d_6_stoppedOnOpenSpan_ = out1_
                            d_7_stoppedOnEos_ = out2_
                            d_8_stepsUsed_ = out3_
                            generated = d_5_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                            if d_7_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_6_stoppedOnOpenSpan_:
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
                            elif (d_1_steps_) < (maxSteps):
                                d_12_openedGenerated_: _dafny.Seq
                                d_13_openedInside_: bool
                                d_14_openedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_12_openedGenerated_ = out7_
                                d_13_openedInside_ = out8_
                                d_14_openedCurrent_ = out9_
                                generated = d_12_openedGenerated_
                                insideConstrainedOut = d_13_openedInside_
                                currentConstrainedOut = d_14_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_openedFallback_ = True
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_15_openedGenerated2_: _dafny.Seq
                            d_16_openedInside2_: bool
                            d_17_openedCurrent2_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_15_openedGenerated2_ = out10_
                            d_16_openedInside2_ = out11_
                            d_17_openedCurrent2_ = out12_
                            generated = d_15_openedGenerated2_
                            insideConstrainedOut = d_16_openedInside2_
                            currentConstrainedOut = d_17_openedCurrent2_
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_18_closedGenerated_: _dafny.Seq
                        d_19_closedInside_: bool
                        d_20_closedCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_18_closedGenerated_ = out13_
                        d_19_closedInside_ = out14_
                        d_20_closedCurrent_ = out15_
                        generated = d_18_closedGenerated_
                        insideConstrainedOut = d_19_closedInside_
                        currentConstrainedOut = d_20_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_21_stablePrefix_: _dafny.Seq
                        d_21_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_22_constrainedPrompt_: _dafny.Seq
                        d_22_constrainedPrompt_ = (prompt) + (d_21_stablePrefix_)
                        d_23_validCount_: int
                        out16_: int
                        out16_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_23_validCount_ = out16_
                        d_24_deadEndSoon_: bool
                        out17_: bool
                        out17_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                        d_24_deadEndSoon_ = out17_
                        d_25_remaining_: int
                        d_25_remaining_ = (maxSteps) - (d_1_steps_)
                        if ((((stepTokenBudget) > (1)) and ((d_25_remaining_) > (0))) and ((d_23_validCount_) > (10))) and (not(d_24_deadEndSoon_)):
                            d_26_symbolBudget_: int
                            if (stepTokenBudget) > (d_25_remaining_):
                                d_26_symbolBudget_ = d_25_remaining_
                            elif True:
                                d_26_symbolBudget_ = stepTokenBudget
                            d_27_symbolGenerated_: _dafny.Seq
                            d_28_symbolOut_: _dafny.Seq
                            d_29_hitEos_: bool
                            d_30_stepsUsed2_: int
                            out18_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: int
                            out18_, out19_, out20_, out21_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_22_constrainedPrompt_, generated, currentConstrainedOut, d_26_symbolBudget_, eosToken)
                            d_27_symbolGenerated_ = out18_
                            d_28_symbolOut_ = out19_
                            d_29_hitEos_ = out20_
                            d_30_stepsUsed2_ = out21_
                            generated = d_27_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_28_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_30_stepsUsed2_)
                            if d_29_hitEos_:
                                raise _dafny.Break("0")
                        elif (d_24_deadEndSoon_) or ((d_23_validCount_) <= (4)):
                            d_31_nextHard_: _dafny.Seq
                            out22_: _dafny.Seq
                            out22_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_31_nextHard_ = out22_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_31_nextHard_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_32_appendedGenerated1_: _dafny.Seq
                                d_33_appendedInside1_: bool
                                d_34_appendedCurrent1_: _dafny.Seq
                                out23_: _dafny.Seq
                                out24_: bool
                                out25_: _dafny.Seq
                                out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_31_nextHard_)
                                d_32_appendedGenerated1_ = out23_
                                d_33_appendedInside1_ = out24_
                                d_34_appendedCurrent1_ = out25_
                                generated = d_32_appendedGenerated1_
                                insideConstrainedOut = d_33_appendedInside1_
                                currentConstrainedOut = d_34_appendedCurrent1_
                        elif True:
                            d_35_nextSoft_: _dafny.Seq
                            d_36_wasConstrained_: bool
                            out26_: _dafny.Seq
                            out27_: bool
                            out26_, out27_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_35_nextSoft_ = out26_
                            d_36_wasConstrained_ = out27_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_35_nextSoft_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_37_appendedGenerated2_: _dafny.Seq
                                d_38_appendedInside2_: bool
                                d_39_appendedCurrent2_: _dafny.Seq
                                out28_: _dafny.Seq
                                out29_: bool
                                out30_: _dafny.Seq
                                out28_, out29_, out30_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_35_nextSoft_)
                                d_37_appendedGenerated2_ = out28_
                                d_38_appendedInside2_ = out29_
                                d_39_appendedCurrent2_ = out30_
                                generated = d_37_appendedGenerated2_
                                insideConstrainedOut = d_38_appendedInside2_
                                currentConstrainedOut = d_39_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


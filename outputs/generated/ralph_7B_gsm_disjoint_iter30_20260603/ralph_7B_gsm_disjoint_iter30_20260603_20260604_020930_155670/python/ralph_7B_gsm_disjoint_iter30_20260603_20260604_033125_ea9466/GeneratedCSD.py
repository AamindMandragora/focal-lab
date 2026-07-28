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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve math word problems step by step. For each calculation, write the expression inside << >> delimiters, for example <<3 + 4 = 7>>. Write the final answer as #### <<number>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeTokens_: int
        d_2_freeTokens_ = 0
        d_3_forceOpenThreshold_: int
        d_3_forceOpenThreshold_ = 15
        d_4_maxConstrainedTokens_: int
        d_4_maxConstrainedTokens_ = 40
        d_5_constrainedTokenCount_: int
        d_5_constrainedTokenCount_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_2_freeTokens_) >= (d_3_forceOpenThreshold_)) and (((maxSteps) - (d_1_steps_)) >= (2)):
                            d_6_openedGenerated_: _dafny.Seq
                            d_7_openedInside_: bool
                            d_8_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openedGenerated_ = out0_
                            d_7_openedInside_ = out1_
                            d_8_openedCurrent_ = out2_
                            generated = d_6_openedGenerated_
                            insideConstrainedOut = d_7_openedInside_
                            currentConstrainedOut = d_8_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_freeTokens_ = 0
                            d_5_constrainedTokenCount_ = 0
                        elif True:
                            d_9_chunkBudget_: int
                            d_9_chunkBudget_ = (maxSteps) - (d_1_steps_)
                            if (d_9_chunkBudget_) > (8):
                                d_9_chunkBudget_ = 8
                            if (d_9_chunkBudget_) == (0):
                                d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            d_10_chunkGenerated_: _dafny.Seq
                            d_11_stoppedOnOpenSpan_: bool
                            d_12_stoppedOnEos_: bool
                            d_13_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_10_chunkGenerated_ = out3_
                            d_11_stoppedOnOpenSpan_ = out4_
                            d_12_stoppedOnEos_ = out5_
                            d_13_stepsUsed_ = out6_
                            d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                            d_2_freeTokens_ = (d_2_freeTokens_) + (d_13_stepsUsed_)
                            generated = d_10_chunkGenerated_
                            if d_12_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_11_stoppedOnOpenSpan_:
                                d_14_enteredGenerated_: _dafny.Seq
                                d_15_enteredInside_: bool
                                d_16_enteredCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_14_enteredGenerated_ = out7_
                                d_15_enteredInside_ = out8_
                                d_16_enteredCurrent_ = out9_
                                generated = d_14_enteredGenerated_
                                insideConstrainedOut = d_15_enteredInside_
                                currentConstrainedOut = d_16_enteredCurrent_
                                d_2_freeTokens_ = 0
                                d_5_constrainedTokenCount_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_17_closedGenerated_: _dafny.Seq
                        d_18_closedInside_: bool
                        d_19_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_closedGenerated_ = out10_
                        d_18_closedInside_ = out11_
                        d_19_closedCurrent_ = out12_
                        generated = d_17_closedGenerated_
                        insideConstrainedOut = d_18_closedInside_
                        currentConstrainedOut = d_19_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_freeTokens_ = 0
                        d_5_constrainedTokenCount_ = 0
                    elif (d_5_constrainedTokenCount_) >= (d_4_maxConstrainedTokens_):
                        d_20_rolledGenerated_: _dafny.Seq
                        d_21_rolledCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_20_rolledGenerated_ = out13_
                        d_21_rolledCurrent_ = out14_
                        generated = d_20_rolledGenerated_
                        currentConstrainedOut = d_21_rolledCurrent_
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_22_closedGenerated2_: _dafny.Seq
                            d_23_closedInside2_: bool
                            d_24_closedCurrent2_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_22_closedGenerated2_ = out15_
                            d_23_closedInside2_ = out16_
                            d_24_closedCurrent2_ = out17_
                            generated = d_22_closedGenerated2_
                            insideConstrainedOut = d_23_closedInside2_
                            currentConstrainedOut = d_24_closedCurrent2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_freeTokens_ = 0
                            d_5_constrainedTokenCount_ = 0
                        elif True:
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                    elif True:
                        d_25_isDeadEnd_: bool
                        out18_: bool
                        out18_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_25_isDeadEnd_ = out18_
                        if d_25_isDeadEnd_:
                            d_26_rolledGenerated_: _dafny.Seq
                            d_27_rolledCurrent_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: _dafny.Seq
                            out19_, out20_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_26_rolledGenerated_ = out19_
                            d_27_rolledCurrent_ = out20_
                            generated = d_26_rolledGenerated_
                            currentConstrainedOut = d_27_rolledCurrent_
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_28_closedGenerated3_: _dafny.Seq
                                d_29_closedInside3_: bool
                                d_30_closedCurrent3_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: bool
                                out23_: _dafny.Seq
                                out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_28_closedGenerated3_ = out21_
                                d_29_closedInside3_ = out22_
                                d_30_closedCurrent3_ = out23_
                                generated = d_28_closedGenerated3_
                                insideConstrainedOut = d_29_closedInside3_
                                currentConstrainedOut = d_30_closedCurrent3_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_freeTokens_ = 0
                                d_5_constrainedTokenCount_ = 0
                            elif True:
                                d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                        elif True:
                            d_31_constrainedPrompt_: _dafny.Seq
                            d_31_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_32_next_: _dafny.Seq
                            out24_: _dafny.Seq
                            out24_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_31_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_32_next_ = out24_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_32_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_33_appendedGenerated_: _dafny.Seq
                                d_34_appendedInside_: bool
                                d_35_appendedCurrent_: _dafny.Seq
                                out25_: _dafny.Seq
                                out26_: bool
                                out27_: _dafny.Seq
                                out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_32_next_)
                                d_33_appendedGenerated_ = out25_
                                d_34_appendedInside_ = out26_
                                d_35_appendedCurrent_ = out27_
                                generated = d_33_appendedGenerated_
                                insideConstrainedOut = d_34_appendedInside_
                                currentConstrainedOut = d_35_appendedCurrent_
                                d_5_constrainedTokenCount_ = (d_5_constrainedTokenCount_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


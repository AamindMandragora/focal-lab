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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Every arithmetic computation must appear inside visible << and >> delimiters. Keep each computation short, open << right after an arithmetic cue such as = or an operator, and close >> immediately when the computation is complete.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openArmed_: bool
        d_2_openArmed_ = False
        d_3_droughtLimit_: int
        d_3_droughtLimit_ = 80
        d_4_rollbackLimit_: int
        d_4_rollbackLimit_ = 24
        d_5_chunkCap_: int
        d_5_chunkCap_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_sinceOpen_: int
                        out0_: int
                        out0_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        d_6_sinceOpen_ = out0_
                        d_7_droughtOpen_: bool
                        d_7_droughtOpen_ = (d_3_droughtLimit_) <= (d_6_sinceOpen_)
                        d_8_forceOpen_: bool
                        d_8_forceOpen_ = d_2_openArmed_
                        if d_7_droughtOpen_:
                            d_8_forceOpen_ = True
                        if d_8_forceOpen_:
                            d_9_openedGenerated_: _dafny.Seq
                            d_10_openedInside_: bool
                            d_11_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_9_openedGenerated_ = out1_
                            d_10_openedInside_ = out2_
                            d_11_openedCurrent_ = out3_
                            generated = d_9_openedGenerated_
                            insideConstrainedOut = d_10_openedInside_
                            currentConstrainedOut = d_11_openedCurrent_
                            d_2_openArmed_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_12_remainingChunk_: int
                            d_12_remainingChunk_ = (maxSteps) - (d_1_steps_)
                            d_13_chunkBudget_: int
                            if (d_5_chunkCap_) < (d_12_remainingChunk_):
                                d_13_chunkBudget_ = d_5_chunkCap_
                            elif True:
                                d_13_chunkBudget_ = d_12_remainingChunk_
                            d_14_chunkedGenerated_: _dafny.Seq
                            d_15_stoppedOpen_: bool
                            d_16_stoppedEos_: bool
                            d_17_stepsUsed_: int
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: bool
                            out7_: int
                            out4_, out5_, out6_, out7_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_13_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_14_chunkedGenerated_ = out4_
                            d_15_stoppedOpen_ = out5_
                            d_16_stoppedEos_ = out6_
                            d_17_stepsUsed_ = out7_
                            generated = d_14_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_17_stepsUsed_)
                            if d_16_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_15_stoppedOpen_:
                                d_18_enteredGenerated_: _dafny.Seq
                                d_19_enteredInside_: bool
                                d_20_enteredCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_18_enteredGenerated_ = out8_
                                d_19_enteredInside_ = out9_
                                d_20_enteredCurrent_ = out10_
                                generated = d_18_enteredGenerated_
                                insideConstrainedOut = d_19_enteredInside_
                                currentConstrainedOut = d_20_enteredCurrent_
                                d_2_openArmed_ = False
                            elif True:
                                d_21_prevEq_: _dafny.Seq
                                d_22_foundEq_: bool
                                out11_: _dafny.Seq
                                out12_: bool
                                out11_, out12_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_21_prevEq_ = out11_
                                d_22_foundEq_ = out12_
                                d_23_prevPlus_: _dafny.Seq
                                d_24_foundPlus_: bool
                                out13_: _dafny.Seq
                                out14_: bool
                                out13_, out14_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")))
                                d_23_prevPlus_ = out13_
                                d_24_foundPlus_ = out14_
                                d_25_prevMinus_: _dafny.Seq
                                d_26_foundMinus_: bool
                                out15_: _dafny.Seq
                                out16_: bool
                                out15_, out16_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")))
                                d_25_prevMinus_ = out15_
                                d_26_foundMinus_ = out16_
                                d_27_prevTimes_: _dafny.Seq
                                d_28_foundTimes_: bool
                                out17_: _dafny.Seq
                                out18_: bool
                                out17_, out18_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")))
                                d_27_prevTimes_ = out17_
                                d_28_foundTimes_ = out18_
                                d_29_prevDiv_: _dafny.Seq
                                d_30_foundDiv_: bool
                                out19_: _dafny.Seq
                                out20_: bool
                                out19_, out20_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")))
                                d_29_prevDiv_ = out19_
                                d_30_foundDiv_ = out20_
                                d_31_prevColon_: _dafny.Seq
                                d_32_foundColon_: bool
                                out21_: _dafny.Seq
                                out22_: bool
                                out21_, out22_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))
                                d_31_prevColon_ = out21_
                                d_32_foundColon_ = out22_
                                d_2_openArmed_ = (((((d_22_foundEq_) or (d_24_foundPlus_)) or (d_26_foundMinus_)) or (d_28_foundTimes_)) or (d_30_foundDiv_)) or (d_32_foundColon_)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_33_closedGenerated_: _dafny.Seq
                        d_34_closedInside_: bool
                        d_35_closedCurrent_: _dafny.Seq
                        out23_: _dafny.Seq
                        out24_: bool
                        out25_: _dafny.Seq
                        out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_33_closedGenerated_ = out23_
                        d_34_closedInside_ = out24_
                        d_35_closedCurrent_ = out25_
                        generated = d_33_closedGenerated_
                        insideConstrainedOut = d_34_closedInside_
                        currentConstrainedOut = d_35_closedCurrent_
                        d_2_openArmed_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_4_rollbackLimit_):
                        d_36_rolledGenerated_: _dafny.Seq
                        d_37_rolledCurrent_: _dafny.Seq
                        out26_: _dafny.Seq
                        out27_: _dafny.Seq
                        out26_, out27_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_36_rolledGenerated_ = out26_
                        d_37_rolledCurrent_ = out27_
                        generated = d_36_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_37_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_38_constrainedPrompt_: _dafny.Seq
                        d_38_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_39_next_: _dafny.Seq
                        out28_: _dafny.Seq
                        out28_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_38_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_39_next_ = out28_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_39_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_40_appendedGenerated_: _dafny.Seq
                            d_41_appendedInside_: bool
                            d_42_appendedCurrent_: _dafny.Seq
                            out29_: _dafny.Seq
                            out30_: bool
                            out31_: _dafny.Seq
                            out29_, out30_, out31_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_39_next_)
                            d_40_appendedGenerated_ = out29_
                            d_41_appendedInside_ = out30_
                            d_42_appendedCurrent_ = out31_
                            generated = d_40_appendedGenerated_
                            insideConstrainedOut = d_41_appendedInside_
                            currentConstrainedOut = d_42_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


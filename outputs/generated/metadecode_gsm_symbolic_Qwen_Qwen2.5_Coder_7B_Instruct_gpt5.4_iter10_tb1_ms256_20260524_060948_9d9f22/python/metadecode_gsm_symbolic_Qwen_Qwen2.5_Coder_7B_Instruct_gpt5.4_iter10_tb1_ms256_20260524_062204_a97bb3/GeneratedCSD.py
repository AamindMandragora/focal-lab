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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Every arithmetic computation must appear inside << >> delimiters, and the final computation should also be inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_completedSpans_: int
        d_2_completedSpans_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remaining_: int
                        d_3_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_3_remaining_) >= (2):
                            d_4_chunkBudget_: int
                            if (d_3_remaining_) <= (3):
                                d_4_chunkBudget_ = d_3_remaining_
                            elif True:
                                d_4_chunkBudget_ = 3
                            d_5_chunkedG_: _dafny.Seq
                            d_6_stoppedOpen_: bool
                            d_7_stoppedEos_: bool
                            d_8_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_5_chunkedG_ = out0_
                            d_6_stoppedOpen_ = out1_
                            d_7_stoppedEos_ = out2_
                            d_8_stepsUsed_ = out3_
                            generated = d_5_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                            if d_7_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_6_stoppedOpen_:
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
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).SafePenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('12e0'))
                            d_12_nextOutside_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (lm).ChooseNextTokenUnconstrained()
                            d_12_nextOutside_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_12_nextOutside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_nextOutside_]))
                                if (d_12_nextOutside_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_13_enteredGenerated2_: _dafny.Seq
                                    d_14_enteredInside2_: bool
                                    d_15_enteredCurrent2_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_13_enteredGenerated2_ = out8_
                                    d_14_enteredInside2_ = out9_
                                    d_15_enteredCurrent2_ = out10_
                                    generated = d_13_enteredGenerated2_
                                    insideConstrainedOut = d_14_enteredInside2_
                                    currentConstrainedOut = d_15_enteredCurrent2_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_closedGenerated_: _dafny.Seq
                        d_17_closedInside_: bool
                        d_18_closedCurrent_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_closedGenerated_ = out11_
                        d_17_closedInside_ = out12_
                        d_18_closedCurrent_ = out13_
                        generated = d_16_closedGenerated_
                        insideConstrainedOut = d_17_closedInside_
                        currentConstrainedOut = d_18_closedCurrent_
                        d_2_completedSpans_ = (d_2_completedSpans_) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_19_stablePrefix_: _dafny.Seq
                        d_19_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (d_19_stablePrefix_)
                        d_21_validCount_: int
                        out14_: int
                        out14_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_21_validCount_ = out14_
                        if (d_21_validCount_) <= (8):
                            d_22_nextInside_: _dafny.Seq
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_22_nextInside_ = out15_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_nextInside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_23_appendedGenerated_: _dafny.Seq
                                d_24_appendedInside_: bool
                                d_25_appendedCurrent_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_nextInside_)
                                d_23_appendedGenerated_ = out16_
                                d_24_appendedInside_ = out17_
                                d_25_appendedCurrent_ = out18_
                                generated = d_23_appendedGenerated_
                                insideConstrainedOut = d_24_appendedInside_
                                currentConstrainedOut = d_25_appendedCurrent_
                        elif True:
                            d_26_remainingInside_: int
                            d_26_remainingInside_ = (maxSteps) - (d_1_steps_)
                            d_27_symbolBudget_: int
                            if (stepTokenBudget) == (0):
                                d_27_symbolBudget_ = 1
                            elif (stepTokenBudget) > (d_26_remainingInside_):
                                d_27_symbolBudget_ = d_26_remainingInside_
                            elif True:
                                d_27_symbolBudget_ = stepTokenBudget
                            d_28_symbolGenerated_: _dafny.Seq
                            d_29_symbolOut_: _dafny.Seq
                            d_30_hitEos_: bool
                            d_31_stepsUsed2_: int
                            out19_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: int
                            out19_, out20_, out21_, out22_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_20_constrainedPrompt_, generated, currentConstrainedOut, d_27_symbolBudget_, eosToken)
                            d_28_symbolGenerated_ = out19_
                            d_29_symbolOut_ = out20_
                            d_30_hitEos_ = out21_
                            d_31_stepsUsed2_ = out22_
                            generated = d_28_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_29_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_31_stepsUsed2_)
                            if d_30_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


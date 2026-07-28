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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. After the reasoning, put only the final numeric answer inside exactly one << >> span.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_answerSpanOpened_: bool
        d_2_answerSpanOpened_ = insideConstrained
        d_3_reasoningLimit_: int
        d_3_reasoningLimit_ = 160
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_1_steps_)
                        d_5_shouldOpen_: bool
                        d_5_shouldOpen_ = False
                        if not(d_2_answerSpanOpened_):
                            if (d_4_remaining_) <= (3):
                                d_5_shouldOpen_ = True
                            elif (len(generated)) >= ((len(generatedPrefix)) + (d_3_reasoningLimit_)):
                                d_5_shouldOpen_ = True
                        if d_5_shouldOpen_:
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
                            d_2_answerSpanOpened_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                if (not(d_2_answerSpanOpened_)) and ((d_1_steps_) < (maxSteps)):
                                    d_10_openedGenerated2_: _dafny.Seq
                                    d_11_openedInside2_: bool
                                    d_12_openedCurrent2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_10_openedGenerated2_ = out4_
                                    d_11_openedInside2_ = out5_
                                    d_12_openedCurrent2_ = out6_
                                    generated = d_10_openedGenerated2_
                                    insideConstrainedOut = d_11_openedInside2_
                                    currentConstrainedOut = d_12_openedCurrent2_
                                    d_2_answerSpanOpened_ = True
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_closedGenerated_: _dafny.Seq
                        d_14_closedInside_: bool
                        d_15_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_closedGenerated_ = out7_
                        d_14_closedInside_ = out8_
                        d_15_closedCurrent_ = out9_
                        generated = d_13_closedGenerated_
                        insideConstrainedOut = d_14_closedInside_
                        currentConstrainedOut = d_15_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_16_stablePrefix_: _dafny.Seq
                        d_16_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (d_16_stablePrefix_)
                        d_18_remainingInside_: int
                        d_18_remainingInside_ = (maxSteps) - (d_1_steps_)
                        d_19_validCount_: int
                        out10_: int
                        out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_19_validCount_ = out10_
                        if (((d_19_validCount_) <= (10)) or ((stepTokenBudget) <= (1))) or ((d_18_remainingInside_) <= (1)):
                            d_20_nextIn_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_20_nextIn_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_20_nextIn_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_21_appendedGenerated_: _dafny.Seq
                                d_22_appendedInside_: bool
                                d_23_appendedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_nextIn_)
                                d_21_appendedGenerated_ = out12_
                                d_22_appendedInside_ = out13_
                                d_23_appendedCurrent_ = out14_
                                generated = d_21_appendedGenerated_
                                insideConstrainedOut = d_22_appendedInside_
                                currentConstrainedOut = d_23_appendedCurrent_
                        elif True:
                            d_24_symbolBudget_: int
                            if (stepTokenBudget) > (d_18_remainingInside_):
                                d_24_symbolBudget_ = d_18_remainingInside_
                            elif True:
                                d_24_symbolBudget_ = stepTokenBudget
                            d_25_symbolGenerated_: _dafny.Seq
                            d_26_symbolCurrent_: _dafny.Seq
                            d_27_hitEos_: bool
                            d_28_stepsUsed_: int
                            out15_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: int
                            out15_, out16_, out17_, out18_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_17_constrainedPrompt_, generated, currentConstrainedOut, d_24_symbolBudget_, eosToken)
                            d_25_symbolGenerated_ = out15_
                            d_26_symbolCurrent_ = out16_
                            d_27_hitEos_ = out17_
                            d_28_stepsUsed_ = out18_
                            generated = d_25_symbolGenerated_
                            currentConstrainedOut = d_26_symbolCurrent_
                            d_1_steps_ = (d_1_steps_) + (d_28_stepsUsed_)
                            if d_27_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


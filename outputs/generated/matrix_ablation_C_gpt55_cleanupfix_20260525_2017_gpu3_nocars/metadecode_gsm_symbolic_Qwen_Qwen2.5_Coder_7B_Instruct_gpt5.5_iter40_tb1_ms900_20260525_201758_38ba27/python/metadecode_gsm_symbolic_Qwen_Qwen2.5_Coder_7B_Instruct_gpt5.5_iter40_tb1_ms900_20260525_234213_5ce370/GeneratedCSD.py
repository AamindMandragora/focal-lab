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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the word problem step by step in plain text first. Do not write << until you are ready for the final answer expression. Then write exactly one final answer inside visible << >> delimiters. Inside the delimiters write only a valid arithmetic expression that computes the requested quantity, not an intermediate value: no prose, no units, no Markdown, no LaTeX. Preserve variable names exactly, including underscores. Use explicit * for multiplication and parentheses for grouping. Use all relevant quantities from the problem, especially subtracting known groups from totals, including current money already available, and summing every quantity*price term.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeSteps_: int
        d_2_freeSteps_ = 0
        d_3_forceReasonSteps_: int
        d_3_forceReasonSteps_ = 160
        d_4_sawSpan_: bool
        d_4_sawSpan_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_4_sawSpan_:
                            raise _dafny.Break("0")
                        elif (d_2_freeSteps_) >= (d_3_forceReasonSteps_):
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
                            d_2_freeSteps_ = 0
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                if (d_1_steps_) < (maxSteps):
                                    d_9_openedGenerated2_: _dafny.Seq
                                    d_10_openedInside2_: bool
                                    d_11_openedCurrent2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_9_openedGenerated2_ = out4_
                                    d_10_openedInside2_ = out5_
                                    d_11_openedCurrent2_ = out6_
                                    generated = d_9_openedGenerated2_
                                    insideConstrainedOut = d_10_openedInside2_
                                    currentConstrainedOut = d_11_openedCurrent2_
                                    d_2_freeSteps_ = 0
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                d_2_freeSteps_ = (d_2_freeSteps_) + (1)
                                if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_freeSteps_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
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
                        d_4_sawSpan_ = True
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_15_stablePrefix_: _dafny.Seq
                        d_15_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_16_constrainedPrompt_: _dafny.Seq
                        d_16_constrainedPrompt_ = (prompt) + (d_15_stablePrefix_)
                        d_17_remaining_: int
                        d_17_remaining_ = (maxSteps) - (d_1_steps_)
                        d_18_symbolBudget_: int
                        d_18_symbolBudget_ = d_17_remaining_
                        if (d_17_remaining_) > (1):
                            d_18_symbolBudget_ = (d_17_remaining_) - (1)
                        d_19_symbolGenerated_: _dafny.Seq
                        d_20_symbolOut_: _dafny.Seq
                        d_21_hitEos_: bool
                        d_22_stepsUsed_: int
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: int
                        out10_, out11_, out12_, out13_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_16_constrainedPrompt_, generated, currentConstrainedOut, d_18_symbolBudget_, eosToken)
                        d_19_symbolGenerated_ = out10_
                        d_20_symbolOut_ = out11_
                        d_21_hitEos_ = out12_
                        d_22_stepsUsed_ = out13_
                        generated = d_19_symbolGenerated_
                        currentConstrainedOut = d_20_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_22_stepsUsed_)
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_23_closedGenerated2_: _dafny.Seq
                            d_24_closedInside2_: bool
                            d_25_closedCurrent2_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_23_closedGenerated2_ = out14_
                            d_24_closedInside2_ = out15_
                            d_25_closedCurrent2_ = out16_
                            generated = d_23_closedGenerated2_
                            insideConstrainedOut = d_24_closedInside2_
                            currentConstrainedOut = d_25_closedCurrent2_
                            d_4_sawSpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif d_21_hitEos_:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


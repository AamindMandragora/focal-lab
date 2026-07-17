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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the word problem symbolically. Think briefly in plain text first, but do not write << or >> during the reasoning. When the final answer span begins, write only the final arithmetic expression inside it: no prose, no units, no Markdown, no LaTeX. Use exact variable names without braces, preserve underscores, use explicit * for multiplication, parentheses for grouping, // for integer division when a count is halved/quartered/floored, and int(...) for integer percentages or required integer conversions. Do not repeat these instructions or any example.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_reasoningSteps_: int
        d_2_reasoningSteps_ = 0
        d_3_reasoningLimit_: int
        d_3_reasoningLimit_ = 96
        d_4_reasoningPhase_: bool
        d_4_reasoningPhase_ = not(insideConstrained)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_4_reasoningPhase_) and ((d_2_reasoningSteps_) < (d_3_reasoningLimit_))) and (((d_1_steps_) + (2)) < (maxSteps)):
                            d_5_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_5_next_ = out0_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_reasoningSteps_ = (d_2_reasoningSteps_) + (1)
                            if (d_5_next_) == (eosToken):
                                d_4_reasoningPhase_ = False
                            elif (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_4_reasoningPhase_ = False
                            elif (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                d_4_reasoningPhase_ = False
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        elif True:
                            d_6_openedGenerated_: _dafny.Seq
                            d_7_openedInside_: bool
                            d_8_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openedGenerated_ = out1_
                            d_7_openedInside_ = out2_
                            d_8_openedCurrent_ = out3_
                            generated = d_6_openedGenerated_
                            insideConstrainedOut = d_7_openedInside_
                            currentConstrainedOut = d_8_openedCurrent_
                            d_4_reasoningPhase_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
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
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_12_stablePrefix_: _dafny.Seq
                        d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                        d_14_remaining_: int
                        d_14_remaining_ = (maxSteps) - (d_1_steps_)
                        d_15_symbolBudget_: int
                        d_15_symbolBudget_ = d_14_remaining_
                        if (d_14_remaining_) > (1):
                            d_15_symbolBudget_ = (d_14_remaining_) - (1)
                        d_16_symbolGenerated_: _dafny.Seq
                        d_17_symbolOut_: _dafny.Seq
                        d_18_hitEos_: bool
                        d_19_stepsUsed_: int
                        out7_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: int
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_13_constrainedPrompt_, generated, currentConstrainedOut, d_15_symbolBudget_, eosToken)
                        d_16_symbolGenerated_ = out7_
                        d_17_symbolOut_ = out8_
                        d_18_hitEos_ = out9_
                        d_19_stepsUsed_ = out10_
                        generated = d_16_symbolGenerated_
                        currentConstrainedOut = d_17_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_19_stepsUsed_)
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_20_closedGenerated2_: _dafny.Seq
                            d_21_closedInside2_: bool
                            d_22_closedCurrent2_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_20_closedGenerated2_ = out11_
                            d_21_closedInside2_ = out12_
                            d_22_closedCurrent2_ = out13_
                            generated = d_20_closedGenerated2_
                            insideConstrainedOut = d_21_closedInside2_
                            currentConstrainedOut = d_22_closedCurrent2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif d_18_hitEos_:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


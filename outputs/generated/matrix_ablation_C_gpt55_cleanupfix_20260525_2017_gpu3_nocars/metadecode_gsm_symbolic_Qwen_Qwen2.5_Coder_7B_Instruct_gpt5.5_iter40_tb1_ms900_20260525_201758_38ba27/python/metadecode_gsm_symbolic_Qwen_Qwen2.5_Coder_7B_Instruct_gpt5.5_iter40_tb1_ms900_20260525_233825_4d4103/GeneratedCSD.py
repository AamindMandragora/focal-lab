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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step, but make each visible constrained span contain only a symbolic arithmetic expression. Use only variables that appear in the problem statement; do not invent placeholder variables such as t, tf, total, answer, result, or n_3. Do not write assignments or equations inside a span: no '='. Preserve variable names exactly, including underscores. Use explicit * for multiplication and parentheses for grouping. For counts, trips, repeated intervals, groups, or whole-number shares, use // when integer division is intended. For percentage answers, express the requested percent difference directly with 100 factors or int(...) as appropriate. Combine all relevant quantities from the problem and output a valid final arithmetic expression inside visible << >> delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_openedGenerated_: _dafny.Seq
                        d_3_openedInside_: bool
                        d_4_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_2_openedGenerated_ = out0_
                        d_3_openedInside_ = out1_
                        d_4_openedCurrent_ = out2_
                        generated = d_2_openedGenerated_
                        insideConstrainedOut = d_3_openedInside_
                        currentConstrainedOut = d_4_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_5_closedGenerated_: _dafny.Seq
                        d_6_closedInside_: bool
                        d_7_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_5_closedGenerated_ = out3_
                        d_6_closedInside_ = out4_
                        d_7_closedCurrent_ = out5_
                        generated = d_5_closedGenerated_
                        insideConstrainedOut = d_6_closedInside_
                        currentConstrainedOut = d_7_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_8_stablePrefix_: _dafny.Seq
                        d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                        d_10_remaining_: int
                        d_10_remaining_ = (maxSteps) - (d_1_steps_)
                        d_11_symbolBudget_: int
                        d_11_symbolBudget_ = d_10_remaining_
                        if (d_10_remaining_) > (1):
                            d_11_symbolBudget_ = (d_10_remaining_) - (1)
                        d_12_symbolGenerated_: _dafny.Seq
                        d_13_symbolOut_: _dafny.Seq
                        d_14_hitEos_: bool
                        d_15_stepsUsed_: int
                        out6_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: int
                        out6_, out7_, out8_, out9_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_9_constrainedPrompt_, generated, currentConstrainedOut, d_11_symbolBudget_, eosToken)
                        d_12_symbolGenerated_ = out6_
                        d_13_symbolOut_ = out7_
                        d_14_hitEos_ = out8_
                        d_15_stepsUsed_ = out9_
                        generated = d_12_symbolGenerated_
                        currentConstrainedOut = d_13_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_15_stepsUsed_)
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_16_closedGenerated2_: _dafny.Seq
                            d_17_closedInside2_: bool
                            d_18_closedCurrent2_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_16_closedGenerated2_ = out10_
                            d_17_closedInside2_ = out11_
                            d_18_closedCurrent2_ = out12_
                            generated = d_16_closedGenerated2_
                            insideConstrainedOut = d_17_closedInside2_
                            currentConstrainedOut = d_18_closedCurrent2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif d_14_hitEos_:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


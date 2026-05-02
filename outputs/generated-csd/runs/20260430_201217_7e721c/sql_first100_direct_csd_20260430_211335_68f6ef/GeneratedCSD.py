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
        d_2_shortPhaseLimit_: int
        d_2_shortPhaseLimit_ = 28
        d_3_finishWindow_: int
        d_3_finishWindow_ = 32
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_5_openedGenerated_: _dafny.Seq
                                d_6_openedInside_: bool
                                d_7_openedCurrent_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_5_openedGenerated_ = out1_
                                d_6_openedInside_ = out2_
                                d_7_openedCurrent_ = out3_
                                generated = d_5_openedGenerated_
                                insideConstrainedOut = d_6_openedInside_
                                currentConstrainedOut = d_7_openedCurrent_
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                    elif True:
                        d_8_completeNow_: bool
                        d_8_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_completeNow_:
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
                        elif True:
                            d_12_stablePrefix_: _dafny.Seq
                            d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                            d_14_remaining_: int
                            d_14_remaining_ = (maxSteps) - (d_1_steps_)
                            if (len(currentConstrainedOut)) < (d_2_shortPhaseLimit_):
                                d_15_nextShort_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_15_nextShort_ = out7_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_15_nextShort_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_16_appendedGenerated_: _dafny.Seq
                                    d_17_appendedInside_: bool
                                    d_18_appendedCurrent_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_nextShort_)
                                    d_16_appendedGenerated_ = out8_
                                    d_17_appendedInside_ = out9_
                                    d_18_appendedCurrent_ = out10_
                                    generated = d_16_appendedGenerated_
                                    insideConstrainedOut = d_17_appendedInside_
                                    currentConstrainedOut = d_18_appendedCurrent_
                            elif True:
                                d_19_symbolBudget_: int
                                d_19_symbolBudget_ = 8
                                if (d_14_remaining_) <= (d_3_finishWindow_):
                                    d_19_symbolBudget_ = 3
                                if (d_14_remaining_) < (d_19_symbolBudget_):
                                    d_19_symbolBudget_ = d_14_remaining_
                                d_20_newCurrent_: _dafny.Seq
                                d_21_hitEos_: bool
                                d_22_stepsUsed_: int
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: int
                                out11_, out12_, out13_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, d_19_symbolBudget_, eosToken)
                                d_20_newCurrent_ = out11_
                                d_21_hitEos_ = out12_
                                d_22_stepsUsed_ = out13_
                                d_1_steps_ = (d_1_steps_) + (d_22_stepsUsed_)
                                if d_21_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    currentConstrainedOut = d_20_newCurrent_
                                    generated = (d_12_stablePrefix_) + (currentConstrainedOut)
                                    insideConstrainedOut = True
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, eosToken):
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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openSpanToken_: _dafny.Seq
        d_2_openSpanToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
        d_3_openedAny_: bool
        d_3_openedAny_ = insideConstrained
        d_4_openDelay_: int
        d_4_openDelay_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_3_openedAny_:
                            d_5_nextAfterSpan_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_5_nextAfterSpan_ = out0_
                            if (d_5_nextAfterSpan_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_nextAfterSpan_]))
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if (len(generated)) < ((len(generatedPrefix)) + (d_4_openDelay_)):
                                d_6_nextOutside_: _dafny.Seq
                                out1_: _dafny.Seq
                                out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_6_nextOutside_ = out1_
                                if (d_6_nextOutside_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_nextOutside_]))
                                    d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_7_remainingToOpen_: int
                                d_7_remainingToOpen_ = (maxSteps) - (d_1_steps_)
                                if (d_7_remainingToOpen_) < (3):
                                    d_8_nextLate_: _dafny.Seq
                                    out2_: _dafny.Seq
                                    out2_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                    d_8_nextLate_ = out2_
                                    if (d_8_nextLate_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_nextLate_]))
                                        d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    (lm).GenerateLogits((prompt) + (generated))
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_2_openSpanToken_]), _dafny.BigRational('1e2'))
                                    d_9_top_: _dafny.Seq
                                    out3_: _dafny.Seq
                                    out3_ = (d_0_helpers_).GetHighestLogitToken(lm)
                                    d_9_top_ = out3_
                                    if VerifiedDecoderAgent.default__.Contains(d_9_top_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                        d_10_openedGenerated_: _dafny.Seq
                                        d_11_openedInside_: bool
                                        d_12_openedCurrent_: _dafny.Seq
                                        out4_: _dafny.Seq
                                        out5_: bool
                                        out6_: _dafny.Seq
                                        out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                        d_10_openedGenerated_ = out4_
                                        d_11_openedInside_ = out5_
                                        d_12_openedCurrent_ = out6_
                                        generated = d_10_openedGenerated_
                                        insideConstrainedOut = d_11_openedInside_
                                        currentConstrainedOut = d_12_openedCurrent_
                                        d_3_openedAny_ = True
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        d_13_nextOutside2_: _dafny.Seq
                                        out7_: _dafny.Seq
                                        out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                        d_13_nextOutside2_ = out7_
                                        if (d_13_nextOutside2_) == (eosToken):
                                            raise _dafny.Break("0")
                                        elif True:
                                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_nextOutside2_]))
                                            d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_14_complete_: bool
                        d_14_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_14_complete_:
                            d_15_closedGenerated_: _dafny.Seq
                            d_16_closedInside_: bool
                            d_17_closedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_15_closedGenerated_ = out8_
                            d_16_closedInside_ = out9_
                            d_17_closedCurrent_ = out10_
                            generated = d_15_closedGenerated_
                            insideConstrainedOut = d_16_closedInside_
                            currentConstrainedOut = d_17_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if (stepTokenBudget) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_18_remaining_: int
                                d_18_remaining_ = (maxSteps) - (d_1_steps_)
                                if (d_18_remaining_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_19_stablePrefix_: _dafny.Seq
                                    d_19_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_20_constrainedPrompt_: _dafny.Seq
                                    d_20_constrainedPrompt_ = (prompt) + (d_19_stablePrefix_)
                                    d_21_currentOut_: _dafny.Seq
                                    d_22_hitEos_: bool
                                    d_23_stepsUsed_: int
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: int
                                    out11_, out12_, out13_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, stepTokenBudget, eosToken)
                                    d_21_currentOut_ = out11_
                                    d_22_hitEos_ = out12_
                                    d_23_stepsUsed_ = out13_
                                    if (d_22_hitEos_) or ((d_23_stepsUsed_) == (0)):
                                        raise _dafny.Break("0")
                                    elif True:
                                        if (d_23_stepsUsed_) <= (d_18_remaining_):
                                            generated = (d_19_stablePrefix_) + (d_21_currentOut_)
                                            insideConstrainedOut = True
                                            currentConstrainedOut = d_21_currentOut_
                                            d_1_steps_ = (d_1_steps_) + (d_23_stepsUsed_)
                                        elif True:
                                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


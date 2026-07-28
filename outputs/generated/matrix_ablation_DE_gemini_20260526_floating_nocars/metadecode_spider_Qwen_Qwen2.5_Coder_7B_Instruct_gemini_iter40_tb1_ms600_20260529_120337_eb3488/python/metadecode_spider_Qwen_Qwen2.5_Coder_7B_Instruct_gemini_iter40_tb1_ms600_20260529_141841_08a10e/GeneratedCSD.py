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
        d_2_unconstrainedSteps_: int
        d_2_unconstrainedSteps_ = 0
        d_3_PREAMBLE__BUDGET_: int
        d_3_PREAMBLE__BUDGET_ = 15
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_unconstrainedSteps_) < (d_3_PREAMBLE__BUDGET_):
                            d_4_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_4_next_ = out0_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_unconstrainedSteps_ = (d_2_unconstrainedSteps_) + (1)
                            if (d_4_next_) == (eosToken):
                                raise _dafny.Break("0")
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_5_enteredGenerated_: _dafny.Seq
                                d_6_enteredInside_: bool
                                d_7_enteredCurrent_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_5_enteredGenerated_ = out1_
                                d_6_enteredInside_ = out2_
                                d_7_enteredCurrent_ = out3_
                                generated = d_5_enteredGenerated_
                                insideConstrainedOut = d_6_enteredInside_
                                currentConstrainedOut = d_7_enteredCurrent_
                        elif True:
                            d_8_openedGenerated_: _dafny.Seq
                            d_9_openedInside_: bool
                            d_10_openedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_openedGenerated_ = out4_
                            d_9_openedInside_ = out5_
                            d_10_openedCurrent_ = out6_
                            generated = d_8_openedGenerated_
                            insideConstrainedOut = d_9_openedInside_
                            currentConstrainedOut = d_10_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_11_closedGenerated_: _dafny.Seq
                        d_12_closedInside_: bool
                        d_13_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_11_closedGenerated_ = out7_
                        d_12_closedInside_ = out8_
                        d_13_closedCurrent_ = out9_
                        generated = d_11_closedGenerated_
                        insideConstrainedOut = d_12_closedInside_
                        currentConstrainedOut = d_13_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_14_constrainedPrompt_: _dafny.Seq
                        d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_15_maxSymbolTokens_: int
                        d_15_maxSymbolTokens_ = 20
                        d_16_stepsAvailable_: int
                        d_16_stepsAvailable_ = (maxSteps) - (d_1_steps_)
                        if (d_15_maxSymbolTokens_) > (d_16_stepsAvailable_):
                            d_15_maxSymbolTokens_ = d_16_stepsAvailable_
                        if (d_15_maxSymbolTokens_) > (0):
                            d_17_generatedOut_: _dafny.Seq
                            d_18_currentOut_: _dafny.Seq
                            d_19_hitEos_: bool
                            d_20_stepsUsed_: int
                            out10_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: int
                            out10_, out11_, out12_, out13_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_14_constrainedPrompt_, generated, currentConstrainedOut, d_15_maxSymbolTokens_, eosToken)
                            d_17_generatedOut_ = out10_
                            d_18_currentOut_ = out11_
                            d_19_hitEos_ = out12_
                            d_20_stepsUsed_ = out13_
                            generated = d_17_generatedOut_
                            currentConstrainedOut = d_18_currentOut_
                            d_1_steps_ = (d_1_steps_) + (d_20_stepsUsed_)
                        elif True:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


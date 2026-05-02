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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_remaining0_: int
                        d_2_remaining0_ = (maxSteps) - (d_1_steps_)
                        if (d_2_remaining0_) == (0):
                            raise _dafny.Break("0")
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            d_3_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (lm).ChooseNextTokenUnconstrained()
                            d_3_next_ = out0_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            if (d_3_next_) == (eosToken):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                                d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                if VerifiedDecoderAgent.default__.Contains(d_3_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_4_openedGenerated_: _dafny.Seq
                                    d_5_openedInside_: bool
                                    d_6_openedCurrent_: _dafny.Seq
                                    out1_: _dafny.Seq
                                    out2_: bool
                                    out3_: _dafny.Seq
                                    out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_4_openedGenerated_ = out1_
                                    d_5_openedInside_ = out2_
                                    d_6_openedCurrent_ = out3_
                                    generated = d_4_openedGenerated_
                                    insideConstrainedOut = d_5_openedInside_
                                    currentConstrainedOut = d_6_openedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_7_isComplete_: bool
                        d_7_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_isComplete_:
                            d_8_remaining1_: int
                            d_8_remaining1_ = (maxSteps) - (d_1_steps_)
                            if (d_8_remaining1_) == (0):
                                raise _dafny.Break("0")
                            elif True:
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
                            d_12_remaining_: int
                            d_12_remaining_ = (maxSteps) - (d_1_steps_)
                            if ((stepTokenBudget) == (0)) or ((d_12_remaining_) == (0)):
                                raise _dafny.Break("0")
                            elif True:
                                d_13_stablePrefix_: _dafny.Seq
                                d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_14_constrainedPrompt_: _dafny.Seq
                                d_14_constrainedPrompt_ = (prompt) + (d_13_stablePrefix_)
                                d_15_symbolBudget_: int
                                d_15_symbolBudget_ = stepTokenBudget
                                if (d_12_remaining_) < (d_15_symbolBudget_):
                                    d_15_symbolBudget_ = d_12_remaining_
                                d_16_symbolOut_: _dafny.Seq
                                d_17_hitEos_: bool
                                d_18_stepsUsed_: int
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: int
                                out7_, out8_, out9_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, d_15_symbolBudget_, eosToken)
                                d_16_symbolOut_ = out7_
                                d_17_hitEos_ = out8_
                                d_18_stepsUsed_ = out9_
                                generated = (d_13_stablePrefix_) + (d_16_symbolOut_)
                                currentConstrainedOut = d_16_symbolOut_
                                d_1_steps_ = (d_1_steps_) + (d_18_stepsUsed_)
                                if d_17_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


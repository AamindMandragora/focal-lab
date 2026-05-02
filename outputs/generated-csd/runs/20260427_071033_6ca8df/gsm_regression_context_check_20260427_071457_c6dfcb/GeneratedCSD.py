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
                        d_2_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_2_next_ = out0_
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_2_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_3_isComplete_: bool
                        d_3_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_isComplete_:
                            d_4_closedGenerated_: _dafny.Seq
                            d_5_closedInside_: bool
                            d_6_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_4_closedGenerated_ = out1_
                            d_5_closedInside_ = out2_
                            d_6_closedCurrent_ = out3_
                            generated = d_4_closedGenerated_
                            insideConstrainedOut = d_5_closedInside_
                            currentConstrainedOut = d_6_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_remaining_: int
                            d_7_remaining_ = (maxSteps) - (d_1_steps_)
                            if ((stepTokenBudget) == (0)) or ((d_7_remaining_) == (0)):
                                raise _dafny.Break("0")
                            elif True:
                                d_8_stablePrefix_: _dafny.Seq
                                d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_9_constrainedPrompt_: _dafny.Seq
                                d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                                d_10_symbolBudget_: int
                                d_10_symbolBudget_ = stepTokenBudget
                                if (d_7_remaining_) < (d_10_symbolBudget_):
                                    d_10_symbolBudget_ = d_7_remaining_
                                d_11_symbolOut_: _dafny.Seq
                                d_12_hitEos_: bool
                                d_13_stepsUsed_: int
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: int
                                out4_, out5_, out6_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, d_10_symbolBudget_, eosToken)
                                d_11_symbolOut_ = out4_
                                d_12_hitEos_ = out5_
                                d_13_stepsUsed_ = out6_
                                generated = (d_8_stablePrefix_) + (d_11_symbolOut_)
                                currentConstrainedOut = d_11_symbolOut_
                                d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                                if d_12_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


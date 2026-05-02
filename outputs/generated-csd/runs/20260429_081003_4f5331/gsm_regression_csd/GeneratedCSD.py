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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
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
                            d_7_constrainedPrompt_: _dafny.Seq
                            d_7_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_8_validCount_: int
                            out4_: int
                            out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_8_validCount_ = out4_
                            if (((d_8_validCount_) <= (d_2_narrowThreshold_)) or ((stepTokenBudget) == (0))) or (((maxSteps) - (d_1_steps_)) == (1)):
                                d_9_next_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_9_next_ = out5_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_9_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_10_appendedGenerated_: _dafny.Seq
                                    d_11_appendedInside_: bool
                                    d_12_appendedCurrent_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                                    d_10_appendedGenerated_ = out6_
                                    d_11_appendedInside_ = out7_
                                    d_12_appendedCurrent_ = out8_
                                    generated = d_10_appendedGenerated_
                                    insideConstrainedOut = d_11_appendedInside_
                                    currentConstrainedOut = d_12_appendedCurrent_
                            elif True:
                                d_13_stablePrefix_: _dafny.Seq
                                d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_14_remaining_: int
                                d_14_remaining_ = (maxSteps) - (d_1_steps_)
                                d_15_symbolBudget_: int
                                if (stepTokenBudget) < (d_14_remaining_):
                                    d_15_symbolBudget_ = stepTokenBudget
                                elif True:
                                    d_15_symbolBudget_ = d_14_remaining_
                                d_16_symbolOut_: _dafny.Seq
                                d_17_hitEos_: bool
                                d_18_stepsUsed_: int
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: int
                                out9_, out10_, out11_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, d_15_symbolBudget_, eosToken)
                                d_16_symbolOut_ = out9_
                                d_17_hitEos_ = out10_
                                d_18_stepsUsed_ = out11_
                                generated = (d_13_stablePrefix_) + (d_16_symbolOut_)
                                insideConstrainedOut = True
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


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
        d_3_outsideThreshold_: int
        d_3_outsideThreshold_ = 3
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (len(generated)) < ((len(generatedPrefix)) + (d_3_outsideThreshold_)):
                            d_4_nextOutside_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_4_nextOutside_ = out0_
                            if (d_4_nextOutside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_nextOutside_]))
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_2_openSpanToken_]), _dafny.BigRational('1e2'))
                            d_5_top_: _dafny.Seq
                            out1_: _dafny.Seq
                            out1_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_5_top_ = out1_
                            if VerifiedDecoderAgent.default__.Contains(d_5_top_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_6_openedGenerated_: _dafny.Seq
                                d_7_openedInside_: bool
                                d_8_openedCurrent_: _dafny.Seq
                                out2_: _dafny.Seq
                                out3_: bool
                                out4_: _dafny.Seq
                                out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_6_openedGenerated_ = out2_
                                d_7_openedInside_ = out3_
                                d_8_openedCurrent_ = out4_
                                generated = d_6_openedGenerated_
                                insideConstrainedOut = d_7_openedInside_
                                currentConstrainedOut = d_8_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_9_nextOutside2_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_9_nextOutside2_ = out5_
                                if (d_9_nextOutside2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_nextOutside2_]))
                                    d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_10_complete_: bool
                        d_10_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_10_complete_:
                            d_11_closedGenerated_: _dafny.Seq
                            d_12_closedInside_: bool
                            d_13_closedCurrent_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_closedGenerated_ = out6_
                            d_12_closedInside_ = out7_
                            d_13_closedCurrent_ = out8_
                            generated = d_11_closedGenerated_
                            insideConstrainedOut = d_12_closedInside_
                            currentConstrainedOut = d_13_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if (stepTokenBudget) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_14_stablePrefix_: _dafny.Seq
                                d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_15_constrainedPrompt_: _dafny.Seq
                                d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                                d_16_currentOut_: _dafny.Seq
                                d_17_hitEos_: bool
                                d_18_stepsUsed_: int
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: int
                                out9_, out10_, out11_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, stepTokenBudget, eosToken)
                                d_16_currentOut_ = out9_
                                d_17_hitEos_ = out10_
                                d_18_stepsUsed_ = out11_
                                d_19_remaining_: int
                                d_19_remaining_ = (maxSteps) - (d_1_steps_)
                                if (d_18_stepsUsed_) <= (d_19_remaining_):
                                    if (d_17_hitEos_) or ((d_18_stepsUsed_) == (0)):
                                        raise _dafny.Break("0")
                                    elif True:
                                        generated = (d_14_stablePrefix_) + (d_16_currentOut_)
                                        insideConstrainedOut = True
                                        currentConstrainedOut = d_16_currentOut_
                                        d_1_steps_ = (d_1_steps_) + (d_18_stepsUsed_)
                                elif True:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


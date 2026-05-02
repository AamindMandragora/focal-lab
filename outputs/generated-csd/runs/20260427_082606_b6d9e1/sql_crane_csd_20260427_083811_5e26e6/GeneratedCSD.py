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
        if True:
            generated = generatedPrefix
            insideConstrainedOut = insideConstrained
            currentConstrainedOut = currentConstrained
            cost = 0
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_done_: bool
            d_2_done_ = False
            while ((d_1_steps_) < (maxSteps)) and (not(d_2_done_)):
                if not(insideConstrainedOut):
                    d_3_g1_: _dafny.Seq
                    d_4_ic1_: bool
                    d_5_c1_: _dafny.Seq
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: _dafny.Seq
                    out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_3_g1_ = out0_
                    d_4_ic1_ = out1_
                    d_5_c1_ = out2_
                    generated = d_3_g1_
                    insideConstrainedOut = d_4_ic1_
                    currentConstrainedOut = d_5_c1_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_6_isComplete_: bool
                    d_6_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if d_6_isComplete_:
                        if (d_1_steps_) < (maxSteps):
                            d_7_g2_: _dafny.Seq
                            d_8_ic2_: bool
                            d_9_c2_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_g2_ = out3_
                            d_8_ic2_ = out4_
                            d_9_c2_ = out5_
                            generated = d_7_g2_
                            insideConstrainedOut = d_8_ic2_
                            currentConstrainedOut = d_9_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        d_2_done_ = True
                    elif True:
                        if (stepTokenBudget) == (0):
                            d_2_done_ = True
                        elif True:
                            d_10_remaining_: int
                            d_10_remaining_ = (maxSteps) - (d_1_steps_)
                            if (stepTokenBudget) <= (d_10_remaining_):
                                d_11_stablePrefix_: _dafny.Seq
                                d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_12_constrainedPrompt_: _dafny.Seq
                                d_12_constrainedPrompt_ = (prompt) + (d_11_stablePrefix_)
                                d_13_currentOut_: _dafny.Seq
                                d_14_hitEos_: bool
                                d_15_stepsUsed_: int
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: int
                                out6_, out7_, out8_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, stepTokenBudget, eosToken)
                                d_13_currentOut_ = out6_
                                d_14_hitEos_ = out7_
                                d_15_stepsUsed_ = out8_
                                generated = (d_11_stablePrefix_) + (d_13_currentOut_)
                                currentConstrainedOut = d_13_currentOut_
                                d_1_steps_ = (d_1_steps_) + (d_15_stepsUsed_)
                                if (d_14_hitEos_) or ((d_15_stepsUsed_) == (0)):
                                    d_2_done_ = True
                            elif True:
                                d_2_done_ = True
            cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


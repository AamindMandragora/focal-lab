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
            d_3_openTok_: _dafny.Seq
            d_3_openTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
            d_4_closeTok_: _dafny.Seq
            d_4_closeTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))
            while ((d_1_steps_) < (maxSteps)) and (not(d_2_done_)):
                if not(insideConstrainedOut):
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_openTok_]))
                    insideConstrainedOut = True
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_5_isComplete_: bool
                    d_5_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if d_5_isComplete_:
                        if (d_1_steps_) < (maxSteps):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_closeTok_]))
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (1)
                        d_2_done_ = True
                    elif True:
                        d_6_remaining_: int
                        d_6_remaining_ = (maxSteps) - (d_1_steps_)
                        d_7_stablePrefix_: _dafny.Seq
                        d_7_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_8_constrainedPrompt_: _dafny.Seq
                        d_8_constrainedPrompt_ = (prompt) + (d_7_stablePrefix_)
                        d_9_currentOut_: _dafny.Seq
                        d_10_hitEos_: bool
                        d_11_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: int
                        out0_, out1_, out2_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, d_6_remaining_, eosToken)
                        d_9_currentOut_ = out0_
                        d_10_hitEos_ = out1_
                        d_11_stepsUsed_ = out2_
                        generated = (d_7_stablePrefix_) + (d_9_currentOut_)
                        currentConstrainedOut = d_9_currentOut_
                        d_1_steps_ = (d_1_steps_) + (d_11_stepsUsed_)
                        if d_10_hitEos_:
                            d_2_done_ = True
            cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


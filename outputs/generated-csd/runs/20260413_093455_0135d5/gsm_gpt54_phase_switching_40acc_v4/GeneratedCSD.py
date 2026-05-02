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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, eosToken):
        generated: _dafny.Seq = _dafny.Seq({})
        insideConstrainedOut: bool = False
        currentConstrainedOut: _dafny.Seq = _dafny.Seq({})
        cost: int = int(0)
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        if True:
            d_1_helpers_: VerifiedDecoderAgent.CSDHelpers
            nw1_ = VerifiedDecoderAgent.CSDHelpers()
            nw1_.ctor__()
            d_1_helpers_ = nw1_
            generated = generatedPrefix
            insideConstrainedOut = insideConstrained
            currentConstrainedOut = currentConstrained
            cost = 0
            d_2_suffix_: _dafny.Seq
            d_2_suffix_ = _dafny.SeqWithoutIsStrInference([])
            d_3_steps_: int
            d_3_steps_ = 0
            d_4_remaining_: int
            d_4_remaining_ = maxSteps
            if insideConstrainedOut:
                d_5_initiallyComplete_: bool
                d_5_initiallyComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_5_initiallyComplete_:
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            while (d_4_remaining_) > (0):
                if not(insideConstrainedOut):
                    d_6_candidates_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_1_helpers_).TopUnconstrainedCandidates(lm, prompt, generated, 3)
                    d_6_candidates_ = out0_
                    d_7_next_: _dafny.Seq
                    d_7_next_ = (d_6_candidates_)[0]
                    d_8_i_: int
                    d_8_i_ = 0
                    while (d_8_i_) < (len(d_6_candidates_)):
                        if VerifiedDecoderAgent.default__.Contains((d_6_candidates_)[d_8_i_], _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_7_next_ = (d_6_candidates_)[d_8_i_]
                            d_8_i_ = len(d_6_candidates_)
                        elif True:
                            d_8_i_ = (d_8_i_) + (1)
                    d_3_steps_ = (d_3_steps_) + (1)
                    d_4_remaining_ = (d_4_remaining_) - (1)
                    if (d_7_next_) == (eosToken):
                        d_4_remaining_ = 0
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                        d_2_suffix_ = (d_2_suffix_) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                        if VerifiedDecoderAgent.default__.Contains(d_7_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                elif True:
                    d_9_canStep_: bool
                    d_9_canStep_ = not((parser).IsCompletePrefix(currentConstrainedOut))
                    if not(d_9_canStep_):
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_4_remaining_ = (d_4_remaining_) - (1)
                        d_3_steps_ = (d_3_steps_) + (1)
                    elif True:
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_11_next_: _dafny.Seq
                        d_12_wasConstrained_: bool
                        out1_: _dafny.Seq
                        out2_: bool
                        out1_, out2_ = (d_1_helpers_).ConfidenceGatedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_11_next_ = out1_
                        d_12_wasConstrained_ = out2_
                        d_3_steps_ = (d_3_steps_) + (1)
                        d_4_remaining_ = (d_4_remaining_) - (1)
                        if (d_11_next_) == (eosToken):
                            d_4_remaining_ = 0
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                            d_2_suffix_ = (d_2_suffix_) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                            if VerifiedDecoderAgent.default__.Contains(d_11_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                d_13_updatedConstrained_: _dafny.Seq
                                d_13_updatedConstrained_ = (currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                d_14_nowComplete_: bool
                                d_14_nowComplete_ = (parser).IsCompletePrefix(d_13_updatedConstrained_)
                                if d_14_nowComplete_:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                elif True:
                                    currentConstrainedOut = d_13_updatedConstrained_
            cost = d_1_helpers_.cost
        return generated, insideConstrainedOut, currentConstrainedOut, cost


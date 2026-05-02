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
            d_3_remaining_: int
            d_3_remaining_ = maxSteps
            while (d_3_remaining_) > (0):
                d_3_remaining_ = (d_3_remaining_) - (1)
                if not(insideConstrainedOut):
                    d_4_candidates_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_1_helpers_).TopUnconstrainedCandidates(lm, prompt, generated, 5)
                    d_4_candidates_ = out0_
                    d_5_next_: _dafny.Seq
                    d_5_next_ = (d_4_candidates_)[0]
                    d_6_i_: int
                    d_6_i_ = 0
                    while (d_6_i_) < (len(d_4_candidates_)):
                        if VerifiedDecoderAgent.default__.Contains((d_4_candidates_)[d_6_i_], _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_5_next_ = (d_4_candidates_)[d_6_i_]
                            d_6_i_ = len(d_4_candidates_)
                        elif True:
                            d_6_i_ = (d_6_i_) + (1)
                    if (d_5_next_) == (eosToken):
                        d_3_remaining_ = 0
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        d_2_suffix_ = (d_2_suffix_) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        if VerifiedDecoderAgent.default__.Contains(d_5_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_7_startsValid_: bool
                            d_7_startsValid_ = (parser).IsValidPrefix(_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            if d_7_startsValid_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([d_5_next_])
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                elif True:
                    d_8_deadEnd_: bool
                    out1_: bool
                    out1_ = (d_1_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                    d_8_deadEnd_ = out1_
                    d_9_isComplete_: bool
                    d_9_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if (d_8_deadEnd_) or (d_9_isComplete_):
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_10_baseGenerated_: _dafny.Seq
                        d_10_baseGenerated_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (d_10_baseGenerated_)
                        d_12_next_: _dafny.Seq
                        out2_: _dafny.Seq
                        out2_ = (d_1_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_12_next_ = out2_
                        if (d_12_next_) == (eosToken):
                            d_3_remaining_ = 0
                        elif True:
                            d_13_canAppend_: bool
                            out3_: bool
                            out3_ = (d_1_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_12_next_)
                            d_13_canAppend_ = out3_
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                            d_2_suffix_ = (d_2_suffix_) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                            if d_13_canAppend_:
                                currentConstrainedOut = (currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                if VerifiedDecoderAgent.default__.Contains(d_12_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            cost = d_1_helpers_.cost
        return generated, insideConstrainedOut, currentConstrainedOut, cost


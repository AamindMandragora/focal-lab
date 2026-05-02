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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_suffix_: _dafny.Seq
        d_1_suffix_ = _dafny.SeqWithoutIsStrInference([])
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_remaining_: int
        d_3_remaining_ = maxSteps
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_candidates_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).TopUnconstrainedCandidates(lm, prompt, generated, 4)
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
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_3_remaining_ = (d_3_remaining_) - (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            if VerifiedDecoderAgent.default__.Contains(d_5_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_7_isComplete_: bool
                        d_7_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_isComplete_:
                            d_8_newGenerated_: _dafny.Seq
                            d_9_newInside_: bool
                            d_10_newCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_8_newGenerated_ = out1_
                            d_9_newInside_ = out2_
                            d_10_newCurrent_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_remaining_ = (d_3_remaining_) - (1)
                            generated = d_8_newGenerated_
                            d_1_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                            insideConstrainedOut = d_9_newInside_
                            currentConstrainedOut = d_10_newCurrent_
                        elif True:
                            d_11_next_: _dafny.Seq
                            d_12_isValid_: bool
                            out4_: _dafny.Seq
                            out5_: bool
                            out4_, out5_ = (d_0_helpers_).SoftConstrainedStep(lm, parser, prompt, currentConstrainedOut, _dafny.BigRational('2e0'), eosToken)
                            d_11_next_ = out4_
                            d_12_isValid_ = out5_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_remaining_ = (d_3_remaining_) - (1)
                            if (d_11_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif d_12_isValid_:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                if VerifiedDecoderAgent.default__.Contains(d_11_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                elif True:
                                    currentConstrainedOut = (currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        cost = d_0_helpers_.cost
        return generated, insideConstrainedOut, currentConstrainedOut, cost


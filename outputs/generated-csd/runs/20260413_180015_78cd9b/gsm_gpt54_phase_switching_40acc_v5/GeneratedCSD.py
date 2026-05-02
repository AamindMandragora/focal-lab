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
        d_2_fuel_: int
        d_2_fuel_ = maxSteps
        with _dafny.label("0"):
            while (d_2_fuel_) > (0):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_candidates_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).TopUnconstrainedCandidates(lm, prompt, generated, 3)
                        d_3_candidates_ = out0_
                        d_4_next_: _dafny.Seq
                        d_4_next_ = (d_3_candidates_)[0]
                        d_5_i_: int
                        d_5_i_ = 0
                        while (d_5_i_) < (len(d_3_candidates_)):
                            if VerifiedDecoderAgent.default__.Contains((d_3_candidates_)[d_5_i_], _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_4_next_ = (d_3_candidates_)[d_5_i_]
                                d_5_i_ = len(d_3_candidates_)
                            elif True:
                                d_5_i_ = (d_5_i_) + (1)
                        d_2_fuel_ = (d_2_fuel_) - (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                        d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                        if VerifiedDecoderAgent.default__.Contains(d_4_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_6_isComplete_: bool
                        d_6_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_6_isComplete_:
                            d_7_next_: _dafny.Seq
                            d_7_next_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))
                            d_2_fuel_ = (d_2_fuel_) - (1)
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                            d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            d_8_next_: _dafny.Seq
                            out1_: _dafny.Seq
                            out1_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])), currentConstrainedOut, eosToken)
                            d_8_next_ = out1_
                            d_2_fuel_ = (d_2_fuel_) - (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                            d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                            if VerifiedDecoderAgent.default__.Contains(d_8_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                currentConstrainedOut = (currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                    pass
            pass
        cost = d_0_helpers_.cost
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
        (d_0_helpers_).cost = 0
        d_1_suffix_: _dafny.Seq
        d_1_suffix_ = _dafny.SeqWithoutIsStrInference([])
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_remaining_: int
        d_3_remaining_ = maxSteps
        with _dafny.label("0"):
            while (d_3_remaining_) > (0):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_3_remaining_ = (d_3_remaining_) - (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                        d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                        if VerifiedDecoderAgent.default__.Contains(d_4_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_5_isComplete_: bool
                        d_5_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_isComplete_:
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            d_6_constrainedPrompt_: _dafny.Seq
                            d_6_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_7_narrow_: bool
                            out1_: bool
                            out1_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_7_narrow_ = out1_
                            if d_7_narrow_:
                                d_8_nextStrict_: _dafny.Seq
                                out2_: _dafny.Seq
                                out2_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_6_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_8_nextStrict_ = out2_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_3_remaining_ = (d_3_remaining_) - (1)
                                if (d_8_nextStrict_) == (eosToken):
                                    raise _dafny.Break("0")
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_nextStrict_]))
                                d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_8_nextStrict_]))
                                if VerifiedDecoderAgent.default__.Contains(d_8_nextStrict_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                elif True:
                                    currentConstrainedOut = (currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([d_8_nextStrict_]))
                            elif True:
                                d_9_nextSoft_: _dafny.Seq
                                d_10_isValid_: bool
                                out3_: _dafny.Seq
                                out4_: bool
                                out3_, out4_ = (d_0_helpers_).SoftConstrainedStep(lm, parser, d_6_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('2e0'), eosToken)
                                d_9_nextSoft_ = out3_
                                d_10_isValid_ = out4_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_3_remaining_ = (d_3_remaining_) - (1)
                                if (d_9_nextSoft_) == (eosToken):
                                    raise _dafny.Break("0")
                                if d_10_isValid_:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_nextSoft_]))
                                    d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_9_nextSoft_]))
                                    if VerifiedDecoderAgent.default__.Contains(d_9_nextSoft_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    elif True:
                                        currentConstrainedOut = (currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([d_9_nextSoft_]))
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        cost = d_0_helpers_.cost
        return generated, insideConstrainedOut, currentConstrainedOut, cost


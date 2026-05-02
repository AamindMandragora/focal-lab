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
        d_2_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_flatGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_3_isComplete_: bool
                        d_3_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_isComplete_:
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            generated = out1_
                            insideConstrainedOut = out2_
                            currentConstrainedOut = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_4_dead_: bool
                            out4_: bool
                            out4_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_4_dead_ = out4_
                            if d_4_dead_:
                                d_5_repaired_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")))
                                d_5_repaired_ = out5_
                                if (len(d_5_repaired_)) == (len(currentConstrainedOut)):
                                    out6_: _dafny.Seq
                                    out6_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                    d_5_repaired_ = out6_
                                out7_: _dafny.Seq
                                out8_: _dafny.Seq
                                out7_, out8_ = (d_0_helpers_).RollbackConstrainedSpan(parser, generatedPrefix, generated, currentConstrainedOut)
                                generated = out7_
                                currentConstrainedOut = out8_
                                out9_: _dafny.Seq
                                out10_: _dafny.Seq
                                out9_, out10_ = (d_0_helpers_).RollbackConstrainedSpan(parser, generatedPrefix, generated, d_5_repaired_)
                                generated = out9_
                                currentConstrainedOut = out10_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                if (len(d_2_flatGroups_)) > (0):
                                    (lm).GenerateLogits((prompt) + (generated))
                                    d_6_i_: int
                                    d_6_i_ = 0
                                    while (d_6_i_) < (len(validTokenGroups)):
                                        d_7_anyValid_: bool
                                        out11_: bool
                                        out11_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, (validTokenGroups)[d_6_i_])
                                        d_7_anyValid_ = out11_
                                        if d_7_anyValid_:
                                            (d_0_helpers_).BoostTokenLogits(lm, (validTokenGroups)[d_6_i_], _dafny.BigRational('3e0'))
                                        d_6_i_ = (d_6_i_) + (1)
                                    (d_0_helpers_).BoostTokenLogits(lm, d_2_flatGroups_, _dafny.BigRational('1e0'))
                                    (d_0_helpers_).ScaleAllLogits(lm, _dafny.BigRational('1e0'))
                                d_8_next_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_8_next_ = out12_
                                if (d_8_next_) == (eosToken):
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                                elif True:
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_next_)
                                    generated = out13_
                                    insideConstrainedOut = out14_
                                    currentConstrainedOut = out15_
                                    d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_9_next_: _dafny.Seq
                        out16_: _dafny.Seq
                        out16_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_9_next_ = out16_
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_9_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if VerifiedDecoderAgent.default__.Contains(d_9_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).OpenConstrainedSpan(lm, _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (1):]))
                                generated = out17_
                                insideConstrainedOut = out18_
                                currentConstrainedOut = out19_
                                d_1_steps_ = d_1_steps_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


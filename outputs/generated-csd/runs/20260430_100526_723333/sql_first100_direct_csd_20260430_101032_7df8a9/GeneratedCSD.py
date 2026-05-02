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
        if insideConstrained:
            insideConstrainedOut = True
            currentConstrainedOut = currentConstrained
        elif True:
            insideConstrainedOut = True
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_2_complete_: bool
                    d_2_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if d_2_complete_:
                        (lm).GenerateLogits((prompt) + (generated))
                        if (len(validTokenGroups)) > (0):
                            (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('8e0'))
                        d_3_next0_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).ConstrainedSample(lm, parser, currentConstrainedOut, eosToken)
                        d_3_next0_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next0_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_4_g0_: _dafny.Seq
                            d_5_i0_: bool
                            d_6_c0_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_3_next0_)
                            d_4_g0_ = out1_
                            d_5_i0_ = out2_
                            d_6_c0_ = out3_
                            generated = d_4_g0_
                            insideConstrainedOut = d_5_i0_
                            currentConstrainedOut = d_6_c0_
                    elif True:
                        d_7_validCount_: int
                        out4_: int
                        out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_7_validCount_ = out4_
                        d_8_dead_: bool
                        out5_: bool
                        out5_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_8_dead_ = out5_
                        if (d_8_dead_) or ((d_7_validCount_) == (0)):
                            d_9_repaired_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ")))
                            d_9_repaired_ = out6_
                            if (len(d_9_repaired_)) == (len(currentConstrainedOut)):
                                d_10_repaired2_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_10_repaired2_ = out7_
                                d_9_repaired_ = d_10_repaired2_
                            d_11_trim_: int
                            d_11_trim_ = (len(currentConstrainedOut)) - (len(d_9_repaired_))
                            generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (d_11_trim_):])
                            currentConstrainedOut = d_9_repaired_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            if (len(validTokenGroups)) > (0):
                                (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('8e0'))
                            d_12_next_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).ConstrainedSample(lm, parser, currentConstrainedOut, eosToken)
                            d_12_next_ = out8_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_13_g1_: _dafny.Seq
                                d_14_i1_: bool
                                d_15_c1_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                d_13_g1_ = out9_
                                d_14_i1_ = out10_
                                d_15_c1_ = out11_
                                generated = d_13_g1_
                                insideConstrainedOut = d_14_i1_
                                currentConstrainedOut = d_15_c1_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


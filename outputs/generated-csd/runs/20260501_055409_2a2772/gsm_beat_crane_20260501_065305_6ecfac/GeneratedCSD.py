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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_closedOneSpan_: bool
        d_2_closedOneSpan_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_3_completeNow_: bool
                        d_3_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_completeNow_:
                            if ((d_1_steps_) + (1)) <= (maxSteps):
                                d_4_gClose_: _dafny.Seq
                                d_5_iClose_: bool
                                d_6_cClose_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_4_gClose_ = out0_
                                d_5_iClose_ = out1_
                                d_6_cClose_ = out2_
                                generated = d_4_gClose_
                                insideConstrainedOut = d_5_iClose_
                                currentConstrainedOut = d_6_cClose_
                                d_2_closedOneSpan_ = True
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            if (8) <= (len(currentConstrainedOut)):
                                d_7_repairedLong_: _dafny.Seq
                                out3_: _dafny.Seq
                                out3_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_7_repairedLong_ = out3_
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(currentConstrainedOut)) - (len(d_7_repairedLong_))):])
                                currentConstrainedOut = d_7_repairedLong_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_8_next_: _dafny.Seq
                                out4_: _dafny.Seq
                                out4_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (generated), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_8_next_ = out4_
                                if (d_8_next_) == (eosToken):
                                    d_9_repaired_: _dafny.Seq
                                    out5_: _dafny.Seq
                                    out5_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                    d_9_repaired_ = out5_
                                    generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(currentConstrainedOut)) - (len(d_9_repaired_))):])
                                    currentConstrainedOut = d_9_repaired_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_10_gApp_: _dafny.Seq
                                    d_11_iApp_: bool
                                    d_12_cApp_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_next_)
                                    d_10_gApp_ = out6_
                                    d_11_iApp_ = out7_
                                    d_12_cApp_ = out8_
                                    generated = d_10_gApp_
                                    insideConstrainedOut = d_11_iApp_
                                    currentConstrainedOut = d_12_cApp_
                                    d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_13_nextFree_: _dafny.Seq
                        out9_: _dafny.Seq
                        out9_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_13_nextFree_ = out9_
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_nextFree_]))
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_13_nextFree_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if ((not(d_2_closedOneSpan_)) and (VerifiedDecoderAgent.default__.Contains(d_13_nextFree_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))))) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)):
                                d_14_gOpen_: _dafny.Seq
                                d_15_iOpen_: bool
                                d_16_cOpen_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (1):]))
                                d_14_gOpen_ = out10_
                                d_15_iOpen_ = out11_
                                d_16_cOpen_ = out12_
                                generated = d_14_gOpen_
                                insideConstrainedOut = d_15_iOpen_
                                currentConstrainedOut = d_16_cOpen_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


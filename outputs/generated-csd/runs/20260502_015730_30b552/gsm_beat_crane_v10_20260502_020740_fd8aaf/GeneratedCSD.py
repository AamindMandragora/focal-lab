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
        d_2_earlySpanLimit_: int
        d_2_earlySpanLimit_ = 8
        d_3_rollbackLimit_: int
        d_3_rollbackLimit_ = 24
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_5_isComplete_: bool
                        d_5_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_isComplete_:
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out1_
                            d_7_closedInside_ = out2_
                            d_8_closedCurrent_ = out3_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if (len(currentConstrainedOut)) >= (d_3_rollbackLimit_):
                                d_9_repaired_: _dafny.Seq
                                out4_: _dafny.Seq
                                out4_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_9_repaired_ = out4_
                                d_10_trim_: int
                                d_10_trim_ = (len(currentConstrainedOut)) - (len(d_9_repaired_))
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (d_10_trim_):])
                                currentConstrainedOut = d_9_repaired_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_11_stablePrefix_: _dafny.Seq
                                d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                if (len(currentConstrainedOut)) < (d_2_earlySpanLimit_):
                                    d_12_next_: _dafny.Seq
                                    out5_: _dafny.Seq
                                    out5_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, (prompt) + (d_11_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), eosToken)
                                    d_12_next_ = out5_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_12_next_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_13_appendedGenerated_: _dafny.Seq
                                        d_14_appendedInside_: bool
                                        d_15_appendedCurrent_: _dafny.Seq
                                        out6_: _dafny.Seq
                                        out7_: bool
                                        out8_: _dafny.Seq
                                        out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                        d_13_appendedGenerated_ = out6_
                                        d_14_appendedInside_ = out7_
                                        d_15_appendedCurrent_ = out8_
                                        generated = d_13_appendedGenerated_
                                        insideConstrainedOut = d_14_appendedInside_
                                        currentConstrainedOut = d_15_appendedCurrent_
                                elif True:
                                    (lm).GenerateLogits(((prompt) + (d_11_stablePrefix_)) + (currentConstrainedOut))
                                    (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                                    d_16_sampled_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out9_ = (d_0_helpers_).ConstrainedSample(lm, parser, currentConstrainedOut, eosToken)
                                    d_16_sampled_ = out9_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_16_sampled_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_17_appendedGenerated2_: _dafny.Seq
                                        d_18_appendedInside2_: bool
                                        d_19_appendedCurrent2_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_sampled_)
                                        d_17_appendedGenerated2_ = out10_
                                        d_18_appendedInside2_ = out11_
                                        d_19_appendedCurrent2_ = out12_
                                        generated = d_17_appendedGenerated2_
                                        insideConstrainedOut = d_18_appendedInside2_
                                        currentConstrainedOut = d_19_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


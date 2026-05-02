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
        d_2_canOpenDirectly_: bool
        d_2_canOpenDirectly_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_canOpenDirectly_) and ((len(generated)) == (len(generatedPrefix))):
                            d_3_openedGenerated_: _dafny.Seq
                            d_4_openedInside_: bool
                            d_5_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_3_openedGenerated_ = out0_
                            d_4_openedInside_ = out1_
                            d_5_openedCurrent_ = out2_
                            generated = d_3_openedGenerated_
                            insideConstrainedOut = d_4_openedInside_
                            currentConstrainedOut = d_5_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_6_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                                if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_7_dead_: bool
                        out4_: bool
                        out4_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 0)
                        d_7_dead_ = out4_
                        if d_7_dead_:
                            d_8_repaired1_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")))
                            d_8_repaired1_ = out5_
                            d_9_repaired2_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")))
                            d_9_repaired2_ = out6_
                            if (len(d_8_repaired1_)) < (len(currentConstrainedOut)):
                                d_10_stablePrefix1_: _dafny.Seq
                                d_10_stablePrefix1_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                generated = (d_10_stablePrefix1_) + (d_8_repaired1_)
                                insideConstrainedOut = True
                                currentConstrainedOut = d_8_repaired1_
                            elif True:
                                d_11_stablePrefix2_: _dafny.Seq
                                d_11_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                generated = (d_11_stablePrefix2_) + (d_9_repaired2_)
                                insideConstrainedOut = True
                                currentConstrainedOut = d_9_repaired2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_12_stablePrefix_: _dafny.Seq
                            d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            (lm).GenerateLogits(((prompt) + (d_12_stablePrefix_)) + (currentConstrainedOut))
                            d_13_completeNow_: bool
                            d_13_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if (len(validTokenGroups)) > (0):
                                (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'))
                            if d_13_completeNow_:
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                                d_14_flat_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                d_14_flat_ = out7_
                                if (len(d_14_flat_)) > (0):
                                    (d_0_helpers_).PenalizeTokenLogits(lm, d_14_flat_, _dafny.BigRational('8e0'))
                            d_15_next2_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).ConstrainedSample(lm, parser, currentConstrainedOut, eosToken)
                            d_15_next2_ = out8_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_15_next2_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_16_appendedGenerated_: _dafny.Seq
                                d_17_appendedInside_: bool
                                d_18_appendedCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next2_)
                                d_16_appendedGenerated_ = out9_
                                d_17_appendedInside_ = out10_
                                d_18_appendedCurrent_ = out11_
                                generated = d_16_appendedGenerated_
                                insideConstrainedOut = d_17_appendedInside_
                                currentConstrainedOut = d_18_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


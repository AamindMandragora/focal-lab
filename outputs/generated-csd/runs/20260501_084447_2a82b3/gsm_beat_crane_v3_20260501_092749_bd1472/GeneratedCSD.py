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
        d_2_openedSpan_: bool
        d_2_openedSpan_ = insideConstrained
        d_3_completedSpan_: bool
        d_3_completedSpan_ = False
        d_4_warmup_: int
        d_4_warmup_ = 3
        d_5_openAvailable_: bool
        d_5_openAvailable_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)
        d_6_reopenPenalty_: _dafny.Seq
        d_6_reopenPenalty_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_3_completedSpan_:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).PenalizeTokenLogits(lm, d_6_reopenPenalty_, _dafny.BigRational('12e0'))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('8e0'))
                            d_7_nextAfter_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (lm).ChooseNextTokenUnconstrained()
                            d_7_nextAfter_ = out0_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_nextAfter_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_nextAfter_]))
                        elif True:
                            if (((not(d_2_openedSpan_)) and (d_5_openAvailable_)) and ((d_1_steps_) >= (d_4_warmup_))) and (((d_1_steps_) + (2)) <= (maxSteps)):
                                d_8_openedGenerated_: _dafny.Seq
                                d_9_openedInside_: bool
                                d_10_openedCurrent_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_8_openedGenerated_ = out1_
                                d_9_openedInside_ = out2_
                                d_10_openedCurrent_ = out3_
                                generated = d_8_openedGenerated_
                                insideConstrainedOut = d_9_openedInside_
                                currentConstrainedOut = d_10_openedCurrent_
                                d_2_openedSpan_ = True
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_11_nextFree_: _dafny.Seq
                                out4_: _dafny.Seq
                                out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_11_nextFree_ = out4_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_11_nextFree_) == (eosToken):
                                    if ((not(d_2_openedSpan_)) and (d_5_openAvailable_)) and ((d_1_steps_) < (maxSteps)):
                                        d_12_forcedGenerated_: _dafny.Seq
                                        d_13_forcedInside_: bool
                                        d_14_forcedCurrent_: _dafny.Seq
                                        out5_: _dafny.Seq
                                        out6_: bool
                                        out7_: _dafny.Seq
                                        out5_, out6_, out7_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                        d_12_forcedGenerated_ = out5_
                                        d_13_forcedInside_ = out6_
                                        d_14_forcedCurrent_ = out7_
                                        generated = d_12_forcedGenerated_
                                        insideConstrainedOut = d_13_forcedInside_
                                        currentConstrainedOut = d_14_forcedCurrent_
                                        d_2_openedSpan_ = True
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_nextFree_]))
                                    if ((d_11_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and (not(d_2_openedSpan_)):
                                        insideConstrainedOut = True
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                        d_2_openedSpan_ = True
                    elif True:
                        d_15_completeNow_: bool
                        d_15_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_15_completeNow_:
                            d_16_closedGenerated_: _dafny.Seq
                            d_17_closedInside_: bool
                            d_18_closedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_16_closedGenerated_ = out8_
                            d_17_closedInside_ = out9_
                            d_18_closedCurrent_ = out10_
                            generated = d_16_closedGenerated_
                            insideConstrainedOut = d_17_closedInside_
                            currentConstrainedOut = d_18_closedCurrent_
                            d_3_completedSpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_19_stablePrefix_: _dafny.Seq
                            d_19_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_20_nextConstrained_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_19_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_20_nextConstrained_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_20_nextConstrained_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_21_appendedGenerated_: _dafny.Seq
                                d_22_appendedInside_: bool
                                d_23_appendedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_nextConstrained_)
                                d_21_appendedGenerated_ = out12_
                                d_22_appendedInside_ = out13_
                                d_23_appendedCurrent_ = out14_
                                generated = d_21_appendedGenerated_
                                insideConstrainedOut = d_22_appendedInside_
                                currentConstrainedOut = d_23_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


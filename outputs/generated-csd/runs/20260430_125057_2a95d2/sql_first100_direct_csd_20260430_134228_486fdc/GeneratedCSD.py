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
        d_2_minCloseLen_: int
        d_2_minCloseLen_ = 8
        d_3_rollbackLen_: int
        d_3_rollbackLen_ = 80
        d_4_boundaryToken_: _dafny.Seq
        d_4_boundaryToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_openedGenerated_: _dafny.Seq
                        d_6_openedInside_: bool
                        d_7_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_5_openedGenerated_ = out0_
                        d_6_openedInside_ = out1_
                        d_7_openedCurrent_ = out2_
                        generated = d_5_openedGenerated_
                        insideConstrainedOut = d_6_openedInside_
                        currentConstrainedOut = d_7_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_8_completeNow_: bool
                        d_8_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if (d_8_completeNow_) and ((len(currentConstrainedOut)) >= (d_2_minCloseLen_)):
                            d_9_closedGenerated_: _dafny.Seq
                            d_10_closedInside_: bool
                            d_11_closedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated_ = out3_
                            d_10_closedInside_ = out4_
                            d_11_closedCurrent_ = out5_
                            generated = d_9_closedGenerated_
                            insideConstrainedOut = d_10_closedInside_
                            currentConstrainedOut = d_11_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            if ((len(currentConstrainedOut)) >= (d_3_rollbackLen_)) and (not(d_8_completeNow_)):
                                d_12_repaired_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_4_boundaryToken_)
                                d_12_repaired_ = out6_
                                if (len(d_12_repaired_)) < (len(currentConstrainedOut)):
                                    d_13_trim_: int
                                    d_13_trim_ = (len(currentConstrainedOut)) - (len(d_12_repaired_))
                                    generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (d_13_trim_):])
                                    currentConstrainedOut = d_12_repaired_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_14_next_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                    d_14_next_ = out7_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_14_next_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_15_appendedGenerated_: _dafny.Seq
                                        d_16_appendedInside_: bool
                                        d_17_appendedCurrent_: _dafny.Seq
                                        out8_: _dafny.Seq
                                        out9_: bool
                                        out10_: _dafny.Seq
                                        out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                        d_15_appendedGenerated_ = out8_
                                        d_16_appendedInside_ = out9_
                                        d_17_appendedCurrent_ = out10_
                                        generated = d_15_appendedGenerated_
                                        insideConstrainedOut = d_16_appendedInside_
                                        currentConstrainedOut = d_17_appendedCurrent_
                            elif True:
                                d_18_next2_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_18_next2_ = out11_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_18_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_19_appendedGenerated2_: _dafny.Seq
                                    d_20_appendedInside2_: bool
                                    d_21_appendedCurrent2_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next2_)
                                    d_19_appendedGenerated2_ = out12_
                                    d_20_appendedInside2_ = out13_
                                    d_21_appendedCurrent2_ = out14_
                                    generated = d_19_appendedGenerated2_
                                    insideConstrainedOut = d_20_appendedInside2_
                                    currentConstrainedOut = d_21_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


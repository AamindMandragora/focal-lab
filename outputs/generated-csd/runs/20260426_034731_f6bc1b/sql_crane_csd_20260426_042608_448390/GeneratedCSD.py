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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_stop_: bool
        d_2_stop_ = False
        while ((d_1_steps_) < (maxSteps)) and (not(d_2_stop_)):
            if not(insideConstrainedOut):
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
                d_6_completeNow_: bool
                d_6_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                d_7_validCount_: int
                out3_: int
                out3_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                d_7_validCount_ = out3_
                if (d_6_completeNow_) and ((d_7_validCount_) == (0)):
                    d_2_stop_ = True
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_8_candidates_: _dafny.Seq
                    out4_: _dafny.Seq
                    out4_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 1, eosToken)
                    d_8_candidates_ = out4_
                    if (len(d_8_candidates_)) > (0):
                        d_9_next_: _dafny.Seq
                        d_9_next_ = (d_8_candidates_)[0]
                        if (d_9_next_) == (eosToken):
                            d_2_stop_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if d_6_completeNow_:
                                d_2_stop_ = True
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_10_appendedGenerated2_: _dafny.Seq
                                d_11_appendedInside2_: bool
                                d_12_appendedCurrent2_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                                d_10_appendedGenerated2_ = out5_
                                d_11_appendedInside2_ = out6_
                                d_12_appendedCurrent2_ = out7_
                                generated = d_10_appendedGenerated2_
                                insideConstrainedOut = d_11_appendedInside2_
                                currentConstrainedOut = d_12_appendedCurrent2_
                                d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_13_next2_: _dafny.Seq
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                        d_13_next2_ = out8_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_13_next2_) == (eosToken):
                            d_2_stop_ = True
                        elif True:
                            if not(d_6_completeNow_):
                                d_14_appendedGenerated3_: _dafny.Seq
                                d_15_appendedInside3_: bool
                                d_16_appendedCurrent3_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next2_)
                                d_14_appendedGenerated3_ = out9_
                                d_15_appendedInside3_ = out10_
                                d_16_appendedCurrent3_ = out11_
                                generated = d_14_appendedGenerated3_
                                insideConstrainedOut = d_15_appendedInside3_
                                currentConstrainedOut = d_16_appendedCurrent3_
                            elif True:
                                d_2_stop_ = True
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_17_completeAtEnd_: bool
            d_17_completeAtEnd_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_17_completeAtEnd_:
                d_18_closedGenerated_: _dafny.Seq
                d_19_closedInside_: bool
                d_20_closedCurrent_: _dafny.Seq
                out12_: _dafny.Seq
                out13_: bool
                out14_: _dafny.Seq
                out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_18_closedGenerated_ = out12_
                d_19_closedInside_ = out13_
                d_20_closedCurrent_ = out14_
                generated = d_18_closedGenerated_
                insideConstrainedOut = d_19_closedInside_
                currentConstrainedOut = d_20_closedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


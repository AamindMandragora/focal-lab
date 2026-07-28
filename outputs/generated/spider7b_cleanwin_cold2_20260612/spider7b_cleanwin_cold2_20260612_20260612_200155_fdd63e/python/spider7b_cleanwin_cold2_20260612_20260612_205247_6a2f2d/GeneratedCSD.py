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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SQL query. Use only table and column names from the schema provided. Output well-formed SQL."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_3_openGenerated_: _dafny.Seq
            d_4_openInside_: bool
            d_5_openCurrent_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_3_openGenerated_ = out0_
            d_4_openInside_ = out1_
            d_5_openCurrent_ = out2_
            generated = d_3_openGenerated_
            insideConstrainedOut = d_4_openInside_
            currentConstrainedOut = d_5_openCurrent_
            d_2_steps_ = (d_2_steps_) + (1)
        while (d_2_steps_) < (maxSteps):
            if insideConstrainedOut:
                d_6_closedGenerated_: _dafny.Seq
                d_7_closedInside_: bool
                d_8_closedCurrent_: _dafny.Seq
                d_9_closed_: bool
                out3_: _dafny.Seq
                out4_: bool
                out5_: _dafny.Seq
                out6_: bool
                out3_, out4_, out5_, out6_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                d_6_closedGenerated_ = out3_
                d_7_closedInside_ = out4_
                d_8_closedCurrent_ = out5_
                d_9_closed_ = out6_
                if d_9_closed_:
                    generated = d_6_closedGenerated_
                    insideConstrainedOut = d_7_closedInside_
                    currentConstrainedOut = d_8_closedCurrent_
                    d_2_steps_ = (d_2_steps_) + (1)
                elif True:
                    d_10_next_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, prompt, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                    d_10_next_ = out7_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_10_next_) == (eosToken):
                        d_11_closeGenerated_: _dafny.Seq
                        d_12_closeInside_: bool
                        d_13_closeCurrent_: _dafny.Seq
                        d_14_wasClosed_: bool
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out11_: bool
                        out8_, out9_, out10_, out11_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_11_closeGenerated_ = out8_
                        d_12_closeInside_ = out9_
                        d_13_closeCurrent_ = out10_
                        d_14_wasClosed_ = out11_
                        if (d_14_wasClosed_) and ((d_2_steps_) < (maxSteps)):
                            generated = d_11_closeGenerated_
                            insideConstrainedOut = d_12_closeInside_
                            currentConstrainedOut = d_13_closeCurrent_
                            d_2_steps_ = (d_2_steps_) + (1)
                        d_2_steps_ = maxSteps
                    elif True:
                        d_15_appGenerated_: _dafny.Seq
                        d_16_appInside_: bool
                        d_17_appCurrent_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                        d_15_appGenerated_ = out12_
                        d_16_appInside_ = out13_
                        d_17_appCurrent_ = out14_
                        generated = d_15_appGenerated_
                        insideConstrainedOut = d_16_appInside_
                        currentConstrainedOut = d_17_appCurrent_
            elif True:
                d_2_steps_ = maxSteps
        cost = d_2_steps_
        if (cost) > (maxSteps):
            cost = maxSteps
        return generated, insideConstrainedOut, currentConstrainedOut, cost


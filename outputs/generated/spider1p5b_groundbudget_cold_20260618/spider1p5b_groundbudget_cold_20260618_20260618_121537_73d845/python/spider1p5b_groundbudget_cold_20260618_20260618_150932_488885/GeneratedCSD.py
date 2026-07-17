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
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_maxChunkTokens_: int
            if (maxSteps) > (20):
                d_2_maxChunkTokens_ = 20
            elif True:
                d_2_maxChunkTokens_ = maxSteps
            if not(insideConstrainedOut):
                d_3_chunkGenerated_: _dafny.Seq
                d_4_stoppedOnOpenSpan_: bool
                d_5_stoppedOnEos_: bool
                d_6_stepsUsed_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_maxChunkTokens_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_3_chunkGenerated_ = out0_
                d_4_stoppedOnOpenSpan_ = out1_
                d_5_stoppedOnEos_ = out2_
                d_6_stepsUsed_ = out3_
                generated = d_3_chunkGenerated_
                d_1_steps_ = d_6_stepsUsed_
                if d_4_stoppedOnOpenSpan_:
                    d_7_eg_: _dafny.Seq
                    d_8_ei_: bool
                    d_9_ec_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_7_eg_ = out4_
                    d_8_ei_ = out5_
                    d_9_ec_ = out6_
                    generated = d_7_eg_
                    insideConstrainedOut = d_8_ei_
                    currentConstrainedOut = d_9_ec_
                elif (not(d_5_stoppedOnEos_)) and ((d_1_steps_) < (maxSteps)):
                    d_10_og_: _dafny.Seq
                    d_11_oi_: bool
                    d_12_oc_: _dafny.Seq
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_10_og_ = out7_
                    d_11_oi_ = out8_
                    d_12_oc_ = out9_
                    generated = d_10_og_
                    insideConstrainedOut = d_11_oi_
                    currentConstrainedOut = d_12_oc_
                    d_1_steps_ = (d_1_steps_) + (1)
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_13_closeBudget_: int
                d_13_closeBudget_ = (maxSteps) - (d_1_steps_)
                d_14_cg_: _dafny.Seq
                d_15_ci_: bool
                d_16_cc_: _dafny.Seq
                out10_: _dafny.Seq
                out11_: bool
                out12_: _dafny.Seq
                out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_13_closeBudget_)
                d_14_cg_ = out10_
                d_15_ci_ = out11_
                d_16_cc_ = out12_
                generated = d_14_cg_
                insideConstrainedOut = d_15_ci_
                currentConstrainedOut = d_16_cc_
                d_1_steps_ = maxSteps
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


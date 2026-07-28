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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate SQL query. Format: SQL: <<QUERY>>"))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_chunkBudget_: int
        if (maxSteps) > (20):
            d_3_chunkBudget_ = 20
        elif True:
            d_3_chunkBudget_ = maxSteps
        if ((d_3_chunkBudget_) > (0)) and (not(insideConstrainedOut)):
            d_4_genOut_: _dafny.Seq
            d_5_stoppedOnOpenSpan_: bool
            d_6_stoppedOnEos_: bool
            d_7_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_4_genOut_ = out0_
            d_5_stoppedOnOpenSpan_ = out1_
            d_6_stoppedOnEos_ = out2_
            d_7_stepsUsed_ = out3_
            d_2_steps_ = (d_2_steps_) + (d_7_stepsUsed_)
            generated = d_4_genOut_
            if d_5_stoppedOnOpenSpan_:
                d_8_eg_: _dafny.Seq
                d_9_ei_: bool
                d_10_ec_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_8_eg_ = out4_
                d_9_ei_ = out5_
                d_10_ec_ = out6_
                generated = d_8_eg_
                insideConstrainedOut = d_9_ei_
                currentConstrainedOut = d_10_ec_
            elif d_6_stoppedOnEos_:
                cost = d_2_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_11_og_: _dafny.Seq
            d_12_oi_: bool
            d_13_oc_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_11_og_ = out7_
            d_12_oi_ = out8_
            d_13_oc_ = out9_
            d_2_steps_ = (d_2_steps_) + (1)
            generated = d_11_og_
            insideConstrainedOut = d_12_oi_
            currentConstrainedOut = d_13_oc_
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_14_closeBudget_: int
            d_14_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_15_cg_: _dafny.Seq
            d_16_ci_: bool
            d_17_cc_: _dafny.Seq
            out10_: _dafny.Seq
            out11_: bool
            out12_: _dafny.Seq
            out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudget_)
            d_15_cg_ = out10_
            d_16_ci_ = out11_
            d_17_cc_ = out12_
            generated = d_15_cg_
            insideConstrainedOut = d_16_ci_
            currentConstrainedOut = d_17_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


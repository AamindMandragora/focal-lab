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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output format must be exactly: SQL: <<SELECT ...>> — a single SQL query inside << and >>. Do not use markdown, backticks, or explanations. Write only: SQL: <<your SQL query here>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_phase1Budget_: int
        d_3_phase1Budget_ = 20
        if (d_3_phase1Budget_) > (maxSteps):
            d_3_phase1Budget_ = maxSteps
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_4_chunkBudget_: int
            if (d_3_phase1Budget_) < ((maxSteps) - (d_2_steps_)):
                d_4_chunkBudget_ = d_3_phase1Budget_
            elif True:
                d_4_chunkBudget_ = (maxSteps) - (d_2_steps_)
            d_5_generatedOut1_: _dafny.Seq
            d_6_stoppedOnOpenSpan1_: bool
            d_7_stoppedOnEos1_: bool
            d_8_stepsUsed1_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_5_generatedOut1_ = out0_
            d_6_stoppedOnOpenSpan1_ = out1_
            d_7_stoppedOnEos1_ = out2_
            d_8_stepsUsed1_ = out3_
            d_2_steps_ = (d_2_steps_) + (d_8_stepsUsed1_)
            generated = d_5_generatedOut1_
            if d_6_stoppedOnOpenSpan1_:
                d_9_eg_: _dafny.Seq
                d_10_ei_: bool
                d_11_ec_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_9_eg_ = out4_
                d_10_ei_ = out5_
                d_11_ec_ = out6_
                generated = d_9_eg_
                insideConstrainedOut = d_10_ei_
                currentConstrainedOut = d_11_ec_
            elif d_7_stoppedOnEos1_:
                cost = d_2_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
        d_12_loopBudget_: int
        d_12_loopBudget_ = 30
        d_13_loopSteps_: int
        d_13_loopSteps_ = 0
        while ((not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps))) and ((d_13_loopSteps_) < (d_12_loopBudget_)):
            d_14_next_: _dafny.Seq
            out7_: _dafny.Seq
            out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_14_next_ = out7_
            d_2_steps_ = (d_2_steps_) + (1)
            d_13_loopSteps_ = (d_13_loopSteps_) + (1)
            if (d_14_next_) == (eosToken):
                cost = d_2_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next_]))
            if (d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                insideConstrainedOut = True
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_15_og_: _dafny.Seq
            d_16_oi_: bool
            d_17_oc_: _dafny.Seq
            out8_: _dafny.Seq
            out9_: bool
            out10_: _dafny.Seq
            out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_15_og_ = out8_
            d_16_oi_ = out9_
            d_17_oc_ = out10_
            generated = d_15_og_
            insideConstrainedOut = d_16_oi_
            currentConstrainedOut = d_17_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_18_remainingBudget_: int
            d_18_remainingBudget_ = (maxSteps) - (d_2_steps_)
            d_19_cg_: _dafny.Seq
            d_20_ci_: bool
            d_21_cc_: _dafny.Seq
            out11_: _dafny.Seq
            out12_: bool
            out13_: _dafny.Seq
            out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_remainingBudget_)
            d_19_cg_ = out11_
            d_20_ci_ = out12_
            d_21_cc_ = out13_
            generated = d_19_cg_
            insideConstrainedOut = d_20_ci_
            currentConstrainedOut = d_21_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


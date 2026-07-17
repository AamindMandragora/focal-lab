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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SQL query. Output format: SQL: <<SELECT ...>>")))
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_2_chunkBudget_: int
            d_2_chunkBudget_ = (maxSteps) - (d_1_steps_)
            if (d_2_chunkBudget_) > (60):
                d_2_chunkBudget_ = 60
            d_3_genOut_: _dafny.Seq
            d_4_stoppedOnOpen_: bool
            d_5_stoppedOnEos_: bool
            d_6_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_3_genOut_ = out0_
            d_4_stoppedOnOpen_ = out1_
            d_5_stoppedOnEos_ = out2_
            d_6_stepsUsed_ = out3_
            generated = d_3_genOut_
            d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
            if d_5_stoppedOnEos_:
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            if d_4_stoppedOnOpen_:
                d_7_g2_: _dafny.Seq
                d_8_i2_: bool
                d_9_c2_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_7_g2_ = out4_
                d_8_i2_ = out5_
                d_9_c2_ = out6_
                generated = d_7_g2_
                insideConstrainedOut = d_8_i2_
                currentConstrainedOut = d_9_c2_
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_10_g2_: _dafny.Seq
            d_11_i2_: bool
            d_12_c2_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_10_g2_ = out7_
            d_11_i2_ = out8_
            d_12_c2_ = out9_
            generated = d_10_g2_
            insideConstrainedOut = d_11_i2_
            currentConstrainedOut = d_12_c2_
            d_1_steps_ = (d_1_steps_) + (1)
        d_13_innerSteps_: int
        d_13_innerSteps_ = 0
        d_14_maxInnerSteps_: int
        if (maxSteps) > (d_1_steps_):
            d_14_maxInnerSteps_ = (maxSteps) - (d_1_steps_)
        elif True:
            d_14_maxInnerSteps_ = 0
        with _dafny.label("0"):
            while ((d_13_innerSteps_) < (d_14_maxInnerSteps_)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_15_cg_: _dafny.Seq
                    d_16_ci_: bool
                    d_17_cc_: _dafny.Seq
                    d_18_closed_: bool
                    out10_: _dafny.Seq
                    out11_: bool
                    out12_: _dafny.Seq
                    out13_: bool
                    out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_15_cg_ = out10_
                    d_16_ci_ = out11_
                    d_17_cc_ = out12_
                    d_18_closed_ = out13_
                    d_13_innerSteps_ = (d_13_innerSteps_) + (1)
                    if d_18_closed_:
                        generated = d_15_cg_
                        insideConstrainedOut = d_16_ci_
                        currentConstrainedOut = d_17_cc_
                        raise _dafny.Break("0")
                    if (d_13_innerSteps_) >= (d_14_maxInnerSteps_):
                        raise _dafny.Break("0")
                    d_19_constrainedPrompt_: _dafny.Seq
                    d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_20_next_: _dafny.Seq
                    out14_: _dafny.Seq
                    out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                    d_20_next_ = out14_
                    d_13_innerSteps_ = (d_13_innerSteps_) + (1)
                    if (d_20_next_) == (eosToken):
                        raise _dafny.Break("0")
                    d_21_valid_: bool
                    out15_: bool
                    out15_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_20_next_)
                    d_21_valid_ = out15_
                    if d_21_valid_:
                        d_22_ag_: _dafny.Seq
                        d_23_ai_: bool
                        d_24_ac_: _dafny.Seq
                        out16_: _dafny.Seq
                        out17_: bool
                        out18_: _dafny.Seq
                        out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                        d_22_ag_ = out16_
                        d_23_ai_ = out17_
                        d_24_ac_ = out18_
                        generated = d_22_ag_
                        insideConstrainedOut = d_23_ai_
                        currentConstrainedOut = d_24_ac_
                    pass
            pass
        d_1_steps_ = (d_1_steps_) + (d_13_innerSteps_)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_25_closeBudget_: int
            d_25_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_26_cg_: _dafny.Seq
            d_27_ci_: bool
            d_28_cc_: _dafny.Seq
            out19_: _dafny.Seq
            out20_: bool
            out21_: _dafny.Seq
            out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_25_closeBudget_)
            d_26_cg_ = out19_
            d_27_ci_ = out20_
            d_28_cc_ = out21_
            generated = d_26_cg_
            insideConstrainedOut = d_27_ci_
            currentConstrainedOut = d_28_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


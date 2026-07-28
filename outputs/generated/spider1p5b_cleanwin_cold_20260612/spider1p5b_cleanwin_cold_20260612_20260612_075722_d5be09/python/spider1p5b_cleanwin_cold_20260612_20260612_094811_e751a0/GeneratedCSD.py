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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write a single concise SQL query. Output exactly: SQL: <<SELECT ...>> Use simple WHERE clauses instead of JOINs when possible. No explanation.")))
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_2_chunkMax_: int
            d_2_chunkMax_ = 6
            if (d_2_chunkMax_) > ((maxSteps) - (d_1_steps_)):
                d_2_chunkMax_ = (maxSteps) - (d_1_steps_)
            if (d_2_chunkMax_) > (0):
                d_3_genOut_: _dafny.Seq
                d_4_stoppedOnOpen_: bool
                d_5_stoppedOnEos_: bool
                d_6_stepsUsed_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkMax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_3_genOut_ = out0_
                d_4_stoppedOnOpen_ = out1_
                d_5_stoppedOnEos_ = out2_
                d_6_stepsUsed_ = out3_
                d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                generated = d_3_genOut_
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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    if ((len(currentConstrainedOut)) >= (55)) and (((d_1_steps_) + (2)) <= (maxSteps)):
                        d_13_rg_: _dafny.Seq
                        d_14_rc_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_13_rg_ = out10_
                        d_14_rc_ = out11_
                        if (parser).IsCompletePrefix(d_14_rc_):
                            generated = d_13_rg_
                            currentConstrainedOut = d_14_rc_
                            insideConstrainedOut = True
                            d_15_cg3_: _dafny.Seq
                            d_16_ci3_: bool
                            d_17_cc3_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_15_cg3_ = out12_
                            d_16_ci3_ = out13_
                            d_17_cc3_ = out14_
                            generated = d_15_cg3_
                            insideConstrainedOut = d_16_ci3_
                            currentConstrainedOut = d_17_cc3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                    d_18_cg_: _dafny.Seq
                    d_19_ci_: bool
                    d_20_cc_: _dafny.Seq
                    d_21_closed_: bool
                    out15_: _dafny.Seq
                    out16_: bool
                    out17_: _dafny.Seq
                    out18_: bool
                    out15_, out16_, out17_, out18_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_18_cg_ = out15_
                    d_19_ci_ = out16_
                    d_20_cc_ = out17_
                    d_21_closed_ = out18_
                    if d_21_closed_:
                        generated = d_18_cg_
                        insideConstrainedOut = d_19_ci_
                        currentConstrainedOut = d_20_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    if (d_1_steps_) >= (maxSteps):
                        raise _dafny.Break("0")
                    d_22_constrainedPrompt_: _dafny.Seq
                    d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_23_next_: _dafny.Seq
                    out19_: _dafny.Seq
                    out19_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                    d_23_next_ = out19_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_23_next_) == (eosToken):
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_24_cg2_: _dafny.Seq
                            d_25_ci2_: bool
                            d_26_cc2_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_24_cg2_ = out20_
                            d_25_ci2_ = out21_
                            d_26_cc2_ = out22_
                            generated = d_24_cg2_
                            insideConstrainedOut = d_25_ci2_
                            currentConstrainedOut = d_26_cc2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_27_ag_: _dafny.Seq
                        d_28_ai_: bool
                        d_29_ac_: _dafny.Seq
                        out23_: _dafny.Seq
                        out24_: bool
                        out25_: _dafny.Seq
                        out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                        d_27_ag_ = out23_
                        d_28_ai_ = out24_
                        d_29_ac_ = out25_
                        generated = d_27_ag_
                        insideConstrainedOut = d_28_ai_
                        currentConstrainedOut = d_29_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_30_rg_: _dafny.Seq
            d_31_rc_: _dafny.Seq
            out26_: _dafny.Seq
            out27_: _dafny.Seq
            out26_, out27_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_30_rg_ = out26_
            d_31_rc_ = out27_
            if (parser).IsCompletePrefix(d_31_rc_):
                generated = d_30_rg_
                currentConstrainedOut = d_31_rc_
                insideConstrainedOut = True
                if (d_1_steps_) < (maxSteps):
                    d_32_cg3_: _dafny.Seq
                    d_33_ci3_: bool
                    d_34_cc3_: _dafny.Seq
                    out28_: _dafny.Seq
                    out29_: bool
                    out30_: _dafny.Seq
                    out28_, out29_, out30_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_32_cg3_ = out28_
                    d_33_ci3_ = out29_
                    d_34_cc3_ = out30_
                    generated = d_32_cg3_
                    insideConstrainedOut = d_33_ci3_
                    currentConstrainedOut = d_34_cc3_
                    d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


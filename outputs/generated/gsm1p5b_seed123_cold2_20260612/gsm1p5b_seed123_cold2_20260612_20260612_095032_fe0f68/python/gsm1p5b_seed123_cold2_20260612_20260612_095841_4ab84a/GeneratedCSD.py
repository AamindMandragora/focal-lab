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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. For each calculation, write the arithmetic expression and result inside << >> delimiters, like <<3+2=5>>. Keep expressions inside << >> to a single concrete arithmetic formula with numbers only. The final answer must also be inside << >>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_chunkSize_: int
        d_3_chunkSize_ = 6
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_2_steps_)
                        d_5_thisChunk_: int
                        if (d_4_remaining_) < (d_3_chunkSize_):
                            d_5_thisChunk_ = d_4_remaining_
                        elif True:
                            d_5_thisChunk_ = d_3_chunkSize_
                        if (d_5_thisChunk_) == (0):
                            raise _dafny.Break("0")
                        d_6_genOut_: _dafny.Seq
                        d_7_stoppedOnOpenSpan_: bool
                        d_8_stoppedOnEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_thisChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_genOut_ = out0_
                        d_7_stoppedOnOpenSpan_ = out1_
                        d_8_stoppedOnEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        d_2_steps_ = (d_2_steps_) + (d_9_stepsUsed_)
                        generated = d_6_genOut_
                        if d_8_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_7_stoppedOnOpenSpan_:
                            d_10_eg_: _dafny.Seq
                            d_11_ei_: bool
                            d_12_ec_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_10_eg_ = out4_
                            d_11_ei_ = out5_
                            d_12_ec_ = out6_
                            generated = d_10_eg_
                            insideConstrainedOut = d_11_ei_
                            currentConstrainedOut = d_12_ec_
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_13_cg_: _dafny.Seq
                            d_14_ci_: bool
                            d_15_cc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_13_cg_ = out7_
                            d_14_ci_ = out8_
                            d_15_cc_ = out9_
                            generated = d_13_cg_
                            insideConstrainedOut = d_14_ci_
                            currentConstrainedOut = d_15_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                        elif True:
                            d_16_spanLen_: int
                            d_16_spanLen_ = len(currentConstrainedOut)
                            d_17_maxSpanTokens_: int
                            d_17_maxSpanTokens_ = 30
                            if (d_16_spanLen_) >= (d_17_maxSpanTokens_):
                                d_18_rg_: _dafny.Seq
                                d_19_rc_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: _dafny.Seq
                                out10_, out11_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_18_rg_ = out10_
                                d_19_rc_ = out11_
                                generated = d_18_rg_
                                currentConstrainedOut = d_19_rc_
                                if (parser).IsCompletePrefix(currentConstrainedOut):
                                    d_20_cg_: _dafny.Seq
                                    d_21_ci_: bool
                                    d_22_cc_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_20_cg_ = out12_
                                    d_21_ci_ = out13_
                                    d_22_cc_ = out14_
                                    generated = d_20_cg_
                                    insideConstrainedOut = d_21_ci_
                                    currentConstrainedOut = d_22_cc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_23_constrainedPrompt_: _dafny.Seq
                                d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_24_next_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_24_next_ = out15_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_24_next_) == (eosToken):
                                    d_25_rg_: _dafny.Seq
                                    d_26_rc_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out16_, out17_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_25_rg_ = out16_
                                    d_26_rc_ = out17_
                                    generated = d_25_rg_
                                    currentConstrainedOut = d_26_rc_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                        d_27_cg_: _dafny.Seq
                                        d_28_ci_: bool
                                        d_29_cc_: _dafny.Seq
                                        out18_: _dafny.Seq
                                        out19_: bool
                                        out20_: _dafny.Seq
                                        out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_27_cg_ = out18_
                                        d_28_ci_ = out19_
                                        d_29_cc_ = out20_
                                        generated = d_27_cg_
                                        insideConstrainedOut = d_28_ci_
                                        currentConstrainedOut = d_29_cc_
                                        d_2_steps_ = (d_2_steps_) + (1)
                                    raise _dafny.Break("0")
                                elif True:
                                    d_30_ag_: _dafny.Seq
                                    d_31_ai_: bool
                                    d_32_ac_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out22_: bool
                                    out23_: _dafny.Seq
                                    out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                    d_30_ag_ = out21_
                                    d_31_ai_ = out22_
                                    d_32_ac_ = out23_
                                    generated = d_30_ag_
                                    insideConstrainedOut = d_31_ai_
                                    currentConstrainedOut = d_32_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


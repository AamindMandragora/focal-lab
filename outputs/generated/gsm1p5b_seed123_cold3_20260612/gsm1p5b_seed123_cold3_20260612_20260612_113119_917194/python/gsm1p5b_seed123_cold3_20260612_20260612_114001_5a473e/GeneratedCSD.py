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
        (d_0_helpers_).AppendTaskGuidance(lm, (((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Wrap ONLY arithmetic expressions (no curly braces, no markdown, no template variables) inside << >> delimiters. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Each << >> span must contain a single valid arithmetic expression like <<3*5+2>> or <<42>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Do not put reasoning text inside << >>. End with <<final_numeric_answer>>."))))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_maxChunkTokens_: int
        d_2_maxChunkTokens_ = 20
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remaining_: int
                        d_3_remaining_ = (maxSteps) - (d_1_steps_)
                        d_4_chunkSize_: int
                        if (d_3_remaining_) < (d_2_maxChunkTokens_):
                            d_4_chunkSize_ = d_3_remaining_
                        elif True:
                            d_4_chunkSize_ = d_2_maxChunkTokens_
                        if (d_4_chunkSize_) == (0):
                            raise _dafny.Break("0")
                        d_5_og_: _dafny.Seq
                        d_6_stoppedOnOpenSpan_: bool
                        d_7_stoppedOnEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_og_ = out0_
                        d_6_stoppedOnOpenSpan_ = out1_
                        d_7_stoppedOnEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        generated = d_5_og_
                        if d_7_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_6_stoppedOnOpenSpan_:
                            d_9_og2_: _dafny.Seq
                            d_10_oi2_: bool
                            d_11_oc2_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_9_og2_ = out4_
                            d_10_oi2_ = out5_
                            d_11_oc2_ = out6_
                            generated = d_9_og2_
                            insideConstrainedOut = d_10_oi2_
                            currentConstrainedOut = d_11_oc2_
                    elif True:
                        if (d_1_steps_) >= (maxSteps):
                            raise _dafny.Break("0")
                        d_12_cg_: _dafny.Seq
                        d_13_ci_: bool
                        d_14_cc_: _dafny.Seq
                        d_15_closed_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out10_: bool
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_12_cg_ = out7_
                        d_13_ci_ = out8_
                        d_14_cc_ = out9_
                        d_15_closed_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_15_closed_:
                            generated = d_12_cg_
                            insideConstrainedOut = d_13_ci_
                            currentConstrainedOut = d_14_cc_
                        elif True:
                            if (d_1_steps_) >= (maxSteps):
                                d_16_rg_: _dafny.Seq
                                d_17_rc_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: _dafny.Seq
                                out11_, out12_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_16_rg_ = out11_
                                d_17_rc_ = out12_
                                generated = d_16_rg_
                                currentConstrainedOut = d_17_rc_
                                if (parser).IsCompletePrefix(currentConstrainedOut):
                                    insideConstrainedOut = True
                                raise _dafny.Break("0")
                            d_18_constrainedPrompt_: _dafny.Seq
                            d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_19_next_: _dafny.Seq
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_19_next_ = out13_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_19_next_) == (eosToken):
                                d_20_rg_: _dafny.Seq
                                d_21_rc_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: _dafny.Seq
                                out14_, out15_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_20_rg_ = out14_
                                d_21_rc_ = out15_
                                generated = d_20_rg_
                                currentConstrainedOut = d_21_rc_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_22_fg_: _dafny.Seq
                                    d_23_fi_: bool
                                    d_24_fc_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_22_fg_ = out16_
                                    d_23_fi_ = out17_
                                    d_24_fc_ = out18_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    generated = d_22_fg_
                                    insideConstrainedOut = d_23_fi_
                                    currentConstrainedOut = d_24_fc_
                                elif True:
                                    insideConstrainedOut = True
                                raise _dafny.Break("0")
                            elif True:
                                d_25_ag_: _dafny.Seq
                                d_26_ai_: bool
                                d_27_ac_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                d_25_ag_ = out19_
                                d_26_ai_ = out20_
                                d_27_ac_ = out21_
                                generated = d_25_ag_
                                insideConstrainedOut = d_26_ai_
                                currentConstrainedOut = d_27_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


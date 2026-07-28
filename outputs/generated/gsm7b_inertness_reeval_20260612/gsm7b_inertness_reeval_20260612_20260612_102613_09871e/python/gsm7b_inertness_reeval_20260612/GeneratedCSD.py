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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Use << >> to wrap each symbolic expression and the final answer. Write only one expression per << >> span. Keep each << >> span short and exact."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_MAX__SPAN__TOKENS_: int
        d_3_MAX__SPAN__TOKENS_ = 40
        d_4_spanTokenCount_: int
        d_4_spanTokenCount_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_maxChunk_: int
                        if ((maxSteps) - (d_2_steps_)) >= (20):
                            d_5_maxChunk_ = 20
                        elif True:
                            d_5_maxChunk_ = (maxSteps) - (d_2_steps_)
                        if (d_5_maxChunk_) == (0):
                            raise _dafny.Break("0")
                        d_6_genOut_: _dafny.Seq
                        d_7_stoppedOnOpenSpan_: bool
                        d_8_stoppedOnEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_maxChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
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
                            d_4_spanTokenCount_ = 0
                    elif (d_4_spanTokenCount_) >= (d_3_MAX__SPAN__TOKENS_):
                        d_13_rg_: _dafny.Seq
                        d_14_rc_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: _dafny.Seq
                        out7_, out8_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_13_rg_ = out7_
                        d_14_rc_ = out8_
                        generated = d_13_rg_
                        currentConstrainedOut = d_14_rc_
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_15_cg2_: _dafny.Seq
                            d_16_ci2_: bool
                            d_17_cc2_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_15_cg2_ = out9_
                            d_16_ci2_ = out10_
                            d_17_cc2_ = out11_
                            generated = d_15_cg2_
                            insideConstrainedOut = d_16_ci2_
                            currentConstrainedOut = d_17_cc2_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_4_spanTokenCount_ = 0
                        elif True:
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_4_spanTokenCount_ = 0
                    elif True:
                        d_18_cg_: _dafny.Seq
                        d_19_ci_: bool
                        d_20_cc_: _dafny.Seq
                        d_21_closed_: bool
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out15_: bool
                        out12_, out13_, out14_, out15_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_18_cg_ = out12_
                        d_19_ci_ = out13_
                        d_20_cc_ = out14_
                        d_21_closed_ = out15_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if d_21_closed_:
                            generated = d_18_cg_
                            insideConstrainedOut = d_19_ci_
                            currentConstrainedOut = d_20_cc_
                            d_4_spanTokenCount_ = 0
                        elif True:
                            d_22_constrainedPrompt_: _dafny.Seq
                            d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_23_next_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_23_next_ = out16_
                            if (d_23_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_24_ag_: _dafny.Seq
                                d_25_ai_: bool
                                d_26_ac_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                d_24_ag_ = out17_
                                d_25_ai_ = out18_
                                d_26_ac_ = out19_
                                generated = d_24_ag_
                                insideConstrainedOut = d_25_ai_
                                currentConstrainedOut = d_26_ac_
                                d_4_spanTokenCount_ = (d_4_spanTokenCount_) + (1)
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


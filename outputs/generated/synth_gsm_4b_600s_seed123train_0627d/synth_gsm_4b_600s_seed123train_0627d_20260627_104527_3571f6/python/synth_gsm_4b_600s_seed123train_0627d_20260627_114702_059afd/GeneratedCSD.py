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
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out3_: int
        out0_, out1_, out2_, out3_ = default__.AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken)
        generated = out0_
        insideConstrainedOut = out1_
        currentConstrainedOut = out2_
        cost = out3_
        if ((maxSteps) > (0)) and ((cost) <= (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

    @staticmethod
    def AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. For each calculation, write ONLY the complete arithmetic expression inside << >> delimiters. Use only numbers, variable names, +, -, *, /, //, %, (, ) inside delimiters. The final answer must be in a << >> span. Never put text like {variable} inside << >> spans."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_closeReserve_: int
        d_3_closeReserve_ = 20
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (insideConstrainedOut) and (((maxSteps) - (d_2_steps_)) <= (d_3_closeReserve_)):
                        d_4_closeBudget_: int
                        d_4_closeBudget_ = (maxSteps) - (d_2_steps_)
                        d_5_cg_: _dafny.Seq
                        d_6_ci_: bool
                        d_7_cc_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_4_closeBudget_)
                        d_5_cg_ = out0_
                        d_6_ci_ = out1_
                        d_7_cc_ = out2_
                        generated = d_5_cg_
                        insideConstrainedOut = d_6_ci_
                        currentConstrainedOut = d_7_cc_
                        d_2_steps_ = maxSteps
                    elif not(insideConstrainedOut):
                        d_8_chunkBudget_: int
                        d_8_chunkBudget_ = (maxSteps) - (d_2_steps_)
                        if (d_8_chunkBudget_) > (40):
                            d_8_chunkBudget_ = 40
                        if (d_8_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        d_9_cg_: _dafny.Seq
                        d_10_stoppedOnOpen_: bool
                        d_11_stoppedOnEos_: bool
                        d_12_stepsUsed_: int
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: bool
                        out6_: int
                        out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_9_cg_ = out3_
                        d_10_stoppedOnOpen_ = out4_
                        d_11_stoppedOnEos_ = out5_
                        d_12_stepsUsed_ = out6_
                        d_2_steps_ = (d_2_steps_) + (d_12_stepsUsed_)
                        if d_11_stoppedOnEos_:
                            generated = d_9_cg_
                            raise _dafny.Break("0")
                        generated = d_9_cg_
                        if d_10_stoppedOnOpen_:
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            generated = out7_
                            insideConstrainedOut = out8_
                            currentConstrainedOut = out9_
                    elif True:
                        d_13_cg_: _dafny.Seq
                        d_14_ci_: bool
                        d_15_cc_: _dafny.Seq
                        d_16_closed_: bool
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out13_: bool
                        out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_13_cg_ = out10_
                        d_14_ci_ = out11_
                        d_15_cc_ = out12_
                        d_16_closed_ = out13_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if d_16_closed_:
                            generated = d_13_cg_
                            insideConstrainedOut = d_14_ci_
                            currentConstrainedOut = d_15_cc_
                        elif (d_2_steps_) < (maxSteps):
                            d_17_constrainedPrompt_: _dafny.Seq
                            d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_18_next_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_18_next_ = out14_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_18_next_) == (eosToken):
                                if (d_2_steps_) < (maxSteps):
                                    d_19_closeBudget2_: int
                                    d_19_closeBudget2_ = (maxSteps) - (d_2_steps_)
                                    d_20_cg2_: _dafny.Seq
                                    d_21_ci2_: bool
                                    d_22_cc2_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_closeBudget2_)
                                    d_20_cg2_ = out15_
                                    d_21_ci2_ = out16_
                                    d_22_cc2_ = out17_
                                    generated = d_20_cg2_
                                    insideConstrainedOut = d_21_ci2_
                                    currentConstrainedOut = d_22_cc2_
                                    d_2_steps_ = maxSteps
                                raise _dafny.Break("0")
                            elif True:
                                d_23_ag_: _dafny.Seq
                                d_24_ai_: bool
                                d_25_ac_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                d_23_ag_ = out18_
                                d_24_ai_ = out19_
                                d_25_ac_ = out20_
                                generated = d_23_ag_
                                insideConstrainedOut = d_24_ai_
                                currentConstrainedOut = d_25_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_26_closeBudget_: int
            d_26_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_27_cg_: _dafny.Seq
            d_28_ci_: bool
            d_29_cc_: _dafny.Seq
            out21_: _dafny.Seq
            out22_: bool
            out23_: _dafny.Seq
            out21_, out22_, out23_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_26_closeBudget_)
            d_27_cg_ = out21_
            d_28_ci_ = out22_
            d_29_cc_ = out23_
            generated = d_27_cg_
            insideConstrainedOut = d_28_ci_
            currentConstrainedOut = d_29_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


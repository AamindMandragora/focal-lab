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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Wrap every intermediate symbolic expression in << >> delimiters. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "For the final answer, write: The final answer is <<symbolic expression>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Inside << >> use only: numbers, problem variables, +, -, *, /, //, %, (), int(). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use int() when the result must be a whole number. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use // for integer (floor) division. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use the exact variable names given in the problem. Do not substitute numeric values for variables. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Each << >> must contain exactly ONE complete arithmetic expression."))))
        d_1_closeReserve_: int
        d_1_closeReserve_ = 150
        d_2_maxSpanLength_: int
        d_2_maxSpanLength_ = 35
        d_3_steps_: int
        d_3_steps_ = 0
        with _dafny.label("0"):
            while (d_3_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_4_remaining_: int
                    d_4_remaining_ = (maxSteps) - (d_3_steps_)
                    d_5_spanTooLong_: bool
                    d_5_spanTooLong_ = (len(currentConstrainedOut)) > (d_2_maxSpanLength_)
                    if (insideConstrainedOut) and (((d_4_remaining_) <= (d_1_closeReserve_)) or (d_5_spanTooLong_)):
                        d_6_cg_: _dafny.Seq
                        d_7_ci_: bool
                        d_8_cc_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_4_remaining_)
                        d_6_cg_ = out0_
                        d_7_ci_ = out1_
                        d_8_cc_ = out2_
                        generated = d_6_cg_
                        insideConstrainedOut = d_7_ci_
                        currentConstrainedOut = d_8_cc_
                        d_3_steps_ = maxSteps
                        raise _dafny.Break("0")
                    elif not(insideConstrainedOut):
                        d_9_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_9_next_ = out3_
                        d_3_steps_ = (d_3_steps_) + (1)
                        if (d_9_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                            if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_10_cg_: _dafny.Seq
                        d_11_ci_: bool
                        d_12_cc_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_10_cg_ = out4_
                        d_11_ci_ = out5_
                        d_12_cc_ = out6_
                        generated = d_10_cg_
                        insideConstrainedOut = d_11_ci_
                        currentConstrainedOut = d_12_cc_
                        d_3_steps_ = (d_3_steps_) + (1)
                    elif True:
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_14_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('4e0'), eosToken)
                        d_14_next_ = out7_
                        d_3_steps_ = (d_3_steps_) + (1)
                        if (d_14_next_) == (eosToken):
                            d_15_remaining2_: int
                            d_15_remaining2_ = (maxSteps) - (d_3_steps_)
                            if (d_15_remaining2_) > (0):
                                d_16_closeBudget2_: int
                                if (d_15_remaining2_) <= (30):
                                    d_16_closeBudget2_ = d_15_remaining2_
                                elif True:
                                    d_16_closeBudget2_ = 30
                                d_17_cg_: _dafny.Seq
                                d_18_ci_: bool
                                d_19_cc_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_16_closeBudget2_)
                                d_17_cg_ = out8_
                                d_18_ci_ = out9_
                                d_19_cc_ = out10_
                                generated = d_17_cg_
                                insideConstrainedOut = d_18_ci_
                                currentConstrainedOut = d_19_cc_
                                d_3_steps_ = (d_3_steps_) + (d_16_closeBudget2_)
                            raise _dafny.Break("0")
                        elif True:
                            d_20_ag_: _dafny.Seq
                            d_21_ai_: bool
                            d_22_ac_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                            d_20_ag_ = out11_
                            d_21_ai_ = out12_
                            d_22_ac_ = out13_
                            generated = d_20_ag_
                            insideConstrainedOut = d_21_ai_
                            currentConstrainedOut = d_22_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_3_steps_) < (maxSteps)):
            d_23_remaining3_: int
            d_23_remaining3_ = (maxSteps) - (d_3_steps_)
            d_24_cg_: _dafny.Seq
            d_25_ci_: bool
            d_26_cc_: _dafny.Seq
            out14_: _dafny.Seq
            out15_: bool
            out16_: _dafny.Seq
            out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_remaining3_)
            d_24_cg_ = out14_
            d_25_ci_ = out15_
            d_26_cc_ = out16_
            generated = d_24_cg_
            insideConstrainedOut = d_25_ci_
            currentConstrainedOut = d_26_cc_
            d_3_steps_ = maxSteps
        cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


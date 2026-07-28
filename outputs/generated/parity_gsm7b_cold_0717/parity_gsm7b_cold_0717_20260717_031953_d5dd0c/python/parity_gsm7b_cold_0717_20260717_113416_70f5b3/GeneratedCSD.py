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
        (d_0_helpers_).AppendTaskGuidance(lm, ((((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Wrap every intermediate symbolic expression in << >> delimiters. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "For the final answer, write: The final answer is <<symbolic expression>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Inside << >> use only: numbers, problem variables, +, -, *, /, //, %, (), int(). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use int() when the result must be a whole number. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use // for integer (floor) division. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use the exact variable names given in the problem. Do not substitute numeric values for variables. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Do NOT write assignments (no = inside << >>). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Do NOT write conditional expressions (no if/else inside << >>)."))))
        d_1_closeReserve_: int
        d_1_closeReserve_ = 80
        d_2_spanSteps_: int
        if insideConstrained:
            d_2_spanSteps_ = len(currentConstrained)
        elif True:
            d_2_spanSteps_ = 0
        d_3_steps_: int
        d_3_steps_ = 0
        with _dafny.label("0"):
            while (d_3_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_4_remaining_: int
                    d_4_remaining_ = (maxSteps) - (d_3_steps_)
                    if (insideConstrainedOut) and (((d_4_remaining_) <= (d_1_closeReserve_)) or ((d_2_spanSteps_) >= (25))):
                        d_5_cg_: _dafny.Seq
                        d_6_ci_: bool
                        d_7_cc_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_4_remaining_)
                        d_5_cg_ = out0_
                        d_6_ci_ = out1_
                        d_7_cc_ = out2_
                        generated = d_5_cg_
                        insideConstrainedOut = d_6_ci_
                        currentConstrainedOut = d_7_cc_
                        d_3_steps_ = maxSteps
                        raise _dafny.Break("0")
                    elif not(insideConstrainedOut):
                        d_8_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_8_next_ = out3_
                        d_3_steps_ = (d_3_steps_) + (1)
                        if (d_8_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                            if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_spanSteps_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_9_cg_: _dafny.Seq
                        d_10_ci_: bool
                        d_11_cc_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_9_cg_ = out4_
                        d_10_ci_ = out5_
                        d_11_cc_ = out6_
                        generated = d_9_cg_
                        insideConstrainedOut = d_10_ci_
                        currentConstrainedOut = d_11_cc_
                        d_3_steps_ = (d_3_steps_) + (1)
                        d_2_spanSteps_ = 0
                    elif True:
                        d_12_constrainedPrompt_: _dafny.Seq
                        d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_13_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('4e0'), eosToken)
                        d_13_next_ = out7_
                        d_3_steps_ = (d_3_steps_) + (1)
                        d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                        if (d_13_next_) == (eosToken):
                            if (d_3_steps_) < (maxSteps):
                                d_14_remEos_: int
                                d_14_remEos_ = (maxSteps) - (d_3_steps_)
                                d_15_cg_: _dafny.Seq
                                d_16_ci_: bool
                                d_17_cc_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_remEos_)
                                d_15_cg_ = out8_
                                d_16_ci_ = out9_
                                d_17_cc_ = out10_
                                generated = d_15_cg_
                                insideConstrainedOut = d_16_ci_
                                currentConstrainedOut = d_17_cc_
                                d_3_steps_ = maxSteps
                            raise _dafny.Break("0")
                        elif True:
                            d_18_ag_: _dafny.Seq
                            d_19_ai_: bool
                            d_20_ac_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                            d_18_ag_ = out11_
                            d_19_ai_ = out12_
                            d_20_ac_ = out13_
                            generated = d_18_ag_
                            insideConstrainedOut = d_19_ai_
                            currentConstrainedOut = d_20_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_3_steps_) < (maxSteps)):
            d_21_remaining_: int
            d_21_remaining_ = (maxSteps) - (d_3_steps_)
            d_22_cg_: _dafny.Seq
            d_23_ci_: bool
            d_24_cc_: _dafny.Seq
            out14_: _dafny.Seq
            out15_: bool
            out16_: _dafny.Seq
            out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_21_remaining_)
            d_22_cg_ = out14_
            d_23_ci_ = out15_
            d_24_cc_ = out16_
            generated = d_22_cg_
            insideConstrainedOut = d_23_ci_
            currentConstrainedOut = d_24_cc_
            d_3_steps_ = maxSteps
        cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


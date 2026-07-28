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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve grade-school math step by step using symbolic variables. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Wrap each intermediate symbolic expression in << >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "The final line MUST be exactly: The final answer is <<expr>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Inside << >>: use only the original variable names from the problem, numbers, ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+, -, *, /, //, %, (), and int(). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Do NOT write = inside << >>. Keep each << >> expression short and complete. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use int() for whole-number (floor) division results. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use // for integer division. Never substitute numeric values for symbolic variables."))))
        d_1_closeReserve_: int
        d_1_closeReserve_ = 200
        d_2_deadEndMin_: int
        d_2_deadEndMin_ = 4
        d_3_penaltyToks_: _dafny.Seq
        d_3_penaltyToks_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!"))])
        d_4_steps_: int
        d_4_steps_ = 0
        with _dafny.label("0"):
            while (d_4_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_5_remaining_: int
                    d_5_remaining_ = (maxSteps) - (d_4_steps_)
                    if (insideConstrainedOut) and ((d_5_remaining_) <= (d_1_closeReserve_)):
                        d_6_cg_: _dafny.Seq
                        d_7_ci_: bool
                        d_8_cc_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_5_remaining_)
                        d_6_cg_ = out0_
                        d_7_ci_ = out1_
                        d_8_cc_ = out2_
                        generated = d_6_cg_
                        insideConstrainedOut = d_7_ci_
                        currentConstrainedOut = d_8_cc_
                        d_4_steps_ = maxSteps
                        raise _dafny.Break("0")
                    elif not(insideConstrainedOut):
                        d_9_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_9_next_ = out3_
                        d_4_steps_ = (d_4_steps_) + (1)
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
                        d_4_steps_ = (d_4_steps_) + (1)
                    elif True:
                        d_13_narrow_: bool
                        out7_: bool
                        out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, d_2_deadEndMin_)
                        d_13_narrow_ = out7_
                        if d_13_narrow_:
                            d_14_cg_: _dafny.Seq
                            d_15_ci_: bool
                            d_16_cc_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_5_remaining_)
                            d_14_cg_ = out8_
                            d_15_ci_ = out9_
                            d_16_cc_ = out10_
                            generated = d_14_cg_
                            insideConstrainedOut = d_15_ci_
                            currentConstrainedOut = d_16_cc_
                            d_4_steps_ = maxSteps
                            raise _dafny.Break("0")
                        elif True:
                            d_17_constrainedPrompt_: _dafny.Seq
                            d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_18_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_penaltyToks_, _dafny.BigRational('8e0'), 12, eosToken)
                            d_18_next_ = out11_
                            d_4_steps_ = (d_4_steps_) + (1)
                            if (d_18_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_19_ag_: _dafny.Seq
                                d_20_ai_: bool
                                d_21_ac_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                d_19_ag_ = out12_
                                d_20_ai_ = out13_
                                d_21_ac_ = out14_
                                generated = d_19_ag_
                                insideConstrainedOut = d_20_ai_
                                currentConstrainedOut = d_21_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_4_steps_) < (maxSteps)):
            d_22_remaining2_: int
            d_22_remaining2_ = (maxSteps) - (d_4_steps_)
            d_23_cg_: _dafny.Seq
            d_24_ci_: bool
            d_25_cc_: _dafny.Seq
            out15_: _dafny.Seq
            out16_: bool
            out17_: _dafny.Seq
            out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_remaining2_)
            d_23_cg_ = out15_
            d_24_ci_ = out16_
            d_25_cc_ = out17_
            generated = d_23_cg_
            insideConstrainedOut = d_24_ci_
            currentConstrainedOut = d_25_cc_
            d_4_steps_ = maxSteps
        cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


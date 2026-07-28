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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step using the symbolic variable names "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "exactly as given in the problem. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Identify ALL quantities and how they combine. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write intermediate sub-results inside << >> as you go. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "At the very end, output the single COMPLETE symbolic expression for the ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final answer inside << >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "The final expression MUST combine all relevant variables with arithmetic ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "operations (+, -, *, /, //, %, int()); do not write just one variable name."))))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeBudget_: int
        if (maxSteps) > (200):
            d_2_freeBudget_ = (maxSteps) - (200)
        elif True:
            d_2_freeBudget_ = _dafny.euclidian_division(maxSteps, 2)
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_2_freeBudget_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_3_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_3_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_3_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                    if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_4_og_: _dafny.Seq
            d_5_oi_: bool
            d_6_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_4_og_ = out1_
            d_5_oi_ = out2_
            d_6_oc_ = out3_
            generated = d_4_og_
            insideConstrainedOut = d_5_oi_
            currentConstrainedOut = d_6_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("1"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    d_7_constrainedPrompt_: _dafny.Seq
                    d_7_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_8_next_: _dafny.Seq
                    out4_: _dafny.Seq
                    out4_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                    d_8_next_ = out4_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_8_next_) == (eosToken):
                        d_9_rg_: _dafny.Seq
                        d_10_rc_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: _dafny.Seq
                        out5_, out6_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_9_rg_ = out5_
                        d_10_rc_ = out6_
                        generated = d_9_rg_
                        currentConstrainedOut = d_10_rc_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_11_cg_: _dafny.Seq
                            d_12_ci_: bool
                            d_13_cc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_cg_ = out7_
                            d_12_ci_ = out8_
                            d_13_cc_ = out9_
                            generated = d_11_cg_
                            insideConstrainedOut = d_12_ci_
                            currentConstrainedOut = d_13_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("1")
                    d_14_ag_: _dafny.Seq
                    d_15_ai_: bool
                    d_16_ac_: _dafny.Seq
                    out10_: _dafny.Seq
                    out11_: bool
                    out12_: _dafny.Seq
                    out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_next_)
                    d_14_ag_ = out10_
                    d_15_ai_ = out11_
                    d_16_ac_ = out12_
                    generated = d_14_ag_
                    insideConstrainedOut = d_15_ai_
                    currentConstrainedOut = d_16_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_17_rg_: _dafny.Seq
            d_18_rc_: _dafny.Seq
            out13_: _dafny.Seq
            out14_: _dafny.Seq
            out13_, out14_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_17_rg_ = out13_
            d_18_rc_ = out14_
            generated = d_17_rg_
            currentConstrainedOut = d_18_rc_
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_19_cg_: _dafny.Seq
                d_20_ci_: bool
                d_21_cc_: _dafny.Seq
                out15_: _dafny.Seq
                out16_: bool
                out17_: _dafny.Seq
                out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_19_cg_ = out15_
                d_20_ci_ = out16_
                d_21_cc_ = out17_
                generated = d_19_cg_
                insideConstrainedOut = d_20_ci_
                currentConstrainedOut = d_21_cc_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


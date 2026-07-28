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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step in plain prose. Do NOT use << >> delimiters during your reasoning — write all intermediate steps as plain text. Only at the very end write exactly: The final answer is <<FORMULA>> where FORMULA is a single complete Python arithmetic expression for the final numeric answer. Use ONLY the numeric variable names that appear in curly braces in the problem (e.g. n, n1, w1, price). Do NOT include string-type variables (unit labels, currency symbols) in FORMULA. Your FORMULA must combine ALL necessary variables into ONE complete expression, not just an intermediate sub-result."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                        d_4_cnt_: int
                        out1_: int
                        out1_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, d_3_next_)
                        d_4_cnt_ = out1_
                        if (d_4_cnt_) > (30):
                            raise _dafny.Break("0")
                        if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_5_cg_: _dafny.Seq
                        d_6_ci_: bool
                        d_7_cc_: _dafny.Seq
                        out2_: _dafny.Seq
                        out3_: bool
                        out4_: _dafny.Seq
                        out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_5_cg_ = out2_
                        d_6_ci_ = out3_
                        d_7_cc_ = out4_
                        generated = d_5_cg_
                        insideConstrainedOut = d_6_ci_
                        currentConstrainedOut = d_7_cc_
                        d_2_steps_ = (d_2_steps_) + (1)
                    elif True:
                        d_8_constrainedPrompt_: _dafny.Seq
                        d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_9_next_: _dafny.Seq
                        out5_: _dafny.Seq
                        out5_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_9_next_ = out5_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_9_next_) == (eosToken):
                            raise _dafny.Break("0")
                        d_10_ag_: _dafny.Seq
                        d_11_ai_: bool
                        d_12_ac_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                        d_10_ag_ = out6_
                        d_11_ai_ = out7_
                        d_12_ac_ = out8_
                        generated = d_10_ag_
                        insideConstrainedOut = d_11_ai_
                        currentConstrainedOut = d_12_ac_
                    pass
            pass
        if insideConstrainedOut:
            d_13_rg_: _dafny.Seq
            d_14_rc_: _dafny.Seq
            out9_: _dafny.Seq
            out10_: _dafny.Seq
            out9_, out10_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_13_rg_ = out9_
            d_14_rc_ = out10_
            generated = d_13_rg_
            currentConstrainedOut = d_14_rc_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                d_15_cg_: _dafny.Seq
                d_16_ci_: bool
                d_17_cc_: _dafny.Seq
                out11_: _dafny.Seq
                out12_: bool
                out13_: _dafny.Seq
                out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_15_cg_ = out11_
                d_16_ci_ = out12_
                d_17_cc_ = out13_
                generated = d_15_cg_
                insideConstrainedOut = d_16_ci_
                currentConstrainedOut = d_17_cc_
                d_2_steps_ = (d_2_steps_) + (1)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


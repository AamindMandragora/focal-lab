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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. At the very end, write 'The answer is <<EXPR>>' where EXPR is a plain arithmetic expression using only numbers, variable names, +, -, *, /, //, %, (, ), and no LaTeX, no curly braces, no backslashes. Example: The answer is <<n * p * (1 + r / 100)>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_freeBudget_: int
        d_3_freeBudget_ = _dafny.euclidian_division((maxSteps) * (3), 4)
        if ((d_3_freeBudget_) == (0)) and ((maxSteps) > (0)):
            d_3_freeBudget_ = 1
        with _dafny.label("0"):
            while ((d_2_steps_) < (d_3_freeBudget_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_4_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_4_next_ = out0_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_4_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                        if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        with _dafny.label("1"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    d_5_cg_: _dafny.Seq
                    d_6_ci_: bool
                    d_7_cc_: _dafny.Seq
                    d_8_closed_: bool
                    out1_: _dafny.Seq
                    out2_: bool
                    out3_: _dafny.Seq
                    out4_: bool
                    out1_, out2_, out3_, out4_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_5_cg_ = out1_
                    d_6_ci_ = out2_
                    d_7_cc_ = out3_
                    d_8_closed_ = out4_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if d_8_closed_:
                        generated = d_5_cg_
                        insideConstrainedOut = d_6_ci_
                        currentConstrainedOut = d_7_cc_
                    elif True:
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_10_next_: _dafny.Seq
                        out5_: _dafny.Seq
                        out5_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e-1'), eosToken)
                        d_10_next_ = out5_
                        if (d_10_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_11_ag_: _dafny.Seq
                            d_12_ai_: bool
                            d_13_ac_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                            d_11_ag_ = out6_
                            d_12_ai_ = out7_
                            d_13_ac_ = out8_
                            generated = d_11_ag_
                            insideConstrainedOut = d_12_ai_
                            currentConstrainedOut = d_13_ac_
                    pass
            pass
        with _dafny.label("2"):
            while ((d_2_steps_) < (d_3_freeBudget_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("2"):
                    d_14_next_: _dafny.Seq
                    out9_: _dafny.Seq
                    out9_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_14_next_ = out9_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_14_next_) == (eosToken):
                        raise _dafny.Break("2")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next_]))
                        if (d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        with _dafny.label("3"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("3"):
                    d_15_cg2_: _dafny.Seq
                    d_16_ci2_: bool
                    d_17_cc2_: _dafny.Seq
                    d_18_closed2_: bool
                    out10_: _dafny.Seq
                    out11_: bool
                    out12_: _dafny.Seq
                    out13_: bool
                    out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_15_cg2_ = out10_
                    d_16_ci2_ = out11_
                    d_17_cc2_ = out12_
                    d_18_closed2_ = out13_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if d_18_closed2_:
                        generated = d_15_cg2_
                        insideConstrainedOut = d_16_ci2_
                        currentConstrainedOut = d_17_cc2_
                    elif True:
                        d_19_constrainedPrompt2_: _dafny.Seq
                        d_19_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_20_next2_: _dafny.Seq
                        out14_: _dafny.Seq
                        out14_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_19_constrainedPrompt2_, currentConstrainedOut, _dafny.BigRational('8e-1'), eosToken)
                        d_20_next2_ = out14_
                        if (d_20_next2_) == (eosToken):
                            raise _dafny.Break("3")
                        elif True:
                            d_21_ag2_: _dafny.Seq
                            d_22_ai2_: bool
                            d_23_ac2_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next2_)
                            d_21_ag2_ = out15_
                            d_22_ai2_ = out16_
                            d_23_ac2_ = out17_
                            generated = d_21_ag2_
                            insideConstrainedOut = d_22_ai2_
                            currentConstrainedOut = d_23_ac2_
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_24_openCount_: int
            out18_: int
            out18_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
            d_24_openCount_ = out18_
            d_25_closeCount_: int
            out19_: int
            out19_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
            d_25_closeCount_ = out19_
            if ((d_24_openCount_) == (0)) or ((d_24_openCount_) > (d_25_closeCount_)):
                if ((d_2_steps_) + (2)) <= (maxSteps):
                    d_26_fg_: _dafny.Seq
                    d_27_fi_: bool
                    d_28_fc_: _dafny.Seq
                    out20_: _dafny.Seq
                    out21_: bool
                    out22_: _dafny.Seq
                    out20_, out21_, out22_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_26_fg_ = out20_
                    d_27_fi_ = out21_
                    d_28_fc_ = out22_
                    generated = d_26_fg_
                    insideConstrainedOut = d_27_fi_
                    currentConstrainedOut = d_28_fc_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_29_remainBudget_: int
                    d_29_remainBudget_ = (maxSteps) - (d_2_steps_)
                    if (d_29_remainBudget_) > (0):
                        d_30_wg_: _dafny.Seq
                        d_31_wi_: bool
                        d_32_wc_: _dafny.Seq
                        out23_: _dafny.Seq
                        out24_: bool
                        out25_: _dafny.Seq
                        out23_, out24_, out25_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_29_remainBudget_)
                        d_30_wg_ = out23_
                        d_31_wi_ = out24_
                        d_32_wc_ = out25_
                        generated = d_30_wg_
                        insideConstrainedOut = d_31_wi_
                        currentConstrainedOut = d_32_wc_
                        d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


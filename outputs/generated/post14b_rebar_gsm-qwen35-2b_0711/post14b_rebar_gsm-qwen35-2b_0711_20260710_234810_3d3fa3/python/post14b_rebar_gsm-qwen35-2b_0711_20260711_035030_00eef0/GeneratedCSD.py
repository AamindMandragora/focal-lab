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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write your final answer as exactly: The answer is <<EXPR>> where EXPR uses only variable names, numbers, +, -, *, /, //, %, (, ). No LaTeX, no {}, no **, no backslashes. Close >> immediately after the expression. Example: The answer is <<n * (p + 1)>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_phase1Budget_: int
        d_3_phase1Budget_ = _dafny.euclidian_division((maxSteps) * (7), 10)
        if ((d_3_phase1Budget_) == (0)) and ((maxSteps) > (0)):
            d_3_phase1Budget_ = 1
        with _dafny.label("0"):
            while ((d_2_steps_) < (d_3_phase1Budget_)) and (not(insideConstrainedOut)):
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
                        raise _dafny.Break("1")
                    elif True:
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_10_next_: _dafny.Seq
                        out5_: _dafny.Seq
                        out5_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_10_next_ = out5_
                        if (d_10_next_) == (eosToken):
                            d_11_remainForClose_: int
                            d_11_remainForClose_ = (maxSteps) - (d_2_steps_)
                            if (d_11_remainForClose_) > (0):
                                d_12_wg_: _dafny.Seq
                                d_13_wi_: bool
                                d_14_wc_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out6_, out7_, out8_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_remainForClose_)
                                d_12_wg_ = out6_
                                d_13_wi_ = out7_
                                d_14_wc_ = out8_
                                generated = d_12_wg_
                                insideConstrainedOut = d_13_wi_
                                currentConstrainedOut = d_14_wc_
                                d_2_steps_ = (d_2_steps_) + (d_11_remainForClose_)
                            raise _dafny.Break("1")
                        elif True:
                            d_15_ag_: _dafny.Seq
                            d_16_ai_: bool
                            d_17_ac_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                            d_15_ag_ = out9_
                            d_16_ai_ = out10_
                            d_17_ac_ = out11_
                            generated = d_15_ag_
                            insideConstrainedOut = d_16_ai_
                            currentConstrainedOut = d_17_ac_
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_18_openCount_: int
            out12_: int
            out12_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
            d_18_openCount_ = out12_
            d_19_closeCount_: int
            out13_: int
            out13_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
            d_19_closeCount_ = out13_
            if ((d_18_openCount_) == (0)) or ((d_18_openCount_) > (d_19_closeCount_)):
                if ((d_2_steps_) + (3)) <= (maxSteps):
                    d_20_fg_: _dafny.Seq
                    d_21_fi_: bool
                    d_22_fc_: _dafny.Seq
                    out14_: _dafny.Seq
                    out15_: bool
                    out16_: _dafny.Seq
                    out14_, out15_, out16_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_20_fg_ = out14_
                    d_21_fi_ = out15_
                    d_22_fc_ = out16_
                    generated = d_20_fg_
                    insideConstrainedOut = d_21_fi_
                    currentConstrainedOut = d_22_fc_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_23_remainBudget_: int
                    d_23_remainBudget_ = (maxSteps) - (d_2_steps_)
                    if (d_23_remainBudget_) > (0):
                        d_24_wg_: _dafny.Seq
                        d_25_wi_: bool
                        d_26_wc_: _dafny.Seq
                        out17_: _dafny.Seq
                        out18_: bool
                        out19_: _dafny.Seq
                        out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_remainBudget_)
                        d_24_wg_ = out17_
                        d_25_wi_ = out18_
                        d_26_wc_ = out19_
                        generated = d_24_wg_
                        insideConstrainedOut = d_25_wi_
                        currentConstrainedOut = d_26_wc_
                        d_2_steps_ = (d_2_steps_) + (d_23_remainBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final answer as 'The answer is <<EXPR>>' where EXPR is an arithmetic expression using ALL relevant variables and operators from the problem. Examples: <<n * price + fee>>, <<int(n * frac) + base>>, <<(a + b) * rate / 60>>. Use only: variable names from the problem, numbers, +, -, *, /, (, ), int(). The expression MUST combine all relevant quantities with operators.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_phase1Limit_: int = int(0)
        if (maxSteps) >= (5):
            d_2_phase1Limit_ = (maxSteps) - (_dafny.euclidian_division(maxSteps, 5))
        elif True:
            d_2_phase1Limit_ = maxSteps
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_2_phase1Limit_)) and (not(insideConstrainedOut)):
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
                        d_4_eg_: _dafny.Seq
                        d_5_ei_: bool
                        d_6_ec_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_4_eg_ = out1_
                        d_5_ei_ = out2_
                        d_6_ec_ = out3_
                        generated = d_4_eg_
                        insideConstrainedOut = d_5_ei_
                        currentConstrainedOut = d_6_ec_
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_7_fg_: _dafny.Seq
            d_8_fi_: bool
            d_9_fc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_7_fg_ = out4_
            d_8_fi_ = out5_
            d_9_fc_ = out6_
            generated = d_7_fg_
            insideConstrainedOut = d_8_fi_
            currentConstrainedOut = d_9_fc_
            d_1_steps_ = (d_1_steps_) + (1)
        d_10_warmupSteps_: int
        d_10_warmupSteps_ = 0
        with _dafny.label("1"):
            while ((insideConstrainedOut) and ((d_1_steps_) < (maxSteps))) and ((d_10_warmupSteps_) < (8)):
                with _dafny.c_label("1"):
                    d_11_cp_: _dafny.Seq
                    d_11_cp_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_12_next_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_cp_, currentConstrainedOut, eosToken)
                    d_12_next_ = out7_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_10_warmupSteps_ = (d_10_warmupSteps_) + (1)
                    if (d_12_next_) == (eosToken):
                        raise _dafny.Break("1")
                    d_13_ag_: _dafny.Seq
                    d_14_ai_: bool
                    d_15_ac_: _dafny.Seq
                    out8_: _dafny.Seq
                    out9_: bool
                    out10_: _dafny.Seq
                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                    d_13_ag_ = out8_
                    d_14_ai_ = out9_
                    d_15_ac_ = out10_
                    generated = d_13_ag_
                    insideConstrainedOut = d_14_ai_
                    currentConstrainedOut = d_15_ac_
                    pass
            pass
        with _dafny.label("2"):
            while (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                with _dafny.c_label("2"):
                    d_16_cg_: _dafny.Seq
                    d_17_ci_: bool
                    d_18_cc_: _dafny.Seq
                    d_19_closed_: bool
                    out11_: _dafny.Seq
                    out12_: bool
                    out13_: _dafny.Seq
                    out14_: bool
                    out11_, out12_, out13_, out14_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_16_cg_ = out11_
                    d_17_ci_ = out12_
                    d_18_cc_ = out13_
                    d_19_closed_ = out14_
                    if d_19_closed_:
                        generated = d_16_cg_
                        insideConstrainedOut = d_17_ci_
                        currentConstrainedOut = d_18_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_20_cp_: _dafny.Seq
                        d_20_cp_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_21_next_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_20_cp_, currentConstrainedOut, eosToken)
                        d_21_next_ = out15_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_21_next_) == (eosToken):
                            raise _dafny.Break("2")
                        d_22_ag_: _dafny.Seq
                        d_23_ai_: bool
                        d_24_ac_: _dafny.Seq
                        out16_: _dafny.Seq
                        out17_: bool
                        out18_: _dafny.Seq
                        out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                        d_22_ag_ = out16_
                        d_23_ai_ = out17_
                        d_24_ac_ = out18_
                        generated = d_22_ag_
                        insideConstrainedOut = d_23_ai_
                        currentConstrainedOut = d_24_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_25_remBudget_: int
            d_25_remBudget_ = (maxSteps) - (d_1_steps_)
            d_26_wg_: _dafny.Seq
            d_27_wi_: bool
            d_28_wc_: _dafny.Seq
            out19_: _dafny.Seq
            out20_: bool
            out21_: _dafny.Seq
            out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_25_remBudget_)
            d_26_wg_ = out19_
            d_27_wi_ = out20_
            d_28_wc_ = out21_
            generated = d_26_wg_
            insideConstrainedOut = d_27_wi_
            currentConstrainedOut = d_28_wc_
            d_1_steps_ = (d_1_steps_) + (d_25_remBudget_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


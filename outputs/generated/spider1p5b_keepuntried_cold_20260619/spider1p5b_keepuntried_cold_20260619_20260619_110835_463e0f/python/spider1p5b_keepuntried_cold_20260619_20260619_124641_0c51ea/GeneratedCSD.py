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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You must output exactly: SQL: <<your SQL query>>. The SQL query goes between << and >>. Use only single quotes for strings. No backticks, no double quotes, no markdown, no explanation. Output only the SQL query between << and >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 16
        d_3_closingReserve_: int
        d_3_closingReserve_ = 5
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_5_eg_: _dafny.Seq
                                d_6_ei_: bool
                                d_7_ec_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_5_eg_ = out1_
                                d_6_ei_ = out2_
                                d_7_ec_ = out3_
                                generated = d_5_eg_
                                insideConstrainedOut = d_6_ei_
                                currentConstrainedOut = d_7_ec_
                    elif True:
                        d_8_budgetLeft_: int
                        d_8_budgetLeft_ = (maxSteps) - (d_1_steps_)
                        if (d_8_budgetLeft_) <= (d_3_closingReserve_):
                            d_9_rg_: _dafny.Seq
                            d_10_rc_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: _dafny.Seq
                            out4_, out5_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_9_rg_ = out4_
                            d_10_rc_ = out5_
                            generated = d_9_rg_
                            currentConstrainedOut = d_10_rc_
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_11_cg_: _dafny.Seq
                                d_12_ci_: bool
                                d_13_cc_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_11_cg_ = out6_
                                d_12_ci_ = out7_
                                d_13_cc_ = out8_
                                generated = d_11_cg_
                                insideConstrainedOut = d_12_ci_
                                currentConstrainedOut = d_13_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_14_cg_: _dafny.Seq
                            d_15_ci_: bool
                            d_16_cc_: _dafny.Seq
                            d_17_closed_: bool
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out12_: bool
                            out9_, out10_, out11_, out12_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_14_cg_ = out9_
                            d_15_ci_ = out10_
                            d_16_cc_ = out11_
                            d_17_closed_ = out12_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_17_closed_:
                                generated = d_14_cg_
                                insideConstrainedOut = d_15_ci_
                                currentConstrainedOut = d_16_cc_
                            elif True:
                                d_18_constrainedPrompt_: _dafny.Seq
                                d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_19_next_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), d_2_narrowThreshold_, eosToken)
                                d_19_next_ = out13_
                                if (d_19_next_) == (eosToken):
                                    d_20_rg2_: _dafny.Seq
                                    d_21_rc2_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out14_, out15_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_20_rg2_ = out14_
                                    d_21_rc2_ = out15_
                                    generated = d_20_rg2_
                                    currentConstrainedOut = d_21_rc2_
                                    if (parser).IsCompletePrefix(currentConstrainedOut):
                                        d_22_cg3_: _dafny.Seq
                                        d_23_ci3_: bool
                                        d_24_cc3_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out17_: bool
                                        out18_: _dafny.Seq
                                        out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_22_cg3_ = out16_
                                        d_23_ci3_ = out17_
                                        d_24_cc3_ = out18_
                                        generated = d_22_cg3_
                                        insideConstrainedOut = d_23_ci3_
                                        currentConstrainedOut = d_24_cc3_
                                    raise _dafny.Break("0")
                                elif True:
                                    d_25_ag_: _dafny.Seq
                                    d_26_ai_: bool
                                    d_27_ac_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                    d_25_ag_ = out19_
                                    d_26_ai_ = out20_
                                    d_27_ac_ = out21_
                                    generated = d_25_ag_
                                    insideConstrainedOut = d_26_ai_
                                    currentConstrainedOut = d_27_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


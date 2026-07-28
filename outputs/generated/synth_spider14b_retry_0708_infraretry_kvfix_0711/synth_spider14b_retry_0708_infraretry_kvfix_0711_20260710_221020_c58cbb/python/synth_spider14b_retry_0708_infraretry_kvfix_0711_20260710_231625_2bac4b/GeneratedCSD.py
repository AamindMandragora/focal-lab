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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output format: SQL: <<query>> where query is a complete valid SQL SELECT statement using only the schema tables and columns. Generate the SQL query directly, starting with SELECT.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 15
        d_3_freeStepLimit_: int
        d_3_freeStepLimit_ = 2
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    if (d_1_steps_) >= (d_3_freeStepLimit_):
                        d_4_og_: _dafny.Seq
                        d_5_oi_: bool
                        d_6_oc_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_4_og_ = out0_
                        d_5_oi_ = out1_
                        d_6_oc_ = out2_
                        generated = d_4_og_
                        insideConstrainedOut = d_5_oi_
                        currentConstrainedOut = d_6_oc_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_7_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_7_next_ = out3_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_7_next_) == (eosToken):
                            if (d_1_steps_) < (maxSteps):
                                d_8_og2_: _dafny.Seq
                                d_9_oi2_: bool
                                d_10_oc2_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_8_og2_ = out4_
                                d_9_oi2_ = out5_
                                d_10_oc2_ = out6_
                                generated = d_8_og2_
                                insideConstrainedOut = d_9_oi2_
                                currentConstrainedOut = d_10_oc2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                            if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        with _dafny.label("1"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
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
                    elif True:
                        d_14_constrainedPrompt_: _dafny.Seq
                        d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_15_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e-1'), eosToken)
                        d_15_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_15_next_) == (eosToken):
                            if (d_1_steps_) < (maxSteps):
                                d_16_closeBudget_: int
                                d_16_closeBudget_ = (maxSteps) - (d_1_steps_)
                                d_17_cg2_: _dafny.Seq
                                d_18_ci2_: bool
                                d_19_cc2_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_16_closeBudget_)
                                d_17_cg2_ = out11_
                                d_18_ci2_ = out12_
                                d_19_cc2_ = out13_
                                generated = d_17_cg2_
                                insideConstrainedOut = d_18_ci2_
                                currentConstrainedOut = d_19_cc2_
                                d_1_steps_ = maxSteps
                            raise _dafny.Break("1")
                        elif True:
                            d_20_ag_: _dafny.Seq
                            d_21_ai_: bool
                            d_22_ac_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                            d_20_ag_ = out14_
                            d_21_ai_ = out15_
                            d_22_ac_ = out16_
                            generated = d_20_ag_
                            insideConstrainedOut = d_21_ai_
                            currentConstrainedOut = d_22_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_23_closeBudget_: int
            d_23_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_24_cg3_: _dafny.Seq
            d_25_ci3_: bool
            d_26_cc3_: _dafny.Seq
            out17_: _dafny.Seq
            out18_: bool
            out19_: _dafny.Seq
            out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_closeBudget_)
            d_24_cg3_ = out17_
            d_25_ci3_ = out18_
            d_26_cc3_ = out19_
            generated = d_24_cg3_
            insideConstrainedOut = d_25_ci3_
            currentConstrainedOut = d_26_cc3_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


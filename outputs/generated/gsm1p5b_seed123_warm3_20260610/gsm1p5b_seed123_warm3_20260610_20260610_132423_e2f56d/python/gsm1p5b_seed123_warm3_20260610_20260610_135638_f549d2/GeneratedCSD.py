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
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Write the final arithmetic expression for the answer inside << >> delimiters. Use only numbers, +, -, *, /, (, ) inside the delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_maxSpanTokens_: int
        d_2_maxSpanTokens_ = 40
        d_3_spanTokens_: int
        d_3_spanTokens_ = 0
        d_4_forcedOpenThreshold_: int
        if (maxSteps) > (20):
            d_4_forcedOpenThreshold_ = (maxSteps) - (15)
        elif True:
            d_4_forcedOpenThreshold_ = _dafny.euclidian_division(maxSteps, 2)
        d_5_forcedOpen_: bool
        d_5_forcedOpen_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((not(d_5_forcedOpen_)) and ((d_1_steps_) >= (d_4_forcedOpenThreshold_))) and (((d_1_steps_) + (2)) < (maxSteps)):
                            d_6_og_: _dafny.Seq
                            d_7_oi_: bool
                            d_8_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_og_ = out0_
                            d_7_oi_ = out1_
                            d_8_oc_ = out2_
                            generated = d_6_og_
                            insideConstrainedOut = d_7_oi_
                            currentConstrainedOut = d_8_oc_
                            d_3_spanTokens_ = 0
                            d_5_forcedOpen_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                if (not(d_5_forcedOpen_)) and (((d_1_steps_) + (3)) <= (maxSteps)):
                                    d_10_og_: _dafny.Seq
                                    d_11_oi_: bool
                                    d_12_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_10_og_ = out4_
                                    d_11_oi_ = out5_
                                    d_12_oc_ = out6_
                                    generated = d_10_og_
                                    insideConstrainedOut = d_11_oi_
                                    currentConstrainedOut = d_12_oc_
                                    d_3_spanTokens_ = 0
                                    d_5_forcedOpen_ = True
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_13_eg_: _dafny.Seq
                                    d_14_ei_: bool
                                    d_15_ec_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_13_eg_ = out7_
                                    d_14_ei_ = out8_
                                    d_15_ec_ = out9_
                                    generated = d_13_eg_
                                    insideConstrainedOut = d_14_ei_
                                    currentConstrainedOut = d_15_ec_
                                    d_3_spanTokens_ = 0
                                    d_5_forcedOpen_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_cg_: _dafny.Seq
                        d_17_ci_: bool
                        d_18_cc_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_cg_ = out10_
                        d_17_ci_ = out11_
                        d_18_cc_ = out12_
                        generated = d_16_cg_
                        insideConstrainedOut = d_17_ci_
                        currentConstrainedOut = d_18_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanTokens_ = 0
                    elif (d_3_spanTokens_) >= (d_2_maxSpanTokens_):
                        d_19_rg_: _dafny.Seq
                        d_20_rc_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_19_rg_ = out13_
                        d_20_rc_ = out14_
                        generated = d_19_rg_
                        currentConstrainedOut = d_20_rc_
                        if not((parser).IsCompletePrefix(currentConstrainedOut)):
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_3_spanTokens_ = 0
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_22_next_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_22_next_ = out15_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_22_next_) == (eosToken):
                            d_23_rg2_: _dafny.Seq
                            d_24_rc2_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: _dafny.Seq
                            out16_, out17_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_23_rg2_ = out16_
                            d_24_rc2_ = out17_
                            generated = d_23_rg2_
                            currentConstrainedOut = d_24_rc2_
                            if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_3_spanTokens_ = 0
                            raise _dafny.Break("0")
                        elif True:
                            d_25_ag_: _dafny.Seq
                            d_26_ai_: bool
                            d_27_ac_: _dafny.Seq
                            out18_: _dafny.Seq
                            out19_: bool
                            out20_: _dafny.Seq
                            out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                            d_25_ag_ = out18_
                            d_26_ai_ = out19_
                            d_27_ac_ = out20_
                            generated = d_25_ag_
                            insideConstrainedOut = d_26_ai_
                            currentConstrainedOut = d_27_ac_
                            d_3_spanTokens_ = (d_3_spanTokens_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


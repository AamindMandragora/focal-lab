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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Wrap intermediate computations in << >> delimiters.\n"))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CRITICAL RULE: When the answer involves multiple factors or operations, ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "write the COMPLETE combined expression in ONE << >> span.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CORRECT example: <<n * frac_1 * frac_2>> (one span, full formula)\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WRONG example: <<n * frac_1>> * <<frac_2>> (split across two spans — NEVER do this)\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "The LAST << >> span is extracted as the final answer. It must be a ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "complete, self-contained arithmetic expression using original variable names.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use // for integer division. Do not use ** for exponentiation."))))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (((maxSteps) - (d_1_steps_)) <= (4)) and ((d_1_steps_) > (0)):
                            d_3_og_: _dafny.Seq
                            d_4_oi_: bool
                            d_5_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_3_og_ = out0_
                            d_4_oi_ = out1_
                            d_5_oc_ = out2_
                            generated = d_3_og_
                            insideConstrainedOut = d_4_oi_
                            currentConstrainedOut = d_5_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_6_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                                if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_7_og_: _dafny.Seq
                                    d_8_oi_: bool
                                    d_9_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_7_og_ = out4_
                                    d_8_oi_ = out5_
                                    d_9_oc_ = out6_
                                    generated = d_7_og_
                                    insideConstrainedOut = d_8_oi_
                                    currentConstrainedOut = d_9_oc_
                    elif True:
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            if ((maxSteps) - (d_1_steps_)) >= (5):
                                d_11_candTok_: _dafny.Seq
                                d_12_candPre_: _dafny.Seq
                                d_13_hitComp_: bool
                                d_14_hitEos_: bool
                                d_15_stepsUsed_: int
                                out7_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: bool
                                out11_: int
                                out7_, out8_, out9_, out10_, out11_ = (d_0_helpers_).SpeculativeConstrainedRollout(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, 2, eosToken)
                                d_11_candTok_ = out7_
                                d_12_candPre_ = out8_
                                d_13_hitComp_ = out9_
                                d_14_hitEos_ = out10_
                                d_15_stepsUsed_ = out11_
                                d_1_steps_ = (d_1_steps_) + (d_15_stepsUsed_)
                                if ((d_13_hitComp_) and (not(d_14_hitEos_))) and (((maxSteps) - (d_1_steps_)) >= (2)):
                                    d_16_next2_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                                    d_16_next2_ = out12_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_16_next2_) == (eosToken):
                                        d_17_cg2_: _dafny.Seq
                                        d_18_ci2_: bool
                                        d_19_cc2_: _dafny.Seq
                                        out13_: _dafny.Seq
                                        out14_: bool
                                        out15_: _dafny.Seq
                                        out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_17_cg2_ = out13_
                                        d_18_ci2_ = out14_
                                        d_19_cc2_ = out15_
                                        generated = d_17_cg2_
                                        insideConstrainedOut = d_18_ci2_
                                        currentConstrainedOut = d_19_cc2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_20_ag2_: _dafny.Seq
                                        d_21_ai2_: bool
                                        d_22_ac2_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out17_: bool
                                        out18_: _dafny.Seq
                                        out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next2_)
                                        d_20_ag2_ = out16_
                                        d_21_ai2_ = out17_
                                        d_22_ac2_ = out18_
                                        generated = d_20_ag2_
                                        insideConstrainedOut = d_21_ai2_
                                        currentConstrainedOut = d_22_ac2_
                                elif True:
                                    if ((maxSteps) - (d_1_steps_)) >= (1):
                                        d_23_cg3_: _dafny.Seq
                                        d_24_ci3_: bool
                                        d_25_cc3_: _dafny.Seq
                                        out19_: _dafny.Seq
                                        out20_: bool
                                        out21_: _dafny.Seq
                                        out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_23_cg3_ = out19_
                                        d_24_ci3_ = out20_
                                        d_25_cc3_ = out21_
                                        generated = d_23_cg3_
                                        insideConstrainedOut = d_24_ci3_
                                        currentConstrainedOut = d_25_cc3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        raise _dafny.Break("0")
                            elif True:
                                if ((maxSteps) - (d_1_steps_)) >= (1):
                                    d_26_cg4_: _dafny.Seq
                                    d_27_ci4_: bool
                                    d_28_cc4_: _dafny.Seq
                                    out22_: _dafny.Seq
                                    out23_: bool
                                    out24_: _dafny.Seq
                                    out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_26_cg4_ = out22_
                                    d_27_ci4_ = out23_
                                    d_28_cc4_ = out24_
                                    generated = d_26_cg4_
                                    insideConstrainedOut = d_27_ci4_
                                    currentConstrainedOut = d_28_cc4_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                        elif ((maxSteps) - (d_1_steps_)) <= (3):
                            d_29_closeBudget_: int
                            d_29_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_30_cg5_: _dafny.Seq
                            d_31_ci5_: bool
                            d_32_cc5_: _dafny.Seq
                            out25_: _dafny.Seq
                            out26_: bool
                            out27_: _dafny.Seq
                            out25_, out26_, out27_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_29_closeBudget_)
                            d_30_cg5_ = out25_
                            d_31_ci5_ = out26_
                            d_32_cc5_ = out27_
                            generated = d_30_cg5_
                            insideConstrainedOut = d_31_ci5_
                            currentConstrainedOut = d_32_cc5_
                            d_1_steps_ = maxSteps
                        elif True:
                            d_33_next_: _dafny.Seq
                            out28_: _dafny.Seq
                            out28_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                            d_33_next_ = out28_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_33_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_34_ag_: _dafny.Seq
                                d_35_ai_: bool
                                d_36_ac_: _dafny.Seq
                                out29_: _dafny.Seq
                                out30_: bool
                                out31_: _dafny.Seq
                                out29_, out30_, out31_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_33_next_)
                                d_34_ag_ = out29_
                                d_35_ai_ = out30_
                                d_36_ac_ = out31_
                                generated = d_34_ag_
                                insideConstrainedOut = d_35_ai_
                                currentConstrainedOut = d_36_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


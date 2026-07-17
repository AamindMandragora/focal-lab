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
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_1_guidance_: _dafny.Seq
            d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step using plain text. At the end, write a COMPLETE arithmetic expression with operators (not just a single variable) inside << >>. Use only variable names, numbers, +, -, *, /, //, %, (, ), int(). No LaTeX, no {}. Example: <<(n1 - n2) * rate * time // 60>>. Write exactly one << >> at the end."))
            (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
            d_2_prefixBudget_: int
            d_2_prefixBudget_ = _dafny.euclidian_division((maxSteps) * (78), 100)
            if (d_2_prefixBudget_) >= (maxSteps):
                d_2_prefixBudget_ = (maxSteps) - (1)
            d_3_steps_: int
            d_3_steps_ = 0
            d_4_seenClose_: bool
            d_4_seenClose_ = False
            with _dafny.label("1_0"):
                while (((d_3_steps_) < (d_2_prefixBudget_)) and (not(insideConstrainedOut))) and (not(d_4_seenClose_)):
                    with _dafny.c_label("1_0"):
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out0_
                        d_3_steps_ = (d_3_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("1_0")
                        elif (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            d_6_g2_: _dafny.Seq
                            d_7_ic2_: bool
                            d_8_cc2_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_6_g2_ = out1_
                            d_7_ic2_ = out2_
                            d_8_cc2_ = out3_
                            generated = d_6_g2_
                            insideConstrainedOut = d_7_ic2_
                            currentConstrainedOut = d_8_cc2_
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        pass
                pass
            if (insideConstrainedOut) and ((d_3_steps_) < (maxSteps)):
                d_9_closeBudget1_: int
                d_9_closeBudget1_ = (maxSteps) - (d_3_steps_)
                d_10_g3_: _dafny.Seq
                d_11_ic3_: bool
                d_12_cc3_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_9_closeBudget1_)
                d_10_g3_ = out4_
                d_11_ic3_ = out5_
                d_12_cc3_ = out6_
                generated = d_10_g3_
                insideConstrainedOut = d_11_ic3_
                currentConstrainedOut = d_12_cc3_
                d_3_steps_ = maxSteps
            elif ((not(insideConstrainedOut)) and (not(d_4_seenClose_))) and ((d_3_steps_) < (maxSteps)):
                d_13_g2_: _dafny.Seq
                d_14_ic2_: bool
                d_15_cc2_: _dafny.Seq
                out7_: _dafny.Seq
                out8_: bool
                out9_: _dafny.Seq
                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_13_g2_ = out7_
                d_14_ic2_ = out8_
                d_15_cc2_ = out9_
                generated = d_13_g2_
                insideConstrainedOut = d_14_ic2_
                currentConstrainedOut = d_15_cc2_
                d_3_steps_ = (d_3_steps_) + (1)
                d_16_minSpanTokens_: int
                d_16_minSpanTokens_ = 3
                with _dafny.label("1_3_0_0"):
                    while (insideConstrainedOut) and (((d_3_steps_) + (2)) <= (maxSteps)):
                        with _dafny.c_label("1_3_0_0"):
                            if (len(currentConstrainedOut)) >= (d_16_minSpanTokens_):
                                d_17_cg_: _dafny.Seq
                                d_18_ci_: bool
                                d_19_cc_: _dafny.Seq
                                d_20_closed_: bool
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out13_: bool
                                out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_17_cg_ = out10_
                                d_18_ci_ = out11_
                                d_19_cc_ = out12_
                                d_20_closed_ = out13_
                                d_3_steps_ = (d_3_steps_) + (1)
                                if d_20_closed_:
                                    generated = d_17_cg_
                                    insideConstrainedOut = d_18_ci_
                                    currentConstrainedOut = d_19_cc_
                                elif True:
                                    d_21_constrainedPrompt_: _dafny.Seq
                                    d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_22_next_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_22_next_ = out14_
                                    d_3_steps_ = (d_3_steps_) + (1)
                                    if (d_22_next_) == (eosToken):
                                        raise _dafny.Break("1_3_0_0")
                                    elif True:
                                        d_23_g3b_: _dafny.Seq
                                        d_24_ic3b_: bool
                                        d_25_cc3b_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out17_: _dafny.Seq
                                        out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                        d_23_g3b_ = out15_
                                        d_24_ic3b_ = out16_
                                        d_25_cc3b_ = out17_
                                        generated = d_23_g3b_
                                        insideConstrainedOut = d_24_ic3b_
                                        currentConstrainedOut = d_25_cc3b_
                            elif True:
                                d_26_constrainedPrompt_: _dafny.Seq
                                d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_27_next_: _dafny.Seq
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_27_next_ = out18_
                                d_3_steps_ = (d_3_steps_) + (1)
                                if (d_27_next_) == (eosToken):
                                    raise _dafny.Break("1_3_0_0")
                                elif True:
                                    d_28_g3c_: _dafny.Seq
                                    d_29_ic3c_: bool
                                    d_30_cc3c_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                                    d_28_g3c_ = out19_
                                    d_29_ic3c_ = out20_
                                    d_30_cc3c_ = out21_
                                    generated = d_28_g3c_
                                    insideConstrainedOut = d_29_ic3c_
                                    currentConstrainedOut = d_30_cc3c_
                            pass
                    pass
                if (insideConstrainedOut) and ((d_3_steps_) < (maxSteps)):
                    d_31_closeBudget2_: int
                    d_31_closeBudget2_ = (maxSteps) - (d_3_steps_)
                    d_32_g4_: _dafny.Seq
                    d_33_ic4_: bool
                    d_34_cc4_: _dafny.Seq
                    out22_: _dafny.Seq
                    out23_: bool
                    out24_: _dafny.Seq
                    out22_, out23_, out24_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_31_closeBudget2_)
                    d_32_g4_ = out22_
                    d_33_ic4_ = out23_
                    d_34_cc4_ = out24_
                    generated = d_32_g4_
                    insideConstrainedOut = d_33_ic4_
                    currentConstrainedOut = d_34_cc4_
                    d_3_steps_ = maxSteps
            cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
            d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using the variable names from the problem (without curly braces). At the very end, write your final arithmetic expression inside << and >>. Use ONLY plain variable names and operators +, -, *, /, //, %. The expression must include ALL relevant variables from the problem. Examples: <<n * price - discount>>, <<(n1 + n2) * rate // 60>>, <<total - n1 * w1 - n2 * w2>>. Write exactly ONE final expression inside << >>."))
            (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
            d_2_steps_: int
            d_2_steps_ = 0
            d_3_reasoningBudget_: int
            d_3_reasoningBudget_ = _dafny.euclidian_division((maxSteps) * (72), 100)
            d_4_minConstrainedTokens_: int
            d_4_minConstrainedTokens_ = 5
            with _dafny.label("1_0"):
                while ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
                    with _dafny.c_label("1_0"):
                        if (d_2_steps_) >= (d_3_reasoningBudget_):
                            if (d_2_steps_) < (maxSteps):
                                d_5_og_: _dafny.Seq
                                d_6_oi_: bool
                                d_7_oc_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_5_og_ = out0_
                                d_6_oi_ = out1_
                                d_7_oc_ = out2_
                                generated = d_5_og_
                                insideConstrainedOut = d_6_oi_
                                currentConstrainedOut = d_7_oc_
                                d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("1_0")
                        d_8_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_8_next_ = out3_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_8_next_) == (eosToken):
                            raise _dafny.Break("1_0")
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                        if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        pass
                pass
            with _dafny.label("1_1"):
                while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                    with _dafny.c_label("1_1"):
                        if ((len(currentConstrainedOut)) >= (d_4_minConstrainedTokens_)) and (((d_2_steps_) + (1)) < (maxSteps)):
                            d_9_cg_: _dafny.Seq
                            d_10_ci_: bool
                            d_11_cc_: _dafny.Seq
                            d_12_closed_: bool
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out7_: bool
                            out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_9_cg_ = out4_
                            d_10_ci_ = out5_
                            d_11_cc_ = out6_
                            d_12_closed_ = out7_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if d_12_closed_:
                                generated = d_9_cg_
                                insideConstrainedOut = d_10_ci_
                                currentConstrainedOut = d_11_cc_
                                raise _dafny.Break("1_1")
                        if (d_2_steps_) >= (maxSteps):
                            raise _dafny.Break("1_1")
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_14_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (len(currentConstrainedOut)) < (d_4_minConstrainedTokens_):
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('8e0'), eosToken)
                            d_14_next_ = out8_
                        elif True:
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_14_next_ = out9_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_14_next_) == (eosToken):
                            if ((len(currentConstrainedOut)) >= (d_4_minConstrainedTokens_)) and ((d_2_steps_) < (maxSteps)):
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
                            raise _dafny.Break("1_1")
                        elif True:
                            d_19_appendedGenerated_: _dafny.Seq
                            d_20_appendedInside_: bool
                            d_21_appendedCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                            d_19_appendedGenerated_ = out14_
                            d_20_appendedInside_ = out15_
                            d_21_appendedCurrent_ = out16_
                            generated = d_19_appendedGenerated_
                            insideConstrainedOut = d_20_appendedInside_
                            currentConstrainedOut = d_21_appendedCurrent_
                        pass
                pass
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_22_closeBudget_: int
                d_22_closeBudget_ = (maxSteps) - (d_2_steps_)
                d_23_cg_: _dafny.Seq
                d_24_ci_: bool
                d_25_cc_: _dafny.Seq
                out17_: _dafny.Seq
                out18_: bool
                out19_: _dafny.Seq
                out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_closeBudget_)
                d_23_cg_ = out17_
                d_24_ci_ = out18_
                d_25_cc_ = out19_
                generated = d_23_cg_
                insideConstrainedOut = d_24_ci_
                currentConstrainedOut = d_25_cc_
                d_2_steps_ = maxSteps
            cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


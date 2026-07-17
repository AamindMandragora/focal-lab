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
            d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. After your reasoning, write the final arithmetic expression inside << and >>. Use ONLY the exact variable names from the problem (no curly braces {}, no dollar signs, no LaTeX). Use operators: +, -, *, /, //, %. Example: <<(n1 + n2) * rate - discount>>. Write exactly ONE final expression at the very end."))
            (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
            d_2_steps_: int
            d_2_steps_ = 0
            d_3_minReasoning_: int
            if (maxSteps) >= (10):
                d_3_minReasoning_ = _dafny.euclidian_division(maxSteps, 3)
            elif True:
                d_3_minReasoning_ = 1
            d_4_forceBudget_: int
            if (maxSteps) >= (6):
                d_4_forceBudget_ = _dafny.euclidian_division((maxSteps) * (2), 3)
            elif True:
                d_4_forceBudget_ = (maxSteps) - (1)
            d_5_closeReserve_: int
            if (maxSteps) >= (20):
                d_5_closeReserve_ = 30
            elif True:
                d_5_closeReserve_ = 5
            with _dafny.label("1_0"):
                while ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
                    with _dafny.c_label("1_0"):
                        if (d_2_steps_) >= (d_4_forceBudget_):
                            if ((d_2_steps_) + (1)) <= (maxSteps):
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
                                d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("1_0")
                        d_9_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_9_next_ = out3_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_9_next_) == (eosToken):
                            raise _dafny.Break("1_0")
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                        if ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and ((d_2_steps_) >= (d_3_minReasoning_)):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        pass
                pass
            with _dafny.label("1_1"):
                while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                    with _dafny.c_label("1_1"):
                        if ((d_2_steps_) + (d_5_closeReserve_)) >= (maxSteps):
                            raise _dafny.Break("1_1")
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_11_next_: _dafny.Seq
                        d_12_softOk_: bool
                        out4_: _dafny.Seq
                        out5_: bool
                        out4_, out5_ = (d_0_helpers_).SoftConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('3e0'), eosToken)
                        d_11_next_ = out4_
                        d_12_softOk_ = out5_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_11_next_) == (eosToken):
                            if (d_2_steps_) < (maxSteps):
                                d_13_cg_: _dafny.Seq
                                d_14_ci_: bool
                                d_15_cc_: _dafny.Seq
                                d_16_closed_: bool
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out9_: bool
                                out6_, out7_, out8_, out9_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_13_cg_ = out6_
                                d_14_ci_ = out7_
                                d_15_cc_ = out8_
                                d_16_closed_ = out9_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if d_16_closed_:
                                    generated = d_13_cg_
                                    insideConstrainedOut = d_14_ci_
                                    currentConstrainedOut = d_15_cc_
                            raise _dafny.Break("1_1")
                        elif True:
                            d_17_valid_: bool
                            out10_: bool
                            out10_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_11_next_)
                            d_17_valid_ = out10_
                            if d_17_valid_:
                                d_18_ag_: _dafny.Seq
                                d_19_ai_: bool
                                d_20_ac_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                                d_18_ag_ = out11_
                                d_19_ai_ = out12_
                                d_20_ac_ = out13_
                                generated = d_18_ag_
                                insideConstrainedOut = d_19_ai_
                                currentConstrainedOut = d_20_ac_
                        pass
                pass
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_21_closeBudget_: int
                d_21_closeBudget_ = (maxSteps) - (d_2_steps_)
                d_22_cg_: _dafny.Seq
                d_23_ci_: bool
                d_24_cc_: _dafny.Seq
                out14_: _dafny.Seq
                out15_: bool
                out16_: _dafny.Seq
                out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_21_closeBudget_)
                d_22_cg_ = out14_
                d_23_ci_ = out15_
                d_24_cc_ = out16_
                generated = d_22_cg_
                insideConstrainedOut = d_23_ci_
                currentConstrainedOut = d_24_cc_
                d_2_steps_ = maxSteps
            cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
        (d_0_helpers_).AppendTaskGuidance(lm, ((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Wrap every intermediate symbolic expression and the final numeric answer inside << >> delimiters.\n"))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Each << >> span must contain a COMPLETE arithmetic expression with at least one operator—not a single variable or constant alone.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Example: The total cost is <<n * price + tax>>. The percentage is <<100 * hits // total>>.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use // for integer division. Do not use ** for exponentiation.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "The LAST << >> span must contain the complete final answer as one arithmetic expression."))))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (((maxSteps) - (d_1_steps_)) <= (4)) and ((d_1_steps_) > (0)):
                            d_2_og_: _dafny.Seq
                            d_3_oi_: bool
                            d_4_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_2_og_ = out0_
                            d_3_oi_ = out1_
                            d_4_oc_ = out2_
                            generated = d_2_og_
                            insideConstrainedOut = d_3_oi_
                            currentConstrainedOut = d_4_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_5_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_5_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_5_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                                if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_6_og_: _dafny.Seq
                                    d_7_oi_: bool
                                    d_8_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_6_og_ = out4_
                                    d_7_oi_ = out5_
                                    d_8_oc_ = out6_
                                    generated = d_6_og_
                                    insideConstrainedOut = d_7_oi_
                                    currentConstrainedOut = d_8_oc_
                    elif True:
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) > (1)):
                            d_9_cg_: _dafny.Seq
                            d_10_ci_: bool
                            d_11_cc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_cg_ = out7_
                            d_10_ci_ = out8_
                            d_11_cc_ = out9_
                            generated = d_9_cg_
                            insideConstrainedOut = d_10_ci_
                            currentConstrainedOut = d_11_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif ((maxSteps) - (d_1_steps_)) <= (3):
                            d_12_closeBudget_: int
                            d_12_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_13_cg_: _dafny.Seq
                            d_14_ci_: bool
                            d_15_cc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_12_closeBudget_)
                            d_13_cg_ = out10_
                            d_14_ci_ = out11_
                            d_15_cc_ = out12_
                            generated = d_13_cg_
                            insideConstrainedOut = d_14_ci_
                            currentConstrainedOut = d_15_cc_
                            d_1_steps_ = maxSteps
                        elif True:
                            d_16_constrainedPrompt_: _dafny.Seq
                            d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_17_next_: _dafny.Seq
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_17_next_ = out13_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_17_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_18_ag_: _dafny.Seq
                                d_19_ai_: bool
                                d_20_ac_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                d_18_ag_ = out14_
                                d_19_ai_ = out15_
                                d_20_ac_ = out16_
                                generated = d_18_ag_
                                insideConstrainedOut = d_19_ai_
                                currentConstrainedOut = d_20_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


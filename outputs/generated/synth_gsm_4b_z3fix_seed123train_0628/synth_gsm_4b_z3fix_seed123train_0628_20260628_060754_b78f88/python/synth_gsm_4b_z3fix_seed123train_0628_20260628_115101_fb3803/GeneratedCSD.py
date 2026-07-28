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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. At the very end, write ONLY the final answer expression inside << >> delimiters. CRITICAL: Inside << >>, use ONLY plain variable names WITHOUT any curly braces. Write: <<n * rate>> NOT <<{n} * {rate}>>. Use only standard operators: +, -, *, /, //, %, and parentheses. No LaTeX, no **, no {braces} of any kind inside << >>. Keep the expression simple and direct."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_spanOpened_: bool
        d_3_spanOpened_ = insideConstrained
        with _dafny.label("0"):
            while ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_4_remaining_: int
                    d_4_remaining_ = (maxSteps) - (d_2_steps_)
                    if ((not(d_3_spanOpened_)) and ((d_2_steps_) >= (720))) and ((d_4_remaining_) >= (80)):
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
                        d_3_spanOpened_ = True
                        raise _dafny.Break("0")
                    if (d_4_remaining_) <= (10):
                        raise _dafny.Break("0")
                    d_8_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_8_next_ = out3_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_8_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                        if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_9_remAfter_: int
                            d_9_remAfter_ = (maxSteps) - (d_2_steps_)
                            if (d_9_remAfter_) >= (15):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_3_spanOpened_ = True
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_10_rem2_: int
            d_10_rem2_ = (maxSteps) - (d_2_steps_)
            d_11_groundBudget_: int
            d_11_groundBudget_ = 0
            if (d_10_rem2_) > (30):
                d_11_groundBudget_ = _dafny.euclidian_division((d_10_rem2_) - (30), 2)
            if (d_11_groundBudget_) >= (5):
                d_12_stable_: _dafny.Seq
                d_12_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_13_filled_: _dafny.Seq
                out4_: _dafny.Seq
                out4_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, (prompt) + (d_12_stable_), currentConstrainedOut, eosToken, d_11_groundBudget_, 3, d_11_groundBudget_)
                d_13_filled_ = out4_
                generated = (d_12_stable_) + (d_13_filled_)
                currentConstrainedOut = d_13_filled_
                d_2_steps_ = (d_2_steps_) + (d_11_groundBudget_)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_14_closeBudget_: int
            d_14_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_15_cg_: _dafny.Seq
            d_16_ci_: bool
            d_17_cc_: _dafny.Seq
            out5_: _dafny.Seq
            out6_: bool
            out7_: _dafny.Seq
            out5_, out6_, out7_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudget_)
            d_15_cg_ = out5_
            d_16_ci_ = out6_
            d_17_cc_ = out7_
            generated = d_15_cg_
            insideConstrainedOut = d_16_ci_
            currentConstrainedOut = d_17_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


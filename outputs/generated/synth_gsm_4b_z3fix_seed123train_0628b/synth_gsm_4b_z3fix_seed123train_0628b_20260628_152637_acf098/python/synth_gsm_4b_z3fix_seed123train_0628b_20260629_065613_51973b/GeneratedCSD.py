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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final answer expression ONCE at the very end inside << >>. Use only variable names (no braces), numbers, +, -, *, /, //, int(), and parentheses. Example: <<int(n1 * price - discount)>>. Open << >> only once, for the final answer only. Do not repeat tokens.")))
        d_2_unconstrainedBudget_: int
        d_2_unconstrainedBudget_ = _dafny.euclidian_division((maxSteps) * (75), 100)
        if ((maxSteps) >= (60)) and ((d_2_unconstrainedBudget_) > ((maxSteps) - (60))):
            d_2_unconstrainedBudget_ = (maxSteps) - (60)
        if (d_2_unconstrainedBudget_) > (maxSteps):
            d_2_unconstrainedBudget_ = maxSteps
        d_3_closedSpan_: bool
        d_3_closedSpan_ = False
        with _dafny.label("0"):
            while (((d_1_steps_) < (d_2_unconstrainedBudget_)) and (not(insideConstrainedOut))) and (not(d_3_closedSpan_)):
                with _dafny.c_label("0"):
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
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            generated = out1_
                            insideConstrainedOut = out2_
                            currentConstrainedOut = out3_
                            if (d_1_steps_) < (maxSteps):
                                d_5_fillBudget_: int
                                d_5_fillBudget_ = (maxSteps) - (d_1_steps_)
                                if (d_5_fillBudget_) > (50):
                                    d_5_fillBudget_ = 50
                                d_6_rolloutGen_: _dafny.Seq
                                d_7_rolloutSteps_: int
                                d_8_rolloutEos_: bool
                                out4_: _dafny.Seq
                                out5_: int
                                out6_: bool
                                out4_, out5_, out6_ = (d_0_helpers_).RolloutConstrainedWithPenalties(lm, parser, prompt, currentConstrainedOut, d_5_fillBudget_, generated, _dafny.BigRational('3e0'), eosToken)
                                d_6_rolloutGen_ = out4_
                                d_7_rolloutSteps_ = out5_
                                d_8_rolloutEos_ = out6_
                                currentConstrainedOut = d_6_rolloutGen_
                                d_9_stablePrefix_: _dafny.Seq
                                d_9_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (0):])
                                generated = (generated) + (d_6_rolloutGen_)
                                d_1_steps_ = (d_1_steps_) + (d_7_rolloutSteps_)
                                if (d_1_steps_) > (maxSteps):
                                    d_1_steps_ = maxSteps
                                if (d_1_steps_) < (maxSteps):
                                    d_10_closeBudget_: int
                                    d_10_closeBudget_ = (maxSteps) - (d_1_steps_)
                                    if (d_10_closeBudget_) > (20):
                                        d_10_closeBudget_ = 20
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_10_closeBudget_)
                                    generated = out7_
                                    insideConstrainedOut = out8_
                                    currentConstrainedOut = out9_
                                    d_1_steps_ = (d_1_steps_) + (d_10_closeBudget_)
                                    if (d_1_steps_) > (maxSteps):
                                        d_1_steps_ = maxSteps
                                    if not(insideConstrainedOut):
                                        d_3_closedSpan_ = True
                            raise _dafny.Break("0")
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_11_closeRemaining_: int
            d_11_closeRemaining_ = (maxSteps) - (d_1_steps_)
            out10_: _dafny.Seq
            out11_: bool
            out12_: _dafny.Seq
            out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_closeRemaining_)
            generated = out10_
            insideConstrainedOut = out11_
            currentConstrainedOut = out12_
            d_1_steps_ = maxSteps
            if not(insideConstrainedOut):
                d_3_closedSpan_ = True
        if ((not(d_3_closedSpan_)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
            d_12_remaining_: int
            d_12_remaining_ = (maxSteps) - (d_1_steps_)
            if (d_12_remaining_) >= (5):
                out13_: _dafny.Seq
                out14_: bool
                out15_: _dafny.Seq
                out13_, out14_, out15_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                generated = out13_
                insideConstrainedOut = out14_
                currentConstrainedOut = out15_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_1_steps_) < (maxSteps):
                    d_13_fillBudget2_: int
                    d_13_fillBudget2_ = (maxSteps) - (d_1_steps_)
                    if (d_13_fillBudget2_) > (50):
                        d_13_fillBudget2_ = 50
                    d_14_rolloutGen2_: _dafny.Seq
                    d_15_rolloutSteps2_: int
                    d_16_rolloutEos2_: bool
                    out16_: _dafny.Seq
                    out17_: int
                    out18_: bool
                    out16_, out17_, out18_ = (d_0_helpers_).RolloutConstrainedWithPenalties(lm, parser, prompt, currentConstrainedOut, d_13_fillBudget2_, generated, _dafny.BigRational('3e0'), eosToken)
                    d_14_rolloutGen2_ = out16_
                    d_15_rolloutSteps2_ = out17_
                    d_16_rolloutEos2_ = out18_
                    currentConstrainedOut = d_14_rolloutGen2_
                    generated = (generated) + (d_14_rolloutGen2_)
                    d_1_steps_ = (d_1_steps_) + (d_15_rolloutSteps2_)
                    if (d_1_steps_) > (maxSteps):
                        d_1_steps_ = maxSteps
                if (d_1_steps_) < (maxSteps):
                    d_17_finalCloseBudget_: int
                    d_17_finalCloseBudget_ = (maxSteps) - (d_1_steps_)
                    out19_: _dafny.Seq
                    out20_: bool
                    out21_: _dafny.Seq
                    out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_finalCloseBudget_)
                    generated = out19_
                    insideConstrainedOut = out20_
                    currentConstrainedOut = out21_
                    d_1_steps_ = maxSteps
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_18_finalBudget_: int
            d_18_finalBudget_ = (maxSteps) - (d_1_steps_)
            out22_: _dafny.Seq
            out23_: bool
            out24_: _dafny.Seq
            out22_, out23_, out24_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_finalBudget_)
            generated = out22_
            insideConstrainedOut = out23_
            currentConstrainedOut = out24_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


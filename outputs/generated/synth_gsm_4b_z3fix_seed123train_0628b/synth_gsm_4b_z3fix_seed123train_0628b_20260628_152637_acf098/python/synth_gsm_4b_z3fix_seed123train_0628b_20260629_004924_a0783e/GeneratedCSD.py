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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. At the very end, write the final numeric answer as a complete arithmetic expression inside << >>. The expression must use: numbers, variable names (no curly braces), +, -, *, /, //, %, int(), parentheses. No {braces}, no **, no words. Write << >> EXACTLY ONCE at the very end.")))
        d_2_unconstrainedBudget_: int
        d_2_unconstrainedBudget_ = _dafny.euclidian_division((maxSteps) * (75), 100)
        if ((maxSteps) >= (60)) and ((d_2_unconstrainedBudget_) > ((maxSteps) - (60))):
            d_2_unconstrainedBudget_ = (maxSteps) - (60)
        if (d_2_unconstrainedBudget_) > (maxSteps):
            d_2_unconstrainedBudget_ = maxSteps
        with _dafny.label("0"):
            while (((d_1_steps_) < (d_2_unconstrainedBudget_)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
                with _dafny.c_label("0"):
                    d_3_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_3_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_3_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                        if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            generated = out1_
                            insideConstrainedOut = out2_
                            currentConstrainedOut = out3_
                            if (d_1_steps_) < (maxSteps):
                                d_4_closeBudget1_: int
                                d_4_closeBudget1_ = (maxSteps) - (d_1_steps_)
                                if (d_4_closeBudget1_) > (50):
                                    d_4_closeBudget1_ = 50
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_4_closeBudget1_)
                                generated = out4_
                                insideConstrainedOut = out5_
                                currentConstrainedOut = out6_
                                d_1_steps_ = (d_1_steps_) + (d_4_closeBudget1_)
                                if (d_1_steps_) > (maxSteps):
                                    d_1_steps_ = maxSteps
                    pass
            pass
        with _dafny.label("1"):
            while (((d_1_steps_) < (d_2_unconstrainedBudget_)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
                with _dafny.c_label("1"):
                    d_5_next_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_5_next_ = out7_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_5_next_) == (eosToken):
                        raise _dafny.Break("1")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            generated = out8_
                            insideConstrainedOut = out9_
                            currentConstrainedOut = out10_
                            if (d_1_steps_) < (maxSteps):
                                d_6_closeBudget2_: int
                                d_6_closeBudget2_ = (maxSteps) - (d_1_steps_)
                                if (d_6_closeBudget2_) > (50):
                                    d_6_closeBudget2_ = 50
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_6_closeBudget2_)
                                generated = out11_
                                insideConstrainedOut = out12_
                                currentConstrainedOut = out13_
                                d_1_steps_ = (d_1_steps_) + (d_6_closeBudget2_)
                                if (d_1_steps_) > (maxSteps):
                                    d_1_steps_ = maxSteps
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_7_remaining_: int
            d_7_remaining_ = (maxSteps) - (d_1_steps_)
            if (d_7_remaining_) >= (5):
                out14_: _dafny.Seq
                out15_: bool
                out16_: _dafny.Seq
                out14_, out15_, out16_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                generated = out14_
                insideConstrainedOut = out15_
                currentConstrainedOut = out16_
                d_1_steps_ = (d_1_steps_) + (1)
                d_8_fillBudget_: int
                d_8_fillBudget_ = 0
                if ((maxSteps) - (d_1_steps_)) > (15):
                    d_8_fillBudget_ = ((maxSteps) - (d_1_steps_)) - (15)
                if (d_8_fillBudget_) > (45):
                    d_8_fillBudget_ = 45
                d_9_fillSteps_: int
                d_9_fillSteps_ = 0
                with _dafny.label("4_0_0"):
                    while (((d_9_fillSteps_) < (d_8_fillBudget_)) and (insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                        with _dafny.c_label("4_0_0"):
                            d_10_constrainedPrompt_: _dafny.Seq
                            d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_11_next_: _dafny.Seq
                            out17_: _dafny.Seq
                            out17_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_11_next_ = out17_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_9_fillSteps_ = (d_9_fillSteps_) + (1)
                            if (d_11_next_) == (eosToken):
                                raise _dafny.Break("4_0_0")
                            elif True:
                                d_12_isComplete_: bool
                                d_12_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if not(d_12_isComplete_):
                                    d_13_ag_: _dafny.Seq
                                    d_14_ai_: bool
                                    d_15_ac_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                                    d_13_ag_ = out18_
                                    d_14_ai_ = out19_
                                    d_15_ac_ = out20_
                                    generated = d_13_ag_
                                    insideConstrainedOut = d_14_ai_
                                    currentConstrainedOut = d_15_ac_
                            pass
                    pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_16_closeBudget_: int
            d_16_closeBudget_ = (maxSteps) - (d_1_steps_)
            out21_: _dafny.Seq
            out22_: bool
            out23_: _dafny.Seq
            out21_, out22_, out23_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_16_closeBudget_)
            generated = out21_
            insideConstrainedOut = out22_
            currentConstrainedOut = out23_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


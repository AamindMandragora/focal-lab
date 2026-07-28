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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem symbolically using the problem's placeholder variable names. The problem statement writes variables in curly braces such as {n}, {k}, {x}, {w1}, {sides}, {target}; treat each as an ordinary algebraic variable BUT WRITE THEM WITHOUT CURLY BRACES in your answer (e.g., n - k*x, not {n} - {k}*{x}). Hard rule: NEVER put `{` or `}` inside <<...>>. Wrap each intermediate calculation AND the final answer between << and >>, and close every span promptly with >>. Use ONLY integer arithmetic: operators + - * and // (two-slash floor division), plus parentheses. Do NOT use /, **, ^, %, =, units, words, or any function name (no int, min, max, ceil, floor, round, abs). CRITICAL: Use EVERY variable that appears in the problem; do not silently drop any. Remember unit/period conversions and bake them into the expression: 1 year = 12 months (so 'every n months in a year' contributes a 12//n factor), 1 foot = 12 inches (use *12), a percentage uses *100, 'half'=//2, 'quarter'=//4, 'twice'=*2. Worked pattern: problem says {a} per {area}, {x} {area}, every {n} months in a year -> answer <<a*x*(12//n)>>. Keep the response under ~120 words and end with: The answer is <<final_expression>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_lastTok_: _dafny.Seq
        d_2_lastTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        d_3_repeatCount_: int
        d_3_repeatCount_ = 0
        d_4_outsideTokens_: int
        d_4_outsideTokens_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            d_4_outsideTokens_ = (d_4_outsideTokens_) + (1)
                            if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_lastTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                                d_3_repeatCount_ = 0
                            elif True:
                                if (d_5_next_) == (d_2_lastTok_):
                                    d_3_repeatCount_ = (d_3_repeatCount_) + (1)
                                    if (d_3_repeatCount_) >= (4):
                                        raise _dafny.Break("0")
                                elif True:
                                    d_2_lastTok_ = d_5_next_
                                    d_3_repeatCount_ = 0
                                if (d_4_outsideTokens_) >= (320):
                                    raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_6_closedGenerated_: _dafny.Seq
                        d_7_closedInside_: bool
                        d_8_closedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_6_closedGenerated_ = out1_
                        d_7_closedInside_ = out2_
                        d_8_closedCurrent_ = out3_
                        generated = d_6_closedGenerated_
                        insideConstrainedOut = d_7_closedInside_
                        currentConstrainedOut = d_8_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_lastTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_3_repeatCount_ = 0
                    elif True:
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_10_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ceil")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "floor")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "round")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "abs")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "int"))]), _dafny.BigRational('6e0'), 12, eosToken)
                        d_10_next_ = out4_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_10_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_11_appendedGenerated_: _dafny.Seq
                            d_12_appendedInside_: bool
                            d_13_appendedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                            d_11_appendedGenerated_ = out5_
                            d_12_appendedInside_ = out6_
                            d_13_appendedCurrent_ = out7_
                            generated = d_11_appendedGenerated_
                            insideConstrainedOut = d_12_appendedInside_
                            currentConstrainedOut = d_13_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


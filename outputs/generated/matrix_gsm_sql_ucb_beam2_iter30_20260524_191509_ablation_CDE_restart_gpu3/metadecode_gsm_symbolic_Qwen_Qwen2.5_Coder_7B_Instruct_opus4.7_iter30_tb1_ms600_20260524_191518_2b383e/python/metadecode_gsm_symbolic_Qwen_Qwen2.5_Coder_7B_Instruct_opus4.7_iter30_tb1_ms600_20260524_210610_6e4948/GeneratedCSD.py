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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the actual math word problem in the question step by step using its specific named quantities. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(1) Inside << >> write BARE identifiers exactly as the question spells them, preserving underscores (n, c, n1, n_1, frac_1, w_2, k_2). NEVER include curly braces: if the question shows {n_1}, write n_1; if it shows {frac_1}, write frac_1. Drop every { and } character. Use ONLY identifiers that literally appear in the question — do not invent variables like tf, t1, x when the question does not use them. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(2) Wrap each arithmetic step in << >>, e.g. <<r * w>>, <<n - n_1 - n_2>>, <<(a + b) // capacity>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(3) Allowed inside << >>: digits, identifiers with underscores, parentheses, + - * /, // (Python integer division), and int(...). NO curly braces, NO LaTeX (no \\frac, no \\cdot, no \\times, no \\[, no \\]), NO $, NO round/ceil/floor/math.*. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(4) For integer counts from dividing totals (trips, batches, people), use //: <<(n1 * w1 + n2 * w2) // total>>. For integer counts derived from fractions or percents, wrap with int(): <<int(n * frac_1 * frac_2)>> or <<int(n * p / 100)>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(5) Be concise. Plain prose only, no LaTeX math blocks, no enumerated 'Step:' headers. End with exactly one line: '#### <<final_expression>>' where final_expression is ONE arithmetic expression over bare identifiers from the question."))))
        d_1_penaltyTokens_: _dafny.Seq
        d_1_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "$")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\frac")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\begin")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\end")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\text")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\cdot")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\times")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\[")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\]")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "round")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ceil")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "floor")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "math")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max"))])
        d_2_steps_: int
        d_2_steps_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            d_4_entered_: bool
                            d_4_entered_ = False
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_4_entered_ = True
                            elif ((len(d_3_next_)) >= (2)) and ((_dafny.SeqWithoutIsStrInference((d_3_next_)[(len(d_3_next_)) - (2)::])) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                                d_4_entered_ = True
                            elif (((len(d_3_next_)) >= (1)) and (((d_3_next_)[(len(d_3_next_)) - (1)]) == (_dafny.CodePoint('<')))) and ((len(generated)) >= (2)):
                                d_5_prev_: _dafny.Seq
                                d_5_prev_ = (generated)[(len(generated)) - (2)]
                                if ((len(d_5_prev_)) >= (1)) and (((d_5_prev_)[(len(d_5_prev_)) - (1)]) == (_dafny.CodePoint('<'))):
                                    d_4_entered_ = True
                            if d_4_entered_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
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
                        d_2_steps_ = (d_2_steps_) + (1)
                    elif True:
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_10_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_1_penaltyTokens_, _dafny.BigRational('1e1'), 12, eosToken)
                        d_10_next_ = out4_
                        d_2_steps_ = (d_2_steps_) + (1)
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
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


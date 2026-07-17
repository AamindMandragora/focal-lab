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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the actual math word problem in the question step by step using its specific named quantities. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(1) Inside << >> write BARE identifiers exactly as the question spells them, preserving underscores (n, c, n1, n_1, frac_1, w_2, k_2). NEVER include curly braces: if the question shows {n_1}, write n_1; if it shows {frac_1}, write frac_1. Drop every { and } character. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(2) Wrap each arithmetic step in << >>, e.g. <<r * w>>, <<n - n_1 - n_2>>, <<(a + b) // capacity>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(3) Allowed inside << >>: digits, identifiers with underscores, parentheses, + - * /, // (Python integer division), and int(...). NO curly braces, NO LaTeX, NO $, NO \\frac, NO round/ceil/floor/math.*. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(4) For integer counts from dividing totals (trips, batches, people), use //: <<(n1 * w1 + n2 * w2) // total>>. For integer counts derived from fractions or percents, wrap with int(): <<int(n * frac_1 * frac_2)>> or <<int(n * p / 100)>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(5) End with exactly one line: '#### <<final_expression>>' where final_expression is ONE arithmetic expression over bare identifiers from the question."))))
        d_1_penaltyTokens_: _dafny.Seq
        d_1_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "$")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\frac")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\begin")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\end")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\text")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\cdot")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\times")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "round")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ceil")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "floor")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "math")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max"))])
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
                            d_4_openDelim_: bool
                            d_4_openDelim_ = False
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_4_openDelim_ = True
                            elif ((len(d_3_next_)) >= (2)) and ((_dafny.SeqWithoutIsStrInference((d_3_next_)[(len(d_3_next_)) - (2)::])) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                                d_4_openDelim_ = True
                            elif (len(generated)) >= (2):
                                d_5_prev_: _dafny.Seq
                                d_5_prev_ = (generated)[(len(generated)) - (2)]
                                d_6_combined_: _dafny.Seq
                                d_6_combined_ = (d_5_prev_) + (d_3_next_)
                                if ((len(d_6_combined_)) >= (2)) and ((_dafny.SeqWithoutIsStrInference((d_6_combined_)[(len(d_6_combined_)) - (2)::])) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                                    d_4_openDelim_ = True
                            if d_4_openDelim_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_7_closedGenerated_: _dafny.Seq
                        d_8_closedInside_: bool
                        d_9_closedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_7_closedGenerated_ = out1_
                        d_8_closedInside_ = out2_
                        d_9_closedCurrent_ = out3_
                        generated = d_7_closedGenerated_
                        insideConstrainedOut = d_8_closedInside_
                        currentConstrainedOut = d_9_closedCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                    elif True:
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_11_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_1_penaltyTokens_, _dafny.BigRational('1e1'), 12, eosToken)
                        d_11_next_ = out4_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_11_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_12_appendedGenerated_: _dafny.Seq
                            d_13_appendedInside_: bool
                            d_14_appendedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                            d_12_appendedGenerated_ = out5_
                            d_13_appendedInside_ = out6_
                            d_14_appendedCurrent_ = out7_
                            generated = d_12_appendedGenerated_
                            insideConstrainedOut = d_13_appendedInside_
                            currentConstrainedOut = d_14_appendedCurrent_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


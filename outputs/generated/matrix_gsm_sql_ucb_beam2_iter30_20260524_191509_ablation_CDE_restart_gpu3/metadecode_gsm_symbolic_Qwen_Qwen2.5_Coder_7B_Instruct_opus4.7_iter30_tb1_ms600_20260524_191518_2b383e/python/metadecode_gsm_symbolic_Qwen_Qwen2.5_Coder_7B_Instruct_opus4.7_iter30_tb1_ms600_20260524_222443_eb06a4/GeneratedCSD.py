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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "RULES (do not echo these rules in the response; apply them silently): "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "* Wrap every arithmetic step in << >>. * Inside << >>, use ONLY: bare identifiers from the question (preserve underscores: n, c, n1, n_1, frac_1, w_2, k_2), digits, parentheses, + - * / and // (integer division), and int(...). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "* NEVER write { or } inside << >>: if the question shows {n_1}, write n_1; if it shows {frac_1}, write frac_1. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "* No LaTeX, no $, no \\frac, no round/ceil/floor/min/max/math.*. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "* For integer counts from dividing totals (trips, batches, people), use //: <<(n1 * w1 + n2 * w2) // total>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "* For integer counts derived from fractions or percents, use int(): <<int(n * frac_1 * frac_2)>>, <<int(n * p / 100)>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "* Keep each << >> expression short; do not repeat the same sub-expression inside one span. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "* End with one line exactly: #### <<final_expression>> where final_expression is one arithmetic expression over the question's bare identifiers."))))
        d_1_penaltyTokens_: _dafny.Seq
        d_1_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "$")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\frac")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\begin")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\end")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\text")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\cdot")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\times")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "round")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ceil")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "floor")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "math")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max"))])
        d_2_divGroup_: _dafny.Seq
        d_2_divGroup_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " //"))])
        d_3_intGroup_: _dafny.Seq
        d_3_intGroup_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "int")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "int(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " int")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " int("))])
        d_4_parenGroup_: _dafny.Seq
        d_4_parenGroup_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ("))])
        d_5_localBoostGroups_: _dafny.Seq
        d_5_localBoostGroups_ = _dafny.SeqWithoutIsStrInference([d_2_divGroup_, d_3_intGroup_, d_4_parenGroup_])
        d_6_steps_: int
        d_6_steps_ = 0
        d_7_spanLimit_: int
        d_7_spanLimit_ = 28
        with _dafny.label("0"):
            while (d_6_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_8_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_8_next_ = out0_
                        d_6_steps_ = (d_6_steps_) + (1)
                        if (d_8_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                            if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_9_closedGenerated_: _dafny.Seq
                        d_10_closedInside_: bool
                        d_11_closedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_9_closedGenerated_ = out1_
                        d_10_closedInside_ = out2_
                        d_11_closedCurrent_ = out3_
                        generated = d_9_closedGenerated_
                        insideConstrainedOut = d_10_closedInside_
                        currentConstrainedOut = d_11_closedCurrent_
                        d_6_steps_ = (d_6_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_7_spanLimit_):
                        d_12_rolledGenerated_: _dafny.Seq
                        d_13_rolledCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: _dafny.Seq
                        out4_, out5_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_12_rolledGenerated_ = out4_
                        d_13_rolledCurrent_ = out5_
                        generated = d_12_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_13_rolledCurrent_
                        d_6_steps_ = (d_6_steps_) + (1)
                    elif True:
                        d_14_constrainedPrompt_: _dafny.Seq
                        d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_15_next_: _dafny.Seq
                        out6_: _dafny.Seq
                        out6_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, d_5_localBoostGroups_, _dafny.BigRational('6e0'), d_1_penaltyTokens_, _dafny.BigRational('1e1'), 12, eosToken)
                        d_15_next_ = out6_
                        d_6_steps_ = (d_6_steps_) + (1)
                        if (d_15_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_16_appendedGenerated_: _dafny.Seq
                            d_17_appendedInside_: bool
                            d_18_appendedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                            d_16_appendedGenerated_ = out7_
                            d_17_appendedInside_ = out8_
                            d_18_appendedCurrent_ = out9_
                            generated = d_16_appendedGenerated_
                            insideConstrainedOut = d_17_appendedInside_
                            currentConstrainedOut = d_18_appendedCurrent_
                    pass
            pass
        cost = d_6_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


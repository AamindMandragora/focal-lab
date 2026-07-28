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
        (d_0_helpers_).AppendTaskGuidance(lm, ((((((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Read the question carefully and use EVERY named quantity it mentions; do not drop any. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "STRICT FORMAT for << >> spans: ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(1) Inside << >> write ONLY bare identifier names exactly as the question spells them (e.g. n, n1, frac_1, w_2, total, target, sides). NEVER write curly braces inside << >> - if the question shows {n1}, write n1; if it shows {frac_1}, write frac_1 (keep the underscore). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(2) Wrap each arithmetic step in << >>, e.g. <<r * w + x>>, <<n - (a + b)>>, <<(r - n) * w>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(3) Allowed symbols inside << >>: digits, identifiers, parentheses, + - * /, // (Python integer division), and int(...). NO LaTeX, NO $, NO \\frac, NO round/ceil/floor/math.*, NO {}. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(4) When the answer is an integer count derived from dividing totals (trips, people, batches), use // not /: write <<(a + b) // capacity>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(5) When the answer is an integer count derived from fractions or percentages, wrap with int(): write <<int(n * frac_1 * frac_2)>> or <<int(n * p1 / 100)>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(6) MANDATORY final line, nothing after it: '#### <<final_expression>>' where final_expression is ONE self-contained arithmetic expression using only the symbols above. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Worked example A (counts with addition): 'She has r plants per ledge on w ledges (<<r * w>>) plus x new plants giving <<r * w + x>>. She gives away <<n * w>>. Remaining: <<r * w + x - n * w>>. #### <<r * w + x - n * w>>'. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Worked example B (integer division for trips): 'Total weight is <<n1 * w1 + n2 * w2>>. Trips needed: <<(n1 * w1 + n2 * w2) // total>>. #### <<(n1 * w1 + n2 * w2) // total>>'. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Worked example C (fractional count): 'Of n balls, frac_1 are type A: <<int(n * frac_1)>>. Of those, frac_2 are color C: <<int(n * frac_1 * frac_2)>>. #### <<int(n * frac_1 * frac_2)>>'."))))
        d_1_penaltyTokens_: _dafny.Seq
        d_1_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "$")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\frac")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\begin")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\end")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\text")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\cdot")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\times")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "round")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ceil")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "floor")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "math"))])
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
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_4_closedGenerated_: _dafny.Seq
                        d_5_closedInside_: bool
                        d_6_closedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_4_closedGenerated_ = out1_
                        d_5_closedInside_ = out2_
                        d_6_closedCurrent_ = out3_
                        generated = d_4_closedGenerated_
                        insideConstrainedOut = d_5_closedInside_
                        currentConstrainedOut = d_6_closedCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                    elif True:
                        d_7_constrainedPrompt_: _dafny.Seq
                        d_7_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_8_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_1_penaltyTokens_, _dafny.BigRational('7e0'), 12, eosToken)
                        d_8_next_ = out4_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_8_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_9_appendedGenerated_: _dafny.Seq
                            d_10_appendedInside_: bool
                            d_11_appendedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_next_)
                            d_9_appendedGenerated_ = out5_
                            d_10_appendedInside_ = out6_
                            d_11_appendedCurrent_ = out7_
                            generated = d_9_appendedGenerated_
                            insideConstrainedOut = d_10_appendedInside_
                            currentConstrainedOut = d_11_appendedCurrent_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


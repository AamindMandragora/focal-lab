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
        (d_0_helpers_).AppendTaskGuidance(lm, ((((((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step.\n"))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Method:\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(a) List every quantity named in the problem (each variable like x, n, r, w, t1, n1, n2, k_2, count, total). Do NOT skip any.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(b) Identify what the final question is asking for.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(c) Combine ALL relevant quantities with the correct operator. If items are added and some removed, the expression must include every term (e.g. r*w + x - w*n, not just w*n). For totals across categories, sum them all (count*(n1+n2+n3+n4+n5)). For ratio shares, use part/(part+rest)*total.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output rules:\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1. Put every intermediate calculation AND the final answer inside << >>. Close each << with a matching >> before any prose.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2. Inside << >>: ONE Python expression. No '=' sign, no LaTeX (no \\, $, \\frac, \\boxed), no curly braces.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3. Use exact variable names from the question, stripping braces: {n}->n, {t1}->t1, {k_2}->k_2, {count}->count, {name}->name.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4. For integer-count answers (people, items, trips, plates, harvests, episodes), use // for division and/or wrap in int(...). Patterns: <<a+b+c>>, <<total-x>>, <<int(total/per)>>, <<int(a*100/b)>>, <<(m*x)//(m+n)>>.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5. End with EXACTLY: The final answer is <<EXPR>>."))))
        d_1_latexPenalty_: _dafny.Seq
        d_1_latexPenalty_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\)")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\[")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\]")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\boxed")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "$")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "$$")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\frac")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\text")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}"))])
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_rollbackLimit_: int
        d_3_rollbackLimit_ = 48
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            d_5_endsWithOpen_: bool
                            d_5_endsWithOpen_ = ((len(d_4_next_)) >= (2)) and ((_dafny.SeqWithoutIsStrInference((d_4_next_)[(len(d_4_next_)) - (2)::])) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))))
                            if ((d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or (d_5_endsWithOpen_):
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
                    elif (len(currentConstrainedOut)) >= (d_3_rollbackLimit_):
                        d_9_rolledGenerated_: _dafny.Seq
                        d_10_rolledCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: _dafny.Seq
                        out4_, out5_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_9_rolledGenerated_ = out4_
                        d_10_rolledCurrent_ = out5_
                        generated = d_9_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_10_rolledCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                    elif True:
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_12_next_: _dafny.Seq
                        out6_: _dafny.Seq
                        out6_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, d_1_latexPenalty_, _dafny.BigRational('4e0'), eosToken)
                        d_12_next_ = out6_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_12_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_13_appendedGenerated_: _dafny.Seq
                            d_14_appendedInside_: bool
                            d_15_appendedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                            d_13_appendedGenerated_ = out7_
                            d_14_appendedInside_ = out8_
                            d_15_appendedCurrent_ = out9_
                            generated = d_13_appendedGenerated_
                            insideConstrainedOut = d_14_appendedInside_
                            currentConstrainedOut = d_15_appendedCurrent_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


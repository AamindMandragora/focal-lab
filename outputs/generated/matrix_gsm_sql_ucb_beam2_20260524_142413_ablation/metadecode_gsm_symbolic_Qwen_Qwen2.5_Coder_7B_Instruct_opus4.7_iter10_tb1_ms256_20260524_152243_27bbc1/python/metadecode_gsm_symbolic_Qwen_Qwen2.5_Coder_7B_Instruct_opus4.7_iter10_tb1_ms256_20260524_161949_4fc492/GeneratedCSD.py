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
        (d_0_helpers_).AppendTaskGuidance(lm, ((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem briefly (1-3 sentences). Output rules (STRICT):\n"))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1. Wrap every intermediate numeric expression AND the final answer in << >> delimiters. Close each << with >> before any prose.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2. Inside << >> write ONE Python expression. No '=' sign, no restated variable name, no LaTeX (no \\(, \\), \\[, \\boxed, \\frac, $), no curly braces.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3. Use the EXACT variable names from the question, removing the braces: {n}->n, {t1}->t1, {k_2}->k_2, {frac}->frac, {p1}->p1, {name}->name, {sides}->sides. Do NOT invent new names. Do NOT keep {x} placeholders inside << >>.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4. Use Python operators +, -, *, /. Always use / for division (never //).\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5. When the answer must be a whole-number count or quantity, wrap it in int(...). Patterns: sum <<a + b + c>>; difference <<total - x>>; whole-number quotient <<int(total / per)>>; percentage <<int(a * 100 / b)>>; compound percentage <<int((a + b) * 100 / (c + d))>>.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6. End with exactly: The final answer is <<EXPR>>."))))
        d_1_latexPenalty_: _dafny.Seq
        d_1_latexPenalty_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\)")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\[")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\]")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\boxed")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "$")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "$$")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\frac")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\text")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\times")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\cdot")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}"))])
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_rollbackLimit_: int
        d_3_rollbackLimit_ = 36
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


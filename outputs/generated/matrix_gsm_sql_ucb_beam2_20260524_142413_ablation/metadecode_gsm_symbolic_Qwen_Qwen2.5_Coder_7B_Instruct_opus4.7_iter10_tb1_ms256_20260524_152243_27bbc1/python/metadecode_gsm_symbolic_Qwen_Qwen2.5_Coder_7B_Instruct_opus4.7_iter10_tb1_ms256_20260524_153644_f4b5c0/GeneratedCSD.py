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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((((((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem with brief reasoning. Wrap every arithmetic computation and the final answer in << >> delimiters using Python-style expressions.\n\n"))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "STRICT RULES:\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1. Use << >> ONLY. Never use LaTeX: no \\(, \\), \\[, \\boxed, $...$, \\frac, \\times, \\cdot.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2. Use // for integer division (whole-number quantities like minutes, people, items). Do NOT use /.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3. Wrap the FINAL answer expression in int(...) when the answer is a count.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4. Use the variable letters/names from the problem literally (e.g. t, d, y, n1, c1, frac1) inside << >>.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5. No '=' inside << >>; only the bare expression. Close every << with >> before continuing prose.\n\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FOLLOW THESE PATTERNS EXACTLY (gold-style examples):\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "  - Speed/coverage:  <<y // d * t>>\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "  - Total cost:      <<int(n1*c1 + n2*c2 + c3)>>\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "  - Difference:      <<total + n2 - n1>>\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "  - Piecewise rate:  <<int((frac1 * t) + frac2*(total - t))>>\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "  - Final line:      'The final answer is <<int(EXPR)>>.'\n\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Now solve. Keep prose to 1-2 sentences. Prefer // over /. Wrap the final answer in int(...)."))))
        d_1_divisionAndEqPenalty_: _dafny.Seq
        d_1_divisionAndEqPenalty_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " / ")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " /")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/ ")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " = ")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " =")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "= ")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\)")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\[")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\]")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\boxed")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\frac")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\times")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\cdot")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "$"))])
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
                        out4_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, d_1_divisionAndEqPenalty_, _dafny.BigRational('6e0'), eosToken)
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


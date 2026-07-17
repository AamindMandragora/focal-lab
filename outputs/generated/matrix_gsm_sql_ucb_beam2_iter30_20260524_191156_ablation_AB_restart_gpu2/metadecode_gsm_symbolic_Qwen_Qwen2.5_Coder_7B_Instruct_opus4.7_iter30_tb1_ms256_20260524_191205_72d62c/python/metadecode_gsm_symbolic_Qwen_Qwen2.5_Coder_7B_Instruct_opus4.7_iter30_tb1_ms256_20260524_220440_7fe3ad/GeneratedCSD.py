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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are solving a parametric math word problem. Produce a very short answer in EXACTLY this form, with the constrained block at the very end:\nFinal answer: <<EXPR>>\n\nEXPR rules (strict):\n- A single Python arithmetic expression in the ORIGINAL variable names from the problem (for example n, n1, n2, t, d, y, k, k1, k2, cur, c1, c2, c3, w1, w2, w3, frac, frac1, frac2, total, size, cal, count, mult, m, p1, r1). Use the names exactly; never write curly braces like {n}; never substitute numeric values.\n- Use EVERY input variable that affects the answer; do not drop any relevant variable.\n- Use // (Python floor division) when dividing whole-item counts (servings, trips, groups, fitting items into bins). Floor-divide EARLY when the rate is per-group integer (e.g. 't minutes per d miles' for y miles is y//d*t, NOT y*t/d).\n- Wrap the WHOLE expression in int(...) when the answer should be a whole number derived from a non-integer computation (percentages, money-like sums of float-cost items, rounded rates). When in doubt for money totals, wrap with int(...).\n- Allowed operators only: + - * / // ( ) and int(). Do not use min, max, round, math, %.\n\nStyle examples (final-line only):\nFinal answer: <<y//d*t>>\nFinal answer: <<int(n1*c1 + n2*c2 + c3)>>\nFinal answer: <<n - (n1*w1) - (n2*w2) - int(n3*w3)>>\nFinal answer: <<int((frac1*t) + frac2*(total-t))>>\nFinal answer: <<count*(n1+n2+n3+n4+n5)>>\nFinal answer: <<n - k*x>>\nFinal answer: <<int(100*(k1+k2)/(n1+n2))>>\nFinal answer: <<t + (k*t)//(mult*m)>>\nFinal answer: <<size//n*(total-spent)//cal>>\nFinal answer: <<(2*n1+n2)*cn + (2*m2-m1)*cm>>\n\nBefore the final line, write AT MOST one very short plain-English sentence (or none). NEVER use << or >> outside the Final answer line. Output exactly ONE << >> block, at the end, and always close << with >>. Stop generating immediately after the closing >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_hardOpenAt_: int
        d_3_hardOpenAt_ = 200
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_chunkBudget_: int
                        d_4_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_4_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        if (d_1_steps_) >= (d_3_hardOpenAt_):
                            d_5_openedG_: _dafny.Seq
                            d_6_openedI_: bool
                            d_7_openedC_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedG_ = out0_
                            d_6_openedI_ = out1_
                            d_7_openedC_ = out2_
                            generated = d_5_openedG_
                            insideConstrainedOut = d_6_openedI_
                            currentConstrainedOut = d_7_openedC_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_cap_: int
                            d_8_cap_ = (d_3_hardOpenAt_) - (d_1_steps_)
                            d_9_capped_: int
                            if (d_4_chunkBudget_) < (d_8_cap_):
                                d_9_capped_ = d_4_chunkBudget_
                            elif True:
                                d_9_capped_ = d_8_cap_
                            d_10_chunkedG_: _dafny.Seq
                            d_11_stoppedOpen_: bool
                            d_12_stoppedEos_: bool
                            d_13_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_capped_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_10_chunkedG_ = out3_
                            d_11_stoppedOpen_ = out4_
                            d_12_stoppedEos_ = out5_
                            d_13_stepsUsed_ = out6_
                            generated = d_10_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                            if d_12_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_11_stoppedOpen_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif (d_13_stepsUsed_) == (0):
                                raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_14_closedGenerated_: _dafny.Seq
                        d_15_closedInside_: bool
                        d_16_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_closedGenerated_ = out7_
                        d_15_closedInside_ = out8_
                        d_16_closedCurrent_ = out9_
                        generated = d_14_closedGenerated_
                        insideConstrainedOut = d_15_closedInside_
                        currentConstrainedOut = d_16_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_18_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                        d_18_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_18_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_19_appendedGenerated_: _dafny.Seq
                            d_20_appendedInside_: bool
                            d_21_appendedCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                            d_19_appendedGenerated_ = out11_
                            d_20_appendedInside_ = out12_
                            d_21_appendedCurrent_ = out13_
                            generated = d_19_appendedGenerated_
                            insideConstrainedOut = d_20_appendedInside_
                            currentConstrainedOut = d_21_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


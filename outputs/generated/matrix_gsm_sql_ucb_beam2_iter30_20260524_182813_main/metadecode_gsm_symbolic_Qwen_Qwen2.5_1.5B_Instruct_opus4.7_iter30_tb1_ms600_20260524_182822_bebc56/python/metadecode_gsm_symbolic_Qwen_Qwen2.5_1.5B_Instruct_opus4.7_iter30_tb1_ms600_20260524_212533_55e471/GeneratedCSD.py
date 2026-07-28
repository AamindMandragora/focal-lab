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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step using the template variable names from the problem (e.g., n, k, x, k_2, frac_1, n_1). After every arithmetic calculation, write it as <<expression=number>> where number is the evaluated numeric result. CRITICAL: never write <<expression>> without =number; always include '=' followed by the numeric value before '>>'. Examples: <<3+4=7>>, <<24//4=6>>, <<n*frac_1=8>>, <<int(n*frac_1*frac_2)=4>>, <<(n_1+n_2)//2=5>>. Use Python integer division '//' whenever the result must be a whole count (people, items, months, trips). Inside << >> use ONLY digits, identifier names (letters, digits, underscores), the operators + - * / //, parentheses, the function int(...), and exactly one '=' before the numeric result. No words, no units, no curly braces, no exclamation marks inside << >>. End with: The answer is <<expression=number>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_2_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_2_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                            if (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_3_closedGenerated_: _dafny.Seq
                        d_4_closedInside_: bool
                        d_5_closedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_3_closedGenerated_ = out1_
                        d_4_closedInside_ = out2_
                        d_5_closedCurrent_ = out3_
                        generated = d_3_closedGenerated_
                        insideConstrainedOut = d_4_closedInside_
                        currentConstrainedOut = d_5_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_6_hasEquals_: bool
                        d_6_hasEquals_ = False
                        d_7_tokensAfterEquals_: int
                        d_7_tokensAfterEquals_ = 0
                        d_8_ii_: int
                        d_8_ii_ = 0
                        while (d_8_ii_) < (len(currentConstrainedOut)):
                            d_9_t_: _dafny.Seq
                            d_9_t_ = (currentConstrainedOut)[d_8_ii_]
                            d_10_foundHere_: bool
                            d_10_foundHere_ = False
                            d_11_jj_: int
                            d_11_jj_ = 0
                            while (d_11_jj_) < (len(d_9_t_)):
                                if ((d_9_t_)[d_11_jj_]) == (_dafny.CodePoint('=')):
                                    d_10_foundHere_ = True
                                d_11_jj_ = (d_11_jj_) + (1)
                            if d_6_hasEquals_:
                                d_7_tokensAfterEquals_ = (d_7_tokensAfterEquals_) + (1)
                            if d_10_foundHere_:
                                d_6_hasEquals_ = True
                            d_8_ii_ = (d_8_ii_) + (1)
                        d_12_constrainedPrompt_: _dafny.Seq
                        d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_13_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (len(currentConstrainedOut)) >= (22):
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('12e0'), eosToken)
                            d_13_next_ = out4_
                        elif not(d_6_hasEquals_):
                            if (len(currentConstrainedOut)) >= (3):
                                out5_: _dafny.Seq
                                out5_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.BigRational('5e0'), eosToken)
                                d_13_next_ = out5_
                            elif True:
                                out6_: _dafny.Seq
                                out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_13_next_ = out6_
                        elif (d_7_tokensAfterEquals_) == (0):
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_13_next_ = out7_
                        elif True:
                            if (len(currentConstrainedOut)) >= (12):
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('8e0'), eosToken)
                                d_13_next_ = out8_
                            elif True:
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('4e0'), eosToken)
                                d_13_next_ = out9_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_13_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_14_appendedGenerated_: _dafny.Seq
                            d_15_appendedInside_: bool
                            d_16_appendedCurrent_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                            d_14_appendedGenerated_ = out10_
                            d_15_appendedInside_ = out11_
                            d_16_appendedCurrent_ = out12_
                            generated = d_14_appendedGenerated_
                            insideConstrainedOut = d_15_appendedInside_
                            currentConstrainedOut = d_16_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


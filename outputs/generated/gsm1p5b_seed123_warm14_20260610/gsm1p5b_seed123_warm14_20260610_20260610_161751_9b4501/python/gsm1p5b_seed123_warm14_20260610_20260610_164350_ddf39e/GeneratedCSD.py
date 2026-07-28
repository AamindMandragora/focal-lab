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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem. Show your reasoning, then at the very end write the final arithmetic expression inside << >>. Inside the delimiters use ONLY: integers, +, -, *, /, (, ). No variables, no text, no curly braces.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanBudget_: int
        if (maxSteps) >= (80):
            d_2_spanBudget_ = 25
        elif (maxSteps) >= (40):
            d_2_spanBudget_ = 15
        elif (maxSteps) >= (10):
            d_2_spanBudget_ = (_dafny.euclidian_division(maxSteps, 4)) + (2)
        elif True:
            d_2_spanBudget_ = maxSteps
        d_3_freePhaseLimit_: int
        if (maxSteps) > ((d_2_spanBudget_) + (1)):
            d_3_freePhaseLimit_ = ((maxSteps) - (d_2_spanBudget_)) - (1)
        elif True:
            d_3_freePhaseLimit_ = 0
        d_4_penaltyToks_: _dafny.Seq
        d_4_penaltyToks_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "%")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "#")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "@")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "$")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "^")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "&")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "|")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "~")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "`")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "?")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "m")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "b")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "p")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "s")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "a")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "e")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "x")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "y")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "z")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "f")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "g")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "h")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "i")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "j")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "l")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "o")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "q")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "r")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "u")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "v")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_1_steps_) >= (d_3_freePhaseLimit_):
                            if ((d_1_steps_) + (1)) <= (maxSteps):
                                d_5_og_: _dafny.Seq
                                d_6_oi_: bool
                                d_7_oc_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_5_og_ = out0_
                                d_6_oi_ = out1_
                                d_7_oc_ = out2_
                                generated = d_5_og_
                                insideConstrainedOut = d_6_oi_
                                currentConstrainedOut = d_7_oc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                if ((d_1_steps_) + (2)) <= (maxSteps):
                                    d_9_og_: _dafny.Seq
                                    d_10_oi_: bool
                                    d_11_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_9_og_ = out4_
                                    d_10_oi_ = out5_
                                    d_11_oc_ = out6_
                                    generated = d_9_og_
                                    insideConstrainedOut = d_10_oi_
                                    currentConstrainedOut = d_11_oc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_freePhaseLimit_ = 0
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_cg_: _dafny.Seq
                        d_13_ci_: bool
                        d_14_cc_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_cg_ = out7_
                        d_13_ci_ = out8_
                        d_14_cc_ = out9_
                        generated = d_12_cg_
                        insideConstrainedOut = d_13_ci_
                        currentConstrainedOut = d_14_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_16_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, d_4_penaltyToks_, _dafny.BigRational('6e0'), eosToken)
                        d_16_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_16_next_) == (eosToken):
                            d_17_rg_: _dafny.Seq
                            d_18_rc_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out11_, out12_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_17_rg_ = out11_
                            d_18_rc_ = out12_
                            generated = d_17_rg_
                            currentConstrainedOut = d_18_rc_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                d_19_cg2_: _dafny.Seq
                                d_20_ci2_: bool
                                d_21_cc2_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_19_cg2_ = out13_
                                d_20_ci2_ = out14_
                                d_21_cc2_ = out15_
                                generated = d_19_cg2_
                                insideConstrainedOut = d_20_ci2_
                                currentConstrainedOut = d_21_cc2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            raise _dafny.Break("0")
                        elif True:
                            d_22_ag_: _dafny.Seq
                            d_23_ai_: bool
                            d_24_ac_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                            d_22_ag_ = out16_
                            d_23_ai_ = out17_
                            d_24_ac_ = out18_
                            generated = d_22_ag_
                            insideConstrainedOut = d_23_ai_
                            currentConstrainedOut = d_24_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_25_cg_: _dafny.Seq
                d_26_ci_: bool
                d_27_cc_: _dafny.Seq
                out19_: _dafny.Seq
                out20_: bool
                out21_: _dafny.Seq
                out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_25_cg_ = out19_
                d_26_ci_ = out20_
                d_27_cc_ = out21_
                generated = d_25_cg_
                insideConstrainedOut = d_26_ci_
                currentConstrainedOut = d_27_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_28_rg_: _dafny.Seq
                d_29_rc_: _dafny.Seq
                out22_: _dafny.Seq
                out23_: _dafny.Seq
                out22_, out23_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                d_28_rg_ = out22_
                d_29_rc_ = out23_
                generated = d_28_rg_
                currentConstrainedOut = d_29_rc_
                if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                    d_30_cg2_: _dafny.Seq
                    d_31_ci2_: bool
                    d_32_cc2_: _dafny.Seq
                    out24_: _dafny.Seq
                    out25_: bool
                    out26_: _dafny.Seq
                    out24_, out25_, out26_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_30_cg2_ = out24_
                    d_31_ci2_ = out25_
                    d_32_cc2_ = out26_
                    generated = d_30_cg2_
                    insideConstrainedOut = d_31_ci2_
                    currentConstrainedOut = d_32_cc2_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


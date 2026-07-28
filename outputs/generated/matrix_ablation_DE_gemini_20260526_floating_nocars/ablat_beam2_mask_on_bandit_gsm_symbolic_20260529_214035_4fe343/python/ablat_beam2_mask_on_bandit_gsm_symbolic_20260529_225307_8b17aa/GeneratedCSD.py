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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step in concise prose. Wrap important intermediate symbolic arithmetic expressions and the final answer in visible << >> delimiters. Inside delimiters, use only compact arithmetic syntax with numbers, variables, parentheses, and operators such as +, -, *, /, and //. Put no words or units inside delimiters."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_mathGroups_: _dafny.Seq
        d_2_mathGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "."))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "int")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ceil")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "floor"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "x")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "y")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "d")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "r")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "m")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total"))])])
        d_3_combinedGroups_: _dafny.Seq
        d_3_combinedGroups_ = (validTokenGroups) + (d_2_mathGroups_)
        d_4_penaltyTokens_: _dafny.Seq
        d_4_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ")), eosToken])
        d_5_steps_: int
        d_5_steps_ = 0
        d_6_forceAfter_: int
        d_6_forceAfter_ = 180
        with _dafny.label("0"):
            while (d_5_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_5_steps_) >= (d_6_forceAfter_):
                            d_7_openedGenerated_: _dafny.Seq
                            d_8_openedInside_: bool
                            d_9_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_openedGenerated_ = out0_
                            d_8_openedInside_ = out1_
                            d_9_openedCurrent_ = out2_
                            generated = d_7_openedGenerated_
                            insideConstrainedOut = d_8_openedInside_
                            currentConstrainedOut = d_9_openedCurrent_
                            d_5_steps_ = (d_5_steps_) + (1)
                        elif True:
                            d_10_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_next_ = out3_
                            d_5_steps_ = (d_5_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                if (d_5_steps_) < (maxSteps):
                                    d_11_eosOpenedGenerated_: _dafny.Seq
                                    d_12_eosOpenedInside_: bool
                                    d_13_eosOpenedCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_11_eosOpenedGenerated_ = out4_
                                    d_12_eosOpenedInside_ = out5_
                                    d_13_eosOpenedCurrent_ = out6_
                                    generated = d_11_eosOpenedGenerated_
                                    insideConstrainedOut = d_12_eosOpenedInside_
                                    currentConstrainedOut = d_13_eosOpenedCurrent_
                                    d_5_steps_ = (d_5_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                pass
                            elif (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
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
                        d_5_steps_ = (d_5_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_18_nextConstrained_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, d_3_combinedGroups_, _dafny.BigRational('8e0'), d_4_penaltyTokens_, _dafny.BigRational('9e0'), 128, eosToken)
                        d_18_nextConstrained_ = out10_
                        d_5_steps_ = (d_5_steps_) + (1)
                        if (((d_18_nextConstrained_) != (eosToken)) and ((d_18_nextConstrained_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))))) and ((d_18_nextConstrained_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
                            d_19_valid_: bool
                            out11_: bool
                            out11_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_18_nextConstrained_)
                            d_19_valid_ = out11_
                            if d_19_valid_:
                                d_20_appendedGenerated_: _dafny.Seq
                                d_21_appendedInside_: bool
                                d_22_appendedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_nextConstrained_)
                                d_20_appendedGenerated_ = out12_
                                d_21_appendedInside_ = out13_
                                d_22_appendedCurrent_ = out14_
                                generated = d_20_appendedGenerated_
                                insideConstrainedOut = d_21_appendedInside_
                                currentConstrainedOut = d_22_appendedCurrent_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_5_steps_) < (maxSteps)):
                                    d_23_appendedClosedGenerated_: _dafny.Seq
                                    d_24_appendedClosedInside_: bool
                                    d_25_appendedClosedCurrent_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_23_appendedClosedGenerated_ = out15_
                                    d_24_appendedClosedInside_ = out16_
                                    d_25_appendedClosedCurrent_ = out17_
                                    generated = d_23_appendedClosedGenerated_
                                    insideConstrainedOut = d_24_appendedClosedInside_
                                    currentConstrainedOut = d_25_appendedClosedCurrent_
                                    d_5_steps_ = (d_5_steps_) + (1)
                                    raise _dafny.Break("0")
                        elif True:
                            if (d_5_steps_) < (maxSteps):
                                d_26_candidates_: _dafny.Seq
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                                d_26_candidates_ = out18_
                                d_5_steps_ = (d_5_steps_) + (1)
                                if (len(d_26_candidates_)) > (0):
                                    d_27_repair_: _dafny.Seq
                                    d_27_repair_ = (d_26_candidates_)[0]
                                    if (((d_27_repair_) != (eosToken)) and ((d_27_repair_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))))) and ((d_27_repair_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
                                        d_28_repairValid_: bool
                                        out19_: bool
                                        out19_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_27_repair_)
                                        d_28_repairValid_ = out19_
                                        if d_28_repairValid_:
                                            d_29_repairGenerated_: _dafny.Seq
                                            d_30_repairInside_: bool
                                            d_31_repairCurrent_: _dafny.Seq
                                            out20_: _dafny.Seq
                                            out21_: bool
                                            out22_: _dafny.Seq
                                            out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_repair_)
                                            d_29_repairGenerated_ = out20_
                                            d_30_repairInside_ = out21_
                                            d_31_repairCurrent_ = out22_
                                            generated = d_29_repairGenerated_
                                            insideConstrainedOut = d_30_repairInside_
                                            currentConstrainedOut = d_31_repairCurrent_
                                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_5_steps_) < (maxSteps)):
                                                d_32_repairClosedGenerated_: _dafny.Seq
                                                d_33_repairClosedInside_: bool
                                                d_34_repairClosedCurrent_: _dafny.Seq
                                                out23_: _dafny.Seq
                                                out24_: bool
                                                out25_: _dafny.Seq
                                                out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                d_32_repairClosedGenerated_ = out23_
                                                d_33_repairClosedInside_ = out24_
                                                d_34_repairClosedCurrent_ = out25_
                                                generated = d_32_repairClosedGenerated_
                                                insideConstrainedOut = d_33_repairClosedInside_
                                                currentConstrainedOut = d_34_repairClosedCurrent_
                                                d_5_steps_ = (d_5_steps_) + (1)
                                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_5_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


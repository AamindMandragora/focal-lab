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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Write each arithmetic computation inside << >> delimiters with actual numbers only. Example: <<3*4=12>>. Use only numeric values, operators (+,-,*,/), and = inside << >>. Never use variable names or template placeholders inside << >>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_spanSteps_: int
        d_3_spanSteps_ = 0
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
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_3_spanSteps_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_5_closedGenerated_: _dafny.Seq
                        d_6_closedInside_: bool
                        d_7_closedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_5_closedGenerated_ = out1_
                        d_6_closedInside_ = out2_
                        d_7_closedCurrent_ = out3_
                        generated = d_5_closedGenerated_
                        insideConstrainedOut = d_6_closedInside_
                        currentConstrainedOut = d_7_closedCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_3_spanSteps_ = 0
                    elif (d_3_spanSteps_) >= (40):
                        d_8_rolledGenerated_: _dafny.Seq
                        d_9_rolledCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: _dafny.Seq
                        out4_, out5_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_8_rolledGenerated_ = out4_
                        d_9_rolledCurrent_ = out5_
                        generated = d_8_rolledGenerated_
                        currentConstrainedOut = d_9_rolledCurrent_
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_10_closedGenerated_: _dafny.Seq
                            d_11_closedInside_: bool
                            d_12_closedCurrent_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_closedGenerated_ = out6_
                            d_11_closedInside_ = out7_
                            d_12_closedCurrent_ = out8_
                            generated = d_10_closedGenerated_
                            insideConstrainedOut = d_11_closedInside_
                            currentConstrainedOut = d_12_closedCurrent_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_spanSteps_ = 0
                        elif True:
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_14_next_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_14_next_ = out9_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_spanSteps_ = (d_3_spanSteps_) + (1)
                            if (d_14_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_15_appendedGenerated_: _dafny.Seq
                                d_16_appendedInside_: bool
                                d_17_appendedCurrent_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                d_15_appendedGenerated_ = out10_
                                d_16_appendedInside_ = out11_
                                d_17_appendedCurrent_ = out12_
                                generated = d_15_appendedGenerated_
                                insideConstrainedOut = d_16_appendedInside_
                                currentConstrainedOut = d_17_appendedCurrent_
                    elif True:
                        d_18_narrow_: bool
                        out13_: bool
                        out13_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                        d_18_narrow_ = out13_
                        if (d_18_narrow_) and ((len(currentConstrainedOut)) > (0)):
                            d_19_rolledGenerated_: _dafny.Seq
                            d_20_rolledCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: _dafny.Seq
                            out14_, out15_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_19_rolledGenerated_ = out14_
                            d_20_rolledCurrent_ = out15_
                            generated = d_19_rolledGenerated_
                            currentConstrainedOut = d_20_rolledCurrent_
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_21_closedGenerated_: _dafny.Seq
                                d_22_closedInside_: bool
                                d_23_closedCurrent_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_21_closedGenerated_ = out16_
                                d_22_closedInside_ = out17_
                                d_23_closedCurrent_ = out18_
                                generated = d_21_closedGenerated_
                                insideConstrainedOut = d_22_closedInside_
                                currentConstrainedOut = d_23_closedCurrent_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_3_spanSteps_ = 0
                            elif True:
                                d_24_constrainedPrompt_: _dafny.Seq
                                d_24_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_25_next_: _dafny.Seq
                                out19_: _dafny.Seq
                                out19_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_25_next_ = out19_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_3_spanSteps_ = (d_3_spanSteps_) + (1)
                                if (d_25_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_26_appendedGenerated_: _dafny.Seq
                                    d_27_appendedInside_: bool
                                    d_28_appendedCurrent_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                                    d_26_appendedGenerated_ = out20_
                                    d_27_appendedInside_ = out21_
                                    d_28_appendedCurrent_ = out22_
                                    generated = d_26_appendedGenerated_
                                    insideConstrainedOut = d_27_appendedInside_
                                    currentConstrainedOut = d_28_appendedCurrent_
                        elif True:
                            d_29_constrainedPrompt_: _dafny.Seq
                            d_29_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_30_next_: _dafny.Seq
                            out23_: _dafny.Seq
                            out23_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_30_next_ = out23_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_spanSteps_ = (d_3_spanSteps_) + (1)
                            if (d_30_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_31_appendedGenerated_: _dafny.Seq
                                d_32_appendedInside_: bool
                                d_33_appendedCurrent_: _dafny.Seq
                                out24_: _dafny.Seq
                                out25_: bool
                                out26_: _dafny.Seq
                                out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_next_)
                                d_31_appendedGenerated_ = out24_
                                d_32_appendedInside_ = out25_
                                d_33_appendedCurrent_ = out26_
                                generated = d_31_appendedGenerated_
                                insideConstrainedOut = d_32_appendedInside_
                                currentConstrainedOut = d_33_appendedCurrent_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


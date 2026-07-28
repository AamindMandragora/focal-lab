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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write a concise, correct SQL query. Use only the tables and columns from the schema. Output the query directly without extra conditions.")))
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_2_openGenerated_: _dafny.Seq
            d_3_openInside_: bool
            d_4_openCurrent_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_2_openGenerated_ = out0_
            d_3_openInside_ = out1_
            d_4_openCurrent_ = out2_
            generated = d_2_openGenerated_
            insideConstrainedOut = d_3_openInside_
            currentConstrainedOut = d_4_openCurrent_
            d_1_steps_ = (d_1_steps_) + (1)
        d_5_veryNarrowThreshold_: int
        d_5_veryNarrowThreshold_ = 5
        d_6_lengthGuard_: int
        d_6_lengthGuard_ = 80
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        d_7_closedGenerated_: _dafny.Seq
                        d_8_closedInside_: bool
                        d_9_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_7_closedGenerated_ = out3_
                        d_8_closedInside_ = out4_
                        d_9_closedCurrent_ = out5_
                        generated = d_7_closedGenerated_
                        insideConstrainedOut = d_8_closedInside_
                        currentConstrainedOut = d_9_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif ((len(currentConstrainedOut)) >= (d_6_lengthGuard_)) and (((d_1_steps_) + (2)) <= (maxSteps)):
                        d_10_rolledGenerated_: _dafny.Seq
                        d_11_rolledCurrent_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: _dafny.Seq
                        out6_, out7_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_10_rolledGenerated_ = out6_
                        d_11_rolledCurrent_ = out7_
                        if (parser).IsCompletePrefix(d_11_rolledCurrent_):
                            generated = d_10_rolledGenerated_
                            currentConstrainedOut = d_11_rolledCurrent_
                            d_12_closedGenerated_: _dafny.Seq
                            d_13_closedInside_: bool
                            d_14_closedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_12_closedGenerated_ = out8_
                            d_13_closedInside_ = out9_
                            d_14_closedCurrent_ = out10_
                            generated = d_12_closedGenerated_
                            insideConstrainedOut = d_13_closedInside_
                            currentConstrainedOut = d_14_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_15_constrainedPrompt_: _dafny.Seq
                            d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_16_next_: _dafny.Seq
                            d_17_wasConstrained_: bool
                            out11_: _dafny.Seq
                            out12_: bool
                            out11_, out12_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_16_next_ = out11_
                            d_17_wasConstrained_ = out12_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_16_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_18_appendedGenerated_: _dafny.Seq
                                d_19_appendedInside_: bool
                                d_20_appendedCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                d_18_appendedGenerated_ = out13_
                                d_19_appendedInside_ = out14_
                                d_20_appendedCurrent_ = out15_
                                generated = d_18_appendedGenerated_
                                insideConstrainedOut = d_19_appendedInside_
                                currentConstrainedOut = d_20_appendedCurrent_
                    elif True:
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_22_validCount_: int
                        out16_: int
                        out16_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_22_validCount_ = out16_
                        d_23_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if ((d_22_validCount_) <= (d_5_veryNarrowThreshold_)) and ((len(validTokenGroups)) > (0)):
                            out17_: _dafny.Seq
                            out17_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), eosToken)
                            d_23_next_ = out17_
                        elif True:
                            d_24_wasConstrained_: bool = False
                            out18_: _dafny.Seq
                            out19_: bool
                            out18_, out19_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_23_next_ = out18_
                            d_24_wasConstrained_ = out19_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_23_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_25_appendedGenerated_: _dafny.Seq
                            d_26_appendedInside_: bool
                            d_27_appendedCurrent_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                            d_25_appendedGenerated_ = out20_
                            d_26_appendedInside_ = out21_
                            d_27_appendedCurrent_ = out22_
                            generated = d_25_appendedGenerated_
                            insideConstrainedOut = d_26_appendedInside_
                            currentConstrainedOut = d_27_appendedCurrent_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_28_rolledGenerated_: _dafny.Seq
            d_29_rolledCurrent_: _dafny.Seq
            out23_: _dafny.Seq
            out24_: _dafny.Seq
            out23_, out24_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_28_rolledGenerated_ = out23_
            d_29_rolledCurrent_ = out24_
            if ((parser).IsCompletePrefix(d_29_rolledCurrent_)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                generated = d_28_rolledGenerated_
                currentConstrainedOut = d_29_rolledCurrent_
                d_30_closedGenerated_: _dafny.Seq
                d_31_closedInside_: bool
                d_32_closedCurrent_: _dafny.Seq
                out25_: _dafny.Seq
                out26_: bool
                out27_: _dafny.Seq
                out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_30_closedGenerated_ = out25_
                d_31_closedInside_ = out26_
                d_32_closedCurrent_ = out27_
                generated = d_30_closedGenerated_
                insideConstrainedOut = d_31_closedInside_
                currentConstrainedOut = d_32_closedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


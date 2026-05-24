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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Write every arithmetic computation inside << >> delimiters, and make the final computation also appear inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openArmed_: bool
        d_2_openArmed_ = False
        d_3_completedSpans_: int
        d_3_completedSpans_ = 0
        d_4_lateOpenThreshold_: int
        d_4_lateOpenThreshold_ = 18
        d_5_narrowThreshold_: int
        d_5_narrowThreshold_ = 8
        d_6_rollbackLimit_: int
        d_6_rollbackLimit_ = 24
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (not(insideConstrainedOut)) and (d_2_openArmed_):
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
                        d_2_openArmed_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif not(insideConstrainedOut):
                        d_10_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_10_next_ = out3_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_10_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                            if ((d_3_completedSpans_) == (0)) and ((len(generated)) >= ((len(generatedPrefix)) + (d_4_lateOpenThreshold_))):
                                d_2_openArmed_ = True
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                            if (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_11_enteredGenerated_: _dafny.Seq
                                d_12_enteredInside_: bool
                                d_13_enteredCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_11_enteredGenerated_ = out4_
                                d_12_enteredInside_ = out5_
                                d_13_enteredCurrent_ = out6_
                                generated = d_11_enteredGenerated_
                                insideConstrainedOut = d_12_enteredInside_
                                currentConstrainedOut = d_13_enteredCurrent_
                                d_2_openArmed_ = False
                            elif True:
                                d_14_sinceEq_: int
                                out7_: int
                                out7_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_14_sinceEq_ = out7_
                                d_15_sincePlus_: int
                                out8_: int
                                out8_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")))
                                d_15_sincePlus_ = out8_
                                d_16_sinceMinus_: int
                                out9_: int
                                out9_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")))
                                d_16_sinceMinus_ = out9_
                                d_17_sinceStar_: int
                                out10_: int
                                out10_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")))
                                d_17_sinceStar_ = out10_
                                d_18_sinceSlash_: int
                                out11_: int
                                out11_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")))
                                d_18_sinceSlash_ = out11_
                                d_19_sinceIs_: int
                                out12_: int
                                out12_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "is")))
                                d_19_sinceIs_ = out12_
                                d_20_sinceAre_: int
                                out13_: int
                                out13_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "are")))
                                d_20_sinceAre_ = out13_
                                d_21_sinceTotal_: int
                                out14_: int
                                out14_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")))
                                d_21_sinceTotal_ = out14_
                                d_22_sinceLeft_: int
                                out15_: int
                                out15_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "left")))
                                d_22_sinceLeft_ = out15_
                                d_23_sinceCost_: int
                                out16_: int
                                out16_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "cost")))
                                d_23_sinceCost_ = out16_
                                d_24_sinceEach_: int
                                out17_: int
                                out17_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "each")))
                                d_24_sinceEach_ = out17_
                                if (((d_3_completedSpans_) == (0)) and ((len(generated)) >= ((len(generatedPrefix)) + (d_4_lateOpenThreshold_)))) and ((((((((((((d_14_sinceEq_) <= (2)) or ((d_15_sincePlus_) <= (2))) or ((d_16_sinceMinus_) <= (2))) or ((d_17_sinceStar_) <= (2))) or ((d_18_sinceSlash_) <= (2))) or ((d_19_sinceIs_) <= (2))) or ((d_20_sinceAre_) <= (2))) or ((d_21_sinceTotal_) <= (2))) or ((d_22_sinceLeft_) <= (2))) or ((d_23_sinceCost_) <= (2))) or ((d_24_sinceEach_) <= (2))):
                                    d_2_openArmed_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_25_closedGenerated_: _dafny.Seq
                        d_26_closedInside_: bool
                        d_27_closedCurrent_: _dafny.Seq
                        out18_: _dafny.Seq
                        out19_: bool
                        out20_: _dafny.Seq
                        out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_25_closedGenerated_ = out18_
                        d_26_closedInside_ = out19_
                        d_27_closedCurrent_ = out20_
                        generated = d_25_closedGenerated_
                        insideConstrainedOut = d_26_closedInside_
                        currentConstrainedOut = d_27_closedCurrent_
                        d_3_completedSpans_ = (d_3_completedSpans_) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_6_rollbackLimit_):
                        d_28_rolledGenerated_: _dafny.Seq
                        d_29_rolledCurrent_: _dafny.Seq
                        out21_: _dafny.Seq
                        out22_: _dafny.Seq
                        out21_, out22_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_28_rolledGenerated_ = out21_
                        d_29_rolledCurrent_ = out22_
                        generated = d_28_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_29_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_30_stablePrefix_: _dafny.Seq
                        d_30_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_31_constrainedPrompt_: _dafny.Seq
                        d_31_constrainedPrompt_ = (prompt) + (d_30_stablePrefix_)
                        d_32_validCount_: int
                        out23_: int
                        out23_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_32_validCount_ = out23_
                        if (d_32_validCount_) <= (d_5_narrowThreshold_):
                            d_33_nextInside_: _dafny.Seq
                            out24_: _dafny.Seq
                            out24_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_31_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_33_nextInside_ = out24_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_33_nextInside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_34_appendedGenerated_: _dafny.Seq
                                d_35_appendedInside_: bool
                                d_36_appendedCurrent_: _dafny.Seq
                                out25_: _dafny.Seq
                                out26_: bool
                                out27_: _dafny.Seq
                                out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_33_nextInside_)
                                d_34_appendedGenerated_ = out25_
                                d_35_appendedInside_ = out26_
                                d_36_appendedCurrent_ = out27_
                                generated = d_34_appendedGenerated_
                                insideConstrainedOut = d_35_appendedInside_
                                currentConstrainedOut = d_36_appendedCurrent_
                        elif True:
                            d_37_remainingInside_: int
                            d_37_remainingInside_ = (maxSteps) - (d_1_steps_)
                            d_38_symbolBudget_: int
                            if (stepTokenBudget) == (0):
                                d_38_symbolBudget_ = 1
                            elif (stepTokenBudget) > (d_37_remainingInside_):
                                d_38_symbolBudget_ = d_37_remainingInside_
                            elif True:
                                d_38_symbolBudget_ = stepTokenBudget
                            d_39_symbolGenerated_: _dafny.Seq
                            d_40_symbolOut_: _dafny.Seq
                            d_41_hitEos_: bool
                            d_42_stepsUsed_: int
                            out28_: _dafny.Seq
                            out29_: _dafny.Seq
                            out30_: bool
                            out31_: int
                            out28_, out29_, out30_, out31_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_31_constrainedPrompt_, generated, currentConstrainedOut, d_38_symbolBudget_, eosToken)
                            d_39_symbolGenerated_ = out28_
                            d_40_symbolOut_ = out29_
                            d_41_hitEos_ = out30_
                            d_42_stepsUsed_ = out31_
                            generated = d_39_symbolGenerated_
                            currentConstrainedOut = d_40_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_42_stepsUsed_)
                            if d_41_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


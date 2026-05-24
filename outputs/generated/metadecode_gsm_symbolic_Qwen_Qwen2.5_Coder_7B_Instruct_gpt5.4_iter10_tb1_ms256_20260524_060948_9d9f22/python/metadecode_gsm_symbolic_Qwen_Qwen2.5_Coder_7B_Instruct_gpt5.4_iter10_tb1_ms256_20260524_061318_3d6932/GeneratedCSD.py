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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Every arithmetic computation must appear inside << >> delimiters, and the final computation should also be inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openArmed_: bool
        d_2_openArmed_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (not(insideConstrainedOut)) and (d_2_openArmed_):
                        d_3_openedGenerated_: _dafny.Seq
                        d_4_openedInside_: bool
                        d_5_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_3_openedGenerated_ = out0_
                        d_4_openedInside_ = out1_
                        d_5_openedCurrent_ = out2_
                        generated = d_3_openedGenerated_
                        insideConstrainedOut = d_4_openedInside_
                        currentConstrainedOut = d_5_openedCurrent_
                        d_2_openArmed_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif not(insideConstrainedOut):
                        d_6_remaining_: int
                        d_6_remaining_ = (maxSteps) - (d_1_steps_)
                        d_7_chunkBudget_: int
                        if (d_6_remaining_) <= (4):
                            d_7_chunkBudget_ = d_6_remaining_
                        elif True:
                            d_7_chunkBudget_ = 4
                        d_8_chunkedG_: _dafny.Seq
                        d_9_stoppedOpen_: bool
                        d_10_stoppedEos_: bool
                        d_11_stepsUsed_: int
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: bool
                        out6_: int
                        out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_8_chunkedG_ = out3_
                        d_9_stoppedOpen_ = out4_
                        d_10_stoppedEos_ = out5_
                        d_11_stepsUsed_ = out6_
                        generated = d_8_chunkedG_
                        d_1_steps_ = (d_1_steps_) + (d_11_stepsUsed_)
                        if d_10_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_9_stoppedOpen_:
                            d_12_enteredGenerated_: _dafny.Seq
                            d_13_enteredInside_: bool
                            d_14_enteredCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_12_enteredGenerated_ = out7_
                            d_13_enteredInside_ = out8_
                            d_14_enteredCurrent_ = out9_
                            generated = d_12_enteredGenerated_
                            insideConstrainedOut = d_13_enteredInside_
                            currentConstrainedOut = d_14_enteredCurrent_
                            d_2_openArmed_ = False
                        elif True:
                            d_15_sinceEq_: int
                            out10_: int
                            out10_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                            d_15_sinceEq_ = out10_
                            d_16_sincePlus_: int
                            out11_: int
                            out11_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")))
                            d_16_sincePlus_ = out11_
                            d_17_sinceMinus_: int
                            out12_: int
                            out12_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")))
                            d_17_sinceMinus_ = out12_
                            d_18_sinceStar_: int
                            out13_: int
                            out13_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")))
                            d_18_sinceStar_ = out13_
                            d_19_sinceSlash_: int
                            out14_: int
                            out14_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")))
                            d_19_sinceSlash_ = out14_
                            d_20_sinceIs_: int
                            out15_: int
                            out15_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "is")))
                            d_20_sinceIs_ = out15_
                            d_21_sinceAre_: int
                            out16_: int
                            out16_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "are")))
                            d_21_sinceAre_ = out16_
                            d_22_sinceTotal_: int
                            out17_: int
                            out17_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")))
                            d_22_sinceTotal_ = out17_
                            d_23_sinceLeft_: int
                            out18_: int
                            out18_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "left")))
                            d_23_sinceLeft_ = out18_
                            d_24_sinceCost_: int
                            out19_: int
                            out19_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "cost")))
                            d_24_sinceCost_ = out19_
                            d_25_sinceEach_: int
                            out20_: int
                            out20_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "each")))
                            d_25_sinceEach_ = out20_
                            if (((((((((((d_15_sinceEq_) <= (2)) or ((d_16_sincePlus_) <= (2))) or ((d_17_sinceMinus_) <= (2))) or ((d_18_sinceStar_) <= (2))) or ((d_19_sinceSlash_) <= (2))) or ((d_20_sinceIs_) <= (2))) or ((d_21_sinceAre_) <= (2))) or ((d_22_sinceTotal_) <= (2))) or ((d_23_sinceLeft_) <= (2))) or ((d_24_sinceCost_) <= (2))) or ((d_25_sinceEach_) <= (2)):
                                d_2_openArmed_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_26_closedGenerated_: _dafny.Seq
                        d_27_closedInside_: bool
                        d_28_closedCurrent_: _dafny.Seq
                        out21_: _dafny.Seq
                        out22_: bool
                        out23_: _dafny.Seq
                        out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_26_closedGenerated_ = out21_
                        d_27_closedInside_ = out22_
                        d_28_closedCurrent_ = out23_
                        generated = d_26_closedGenerated_
                        insideConstrainedOut = d_27_closedInside_
                        currentConstrainedOut = d_28_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_29_stablePrefix_: _dafny.Seq
                        d_29_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_30_constrainedPrompt_: _dafny.Seq
                        d_30_constrainedPrompt_ = (prompt) + (d_29_stablePrefix_)
                        d_31_validCount_: int
                        out24_: int
                        out24_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_31_validCount_ = out24_
                        if (d_31_validCount_) <= (8):
                            d_32_next_: _dafny.Seq
                            out25_: _dafny.Seq
                            out25_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_30_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_32_next_ = out25_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_32_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_33_appendedGenerated_: _dafny.Seq
                                d_34_appendedInside_: bool
                                d_35_appendedCurrent_: _dafny.Seq
                                out26_: _dafny.Seq
                                out27_: bool
                                out28_: _dafny.Seq
                                out26_, out27_, out28_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_32_next_)
                                d_33_appendedGenerated_ = out26_
                                d_34_appendedInside_ = out27_
                                d_35_appendedCurrent_ = out28_
                                generated = d_33_appendedGenerated_
                                insideConstrainedOut = d_34_appendedInside_
                                currentConstrainedOut = d_35_appendedCurrent_
                        elif True:
                            d_36_remainingInside_: int
                            d_36_remainingInside_ = (maxSteps) - (d_1_steps_)
                            d_37_symbolBudget_: int
                            if (stepTokenBudget) == (0):
                                d_37_symbolBudget_ = 1
                            elif (stepTokenBudget) > (d_36_remainingInside_):
                                d_37_symbolBudget_ = d_36_remainingInside_
                            elif True:
                                d_37_symbolBudget_ = stepTokenBudget
                            d_38_symbolGenerated_: _dafny.Seq
                            d_39_symbolOut_: _dafny.Seq
                            d_40_hitEos_: bool
                            d_41_stepsUsed2_: int
                            out29_: _dafny.Seq
                            out30_: _dafny.Seq
                            out31_: bool
                            out32_: int
                            out29_, out30_, out31_, out32_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_30_constrainedPrompt_, generated, currentConstrainedOut, d_37_symbolBudget_, eosToken)
                            d_38_symbolGenerated_ = out29_
                            d_39_symbolOut_ = out30_
                            d_40_hitEos_ = out31_
                            d_41_stepsUsed2_ = out32_
                            generated = d_38_symbolGenerated_
                            currentConstrainedOut = d_39_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_41_stepsUsed2_)
                            if d_40_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


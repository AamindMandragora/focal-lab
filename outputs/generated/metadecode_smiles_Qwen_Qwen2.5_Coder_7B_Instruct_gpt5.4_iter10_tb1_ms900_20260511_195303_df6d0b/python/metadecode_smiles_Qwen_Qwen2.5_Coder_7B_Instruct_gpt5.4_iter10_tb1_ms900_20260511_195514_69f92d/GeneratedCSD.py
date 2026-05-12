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
        d_2_openedExplicitly_: bool
        d_2_openedExplicitly_ = False
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 10
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_2_openedExplicitly_):
                            d_4_remainingChunk_: int
                            d_4_remainingChunk_ = (maxSteps) - (d_1_steps_)
                            d_5_chunkBudget_: int
                            if (d_4_remainingChunk_) > (2):
                                d_5_chunkBudget_ = 2
                            elif True:
                                d_5_chunkBudget_ = d_4_remainingChunk_
                            if (d_5_chunkBudget_) == (0):
                                raise _dafny.Break("0")
                            d_6_chunkedGenerated_: _dafny.Seq
                            d_7_stoppedOpen_: bool
                            d_8_stoppedEos_: bool
                            d_9_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_6_chunkedGenerated_ = out0_
                            d_7_stoppedOpen_ = out1_
                            d_8_stoppedEos_ = out2_
                            d_9_stepsUsed_ = out3_
                            generated = d_6_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                            if d_8_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_7_stoppedOpen_:
                                d_10_enteredGenerated_: _dafny.Seq
                                d_11_enteredInside_: bool
                                d_12_enteredCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_10_enteredGenerated_ = out4_
                                d_11_enteredInside_ = out5_
                                d_12_enteredCurrent_ = out6_
                                generated = d_10_enteredGenerated_
                                insideConstrainedOut = d_11_enteredInside_
                                currentConstrainedOut = d_12_enteredCurrent_
                            elif (d_1_steps_) < (maxSteps):
                                d_13_openedGenerated_: _dafny.Seq
                                d_14_openedInside_: bool
                                d_15_openedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_13_openedGenerated_ = out7_
                                d_14_openedInside_ = out8_
                                d_15_openedCurrent_ = out9_
                                generated = d_13_openedGenerated_
                                insideConstrainedOut = d_14_openedInside_
                                currentConstrainedOut = d_15_openedCurrent_
                                d_2_openedExplicitly_ = True
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_16_nextOutside_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_16_nextOutside_ = out10_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_16_nextOutside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_16_nextOutside_]))
                                if (d_16_nextOutside_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_17_enteredGenerated2_: _dafny.Seq
                                    d_18_enteredInside2_: bool
                                    d_19_enteredCurrent2_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_17_enteredGenerated2_ = out11_
                                    d_18_enteredInside2_ = out12_
                                    d_19_enteredCurrent2_ = out13_
                                    generated = d_17_enteredGenerated2_
                                    insideConstrainedOut = d_18_enteredInside2_
                                    currentConstrainedOut = d_19_enteredCurrent2_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_20_closedGenerated_: _dafny.Seq
                        d_21_closedInside_: bool
                        d_22_closedCurrent_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: bool
                        out16_: _dafny.Seq
                        out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_20_closedGenerated_ = out14_
                        d_21_closedInside_ = out15_
                        d_22_closedCurrent_ = out16_
                        generated = d_20_closedGenerated_
                        insideConstrainedOut = d_21_closedInside_
                        currentConstrainedOut = d_22_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_23_stablePrefix_: _dafny.Seq
                        d_23_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_24_constrainedPrompt_: _dafny.Seq
                        d_24_constrainedPrompt_ = (prompt) + (d_23_stablePrefix_)
                        d_25_validCount_: int
                        out17_: int
                        out17_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_25_validCount_ = out17_
                        if (d_25_validCount_) <= (d_3_narrowThreshold_):
                            d_26_next_: _dafny.Seq
                            d_26_next_ = eosToken
                            if (len(currentConstrainedOut)) == (0):
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_26_next_ = out18_
                            elif (len(currentConstrainedOut)) >= (8):
                                out19_: _dafny.Seq
                                out19_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_26_next_ = out19_
                            elif True:
                                out20_: _dafny.Seq
                                out20_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                d_26_next_ = out20_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_26_next_) == (eosToken):
                                d_27_repairedGenerated_: _dafny.Seq
                                d_28_repairedCurrent_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: _dafny.Seq
                                out21_, out22_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_27_repairedGenerated_ = out21_
                                d_28_repairedCurrent_ = out22_
                                generated = d_27_repairedGenerated_
                                insideConstrainedOut = True
                                currentConstrainedOut = d_28_repairedCurrent_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_29_closedGenerated2_: _dafny.Seq
                                    d_30_closedInside2_: bool
                                    d_31_closedCurrent2_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out24_: bool
                                    out25_: _dafny.Seq
                                    out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_29_closedGenerated2_ = out23_
                                    d_30_closedInside2_ = out24_
                                    d_31_closedCurrent2_ = out25_
                                    generated = d_29_closedGenerated2_
                                    insideConstrainedOut = d_30_closedInside2_
                                    currentConstrainedOut = d_31_closedCurrent2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_32_appendedGenerated_: _dafny.Seq
                                d_33_appendedInside_: bool
                                d_34_appendedCurrent_: _dafny.Seq
                                out26_: _dafny.Seq
                                out27_: bool
                                out28_: _dafny.Seq
                                out26_, out27_, out28_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                                d_32_appendedGenerated_ = out26_
                                d_33_appendedInside_ = out27_
                                d_34_appendedCurrent_ = out28_
                                generated = d_32_appendedGenerated_
                                insideConstrainedOut = d_33_appendedInside_
                                currentConstrainedOut = d_34_appendedCurrent_
                        elif True:
                            d_35_remaining_: int
                            d_35_remaining_ = (maxSteps) - (d_1_steps_)
                            d_36_symbolBudget_: int
                            if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_35_remaining_)):
                                d_36_symbolBudget_ = d_35_remaining_
                            elif True:
                                d_36_symbolBudget_ = stepTokenBudget
                            if (d_36_symbolBudget_) == (0):
                                raise _dafny.Break("0")
                            d_37_symbolGenerated_: _dafny.Seq
                            d_38_symbolCurrent_: _dafny.Seq
                            d_39_hitEos_: bool
                            d_40_stepsUsed2_: int
                            out29_: _dafny.Seq
                            out30_: _dafny.Seq
                            out31_: bool
                            out32_: int
                            out29_, out30_, out31_, out32_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_24_constrainedPrompt_, generated, currentConstrainedOut, d_36_symbolBudget_, eosToken)
                            d_37_symbolGenerated_ = out29_
                            d_38_symbolCurrent_ = out30_
                            d_39_hitEos_ = out31_
                            d_40_stepsUsed2_ = out32_
                            generated = d_37_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_38_symbolCurrent_
                            d_1_steps_ = (d_1_steps_) + (d_40_stepsUsed2_)
                            if d_39_hitEos_:
                                d_41_repairedGenerated2_: _dafny.Seq
                                d_42_repairedCurrent2_: _dafny.Seq
                                out33_: _dafny.Seq
                                out34_: _dafny.Seq
                                out33_, out34_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_41_repairedGenerated2_ = out33_
                                d_42_repairedCurrent2_ = out34_
                                generated = d_41_repairedGenerated2_
                                insideConstrainedOut = True
                                currentConstrainedOut = d_42_repairedCurrent2_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_43_closedGenerated3_: _dafny.Seq
                                    d_44_closedInside3_: bool
                                    d_45_closedCurrent3_: _dafny.Seq
                                    out35_: _dafny.Seq
                                    out36_: bool
                                    out37_: _dafny.Seq
                                    out35_, out36_, out37_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_43_closedGenerated3_ = out35_
                                    d_44_closedInside3_ = out36_
                                    d_45_closedCurrent3_ = out37_
                                    generated = d_43_closedGenerated3_
                                    insideConstrainedOut = d_44_closedInside3_
                                    currentConstrainedOut = d_45_closedCurrent3_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


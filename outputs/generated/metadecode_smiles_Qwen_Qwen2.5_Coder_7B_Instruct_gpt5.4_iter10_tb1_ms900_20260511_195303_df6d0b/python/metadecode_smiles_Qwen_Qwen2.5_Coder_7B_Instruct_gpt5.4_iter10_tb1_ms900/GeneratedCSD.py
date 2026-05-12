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
        d_3_narrowThreshold_ = 12
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
                            d_16_next_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_16_next_ = out10_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_16_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_16_next_]))
                                if (d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
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
                            if (len(currentConstrainedOut)) >= (8):
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_26_next_ = out18_
                            elif True:
                                out19_: _dafny.Seq
                                out19_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                d_26_next_ = out19_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_26_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_27_appendedGenerated_: _dafny.Seq
                                d_28_appendedInside_: bool
                                d_29_appendedCurrent_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                                d_27_appendedGenerated_ = out20_
                                d_28_appendedInside_ = out21_
                                d_29_appendedCurrent_ = out22_
                                generated = d_27_appendedGenerated_
                                insideConstrainedOut = d_28_appendedInside_
                                currentConstrainedOut = d_29_appendedCurrent_
                        elif True:
                            d_30_remaining_: int
                            d_30_remaining_ = (maxSteps) - (d_1_steps_)
                            d_31_symbolBudget_: int
                            if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_30_remaining_)):
                                d_31_symbolBudget_ = d_30_remaining_
                            elif True:
                                d_31_symbolBudget_ = stepTokenBudget
                            d_32_symbolGenerated_: _dafny.Seq
                            d_33_symbolCurrent_: _dafny.Seq
                            d_34_hitEos_: bool
                            d_35_stepsUsed2_: int
                            out23_: _dafny.Seq
                            out24_: _dafny.Seq
                            out25_: bool
                            out26_: int
                            out23_, out24_, out25_, out26_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_24_constrainedPrompt_, generated, currentConstrainedOut, d_31_symbolBudget_, eosToken)
                            d_32_symbolGenerated_ = out23_
                            d_33_symbolCurrent_ = out24_
                            d_34_hitEos_ = out25_
                            d_35_stepsUsed2_ = out26_
                            generated = d_32_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_33_symbolCurrent_
                            d_1_steps_ = (d_1_steps_) + (d_35_stepsUsed2_)
                            if d_34_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


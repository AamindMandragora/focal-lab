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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query inside << and >>, with no explanation.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_outsideBudget_: int
        d_2_outsideBudget_ = 6
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((len(generated)) == (len(generatedPrefix))) and ((d_2_outsideBudget_) > (0)):
                            d_3_remaining0_: int
                            d_3_remaining0_ = (maxSteps) - (d_1_steps_)
                            d_4_chunkBudget_: int
                            if (d_2_outsideBudget_) > (d_3_remaining0_):
                                d_4_chunkBudget_ = d_3_remaining0_
                            elif True:
                                d_4_chunkBudget_ = d_2_outsideBudget_
                            d_5_chunkedG_: _dafny.Seq
                            d_6_stoppedOpen_: bool
                            d_7_stoppedEos_: bool
                            d_8_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_5_chunkedG_ = out0_
                            d_6_stoppedOpen_ = out1_
                            d_7_stoppedEos_ = out2_
                            d_8_stepsUsed_ = out3_
                            generated = d_5_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                            if d_7_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_6_stoppedOpen_:
                                d_9_enteredGenerated_: _dafny.Seq
                                d_10_enteredInside_: bool
                                d_11_enteredCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_9_enteredGenerated_ = out4_
                                d_10_enteredInside_ = out5_
                                d_11_enteredCurrent_ = out6_
                                generated = d_9_enteredGenerated_
                                insideConstrainedOut = d_10_enteredInside_
                                currentConstrainedOut = d_11_enteredCurrent_
                            elif (d_1_steps_) < (maxSteps):
                                d_12_openedGenerated_: _dafny.Seq
                                d_13_openedInside_: bool
                                d_14_openedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_12_openedGenerated_ = out7_
                                d_13_openedInside_ = out8_
                                d_14_openedCurrent_ = out9_
                                generated = d_12_openedGenerated_
                                insideConstrainedOut = d_13_openedInside_
                                currentConstrainedOut = d_14_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_15_openedGenerated2_: _dafny.Seq
                            d_16_openedInside2_: bool
                            d_17_openedCurrent2_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_15_openedGenerated2_ = out10_
                            d_16_openedInside2_ = out11_
                            d_17_openedCurrent2_ = out12_
                            generated = d_15_openedGenerated2_
                            insideConstrainedOut = d_16_openedInside2_
                            currentConstrainedOut = d_17_openedCurrent2_
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_18_closedGenerated_: _dafny.Seq
                        d_19_closedInside_: bool
                        d_20_closedCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_18_closedGenerated_ = out13_
                        d_19_closedInside_ = out14_
                        d_20_closedCurrent_ = out15_
                        generated = d_18_closedGenerated_
                        insideConstrainedOut = d_19_closedInside_
                        currentConstrainedOut = d_20_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_21_stablePrefix_: _dafny.Seq
                        d_21_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_22_constrainedPrompt_: _dafny.Seq
                        d_22_constrainedPrompt_ = (prompt) + (d_21_stablePrefix_)
                        d_23_remaining_: int
                        d_23_remaining_ = (maxSteps) - (d_1_steps_)
                        d_24_symbolBudget_: int
                        if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_23_remaining_)):
                            d_24_symbolBudget_ = d_23_remaining_
                        elif True:
                            d_24_symbolBudget_ = stepTokenBudget
                        d_25_symbolGenerated_: _dafny.Seq
                        d_26_symbolOut_: _dafny.Seq
                        d_27_hitEos_: bool
                        d_28_stepsUsed2_: int
                        out16_: _dafny.Seq
                        out17_: _dafny.Seq
                        out18_: bool
                        out19_: int
                        out16_, out17_, out18_, out19_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_22_constrainedPrompt_, generated, currentConstrainedOut, d_24_symbolBudget_, eosToken)
                        d_25_symbolGenerated_ = out16_
                        d_26_symbolOut_ = out17_
                        d_27_hitEos_ = out18_
                        d_28_stepsUsed2_ = out19_
                        generated = d_25_symbolGenerated_
                        currentConstrainedOut = d_26_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_28_stepsUsed2_)
                        if (insideConstrainedOut) and (not((parser).IsValidPrefix(currentConstrainedOut))):
                            d_29_rolledGenerated_: _dafny.Seq
                            d_30_rolledCurrent_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: _dafny.Seq
                            out20_, out21_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_29_rolledGenerated_ = out20_
                            d_30_rolledCurrent_ = out21_
                            generated = d_29_rolledGenerated_
                            currentConstrainedOut = d_30_rolledCurrent_
                        if d_27_hitEos_:
                            raise _dafny.Break("0")
                    pass
            pass
        if ((insideConstrainedOut) and ((d_1_steps_) < (maxSteps))) and ((parser).IsCompletePrefix(currentConstrainedOut)):
            d_31_closedGenerated2_: _dafny.Seq
            d_32_closedInside2_: bool
            d_33_closedCurrent2_: _dafny.Seq
            out22_: _dafny.Seq
            out23_: bool
            out24_: _dafny.Seq
            out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_31_closedGenerated2_ = out22_
            d_32_closedInside2_ = out23_
            d_33_closedCurrent2_ = out24_
            generated = d_31_closedGenerated2_
            insideConstrainedOut = d_32_closedInside2_
            currentConstrainedOut = d_33_closedCurrent2_
            d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


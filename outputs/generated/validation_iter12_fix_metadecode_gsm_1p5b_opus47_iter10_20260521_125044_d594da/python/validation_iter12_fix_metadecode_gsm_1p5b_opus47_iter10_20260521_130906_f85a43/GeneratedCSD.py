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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. CRITICAL: every arithmetic calculation MUST be wrapped in << >> delimiters. Examples: <<2+3=5>>, <<10*4=40>>, <<100-25=75>>. Use << >> for EVERY computation in your reasoning. End with '#### <final_answer>'.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanStart_: int
        d_2_spanStart_ = 0
        d_3_spanLimit_: int
        d_3_spanLimit_ = 20
        d_4_tokensSinceSpan_: int
        d_4_tokensSinceSpan_ = 0
        d_5_forceOpenAfter_: int
        d_5_forceOpenAfter_ = 40
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_4_tokensSinceSpan_) >= (d_5_forceOpenAfter_)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                            d_6_sawEquals_: bool
                            d_6_sawEquals_ = False
                            d_7_lookback_: int
                            if (len(generated)) < (8):
                                d_7_lookback_ = len(generated)
                            elif True:
                                d_7_lookback_ = 8
                            d_8_i_: int
                            d_8_i_ = (len(generated)) - (d_7_lookback_)
                            while (d_8_i_) < (len(generated)):
                                if (((generated)[d_8_i_]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))) or (((generated)[d_8_i_]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " =")))):
                                    d_6_sawEquals_ = True
                                d_8_i_ = (d_8_i_) + (1)
                            if d_6_sawEquals_:
                                d_9_openedGenerated_: _dafny.Seq
                                d_10_openedInside_: bool
                                d_11_openedCurrent_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_9_openedGenerated_ = out0_
                                d_10_openedInside_ = out1_
                                d_11_openedCurrent_ = out2_
                                generated = d_9_openedGenerated_
                                insideConstrainedOut = d_10_openedInside_
                                currentConstrainedOut = d_11_openedCurrent_
                                d_2_spanStart_ = (d_1_steps_) + (1)
                                d_4_tokensSinceSpan_ = 0
                                d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Continue("0")
                            elif True:
                                d_4_tokensSinceSpan_ = 0
                        d_12_remaining_: int
                        d_12_remaining_ = (maxSteps) - (d_1_steps_)
                        d_13_chunkBudget_: int
                        if (d_12_remaining_) < (15):
                            d_13_chunkBudget_ = d_12_remaining_
                        elif True:
                            d_13_chunkBudget_ = 15
                        if (d_13_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        d_14_oldLen_: int
                        d_14_oldLen_ = len(generated)
                        d_15_chunkedG_: _dafny.Seq
                        d_16_stoppedOpen_: bool
                        d_17_stoppedEos_: bool
                        d_18_stepsUsed_: int
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: bool
                        out6_: int
                        out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_13_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_15_chunkedG_ = out3_
                        d_16_stoppedOpen_ = out4_
                        d_17_stoppedEos_ = out5_
                        d_18_stepsUsed_ = out6_
                        generated = d_15_chunkedG_
                        d_1_steps_ = (d_1_steps_) + (d_18_stepsUsed_)
                        d_19_grew_: int
                        d_19_grew_ = (len(generated)) - (d_14_oldLen_)
                        d_4_tokensSinceSpan_ = (d_4_tokensSinceSpan_) + (d_19_grew_)
                        if d_17_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_16_stoppedOpen_:
                            d_20_openedGenerated_: _dafny.Seq
                            d_21_openedInside_: bool
                            d_22_openedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_20_openedGenerated_ = out7_
                            d_21_openedInside_ = out8_
                            d_22_openedCurrent_ = out9_
                            generated = d_20_openedGenerated_
                            insideConstrainedOut = d_21_openedInside_
                            currentConstrainedOut = d_22_openedCurrent_
                            d_2_spanStart_ = d_1_steps_
                            d_4_tokensSinceSpan_ = 0
                        elif (d_18_stepsUsed_) == (0):
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_23_closedGenerated_: _dafny.Seq
                        d_24_closedInside_: bool
                        d_25_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_23_closedGenerated_ = out10_
                        d_24_closedInside_ = out11_
                        d_25_closedCurrent_ = out12_
                        generated = d_23_closedGenerated_
                        insideConstrainedOut = d_24_closedInside_
                        currentConstrainedOut = d_25_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_4_tokensSinceSpan_ = 0
                    elif ((d_1_steps_) - (d_2_spanStart_)) >= (d_3_spanLimit_):
                        d_26_rolledGenerated_: _dafny.Seq
                        d_27_rolledCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_26_rolledGenerated_ = out13_
                        d_27_rolledCurrent_ = out14_
                        generated = d_26_rolledGenerated_
                        currentConstrainedOut = d_27_rolledCurrent_
                        insideConstrainedOut = True
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            if (d_1_steps_) < (maxSteps):
                                d_28_closedGenerated_: _dafny.Seq
                                d_29_closedInside_: bool
                                d_30_closedCurrent_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_28_closedGenerated_ = out15_
                                d_29_closedInside_ = out16_
                                d_30_closedCurrent_ = out17_
                                generated = d_28_closedGenerated_
                                insideConstrainedOut = d_29_closedInside_
                                currentConstrainedOut = d_30_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_4_tokensSinceSpan_ = 0
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_31_stablePrefix_: _dafny.Seq
                        d_31_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_32_constrainedPrompt_: _dafny.Seq
                        d_32_constrainedPrompt_ = (prompt) + (d_31_stablePrefix_)
                        d_33_remaining_: int
                        d_33_remaining_ = (maxSteps) - (d_1_steps_)
                        d_34_spanRemaining_: int
                        d_34_spanRemaining_ = (d_3_spanLimit_) - ((d_1_steps_) - (d_2_spanStart_))
                        d_35_symBudget_: int
                        if (d_33_remaining_) < (d_34_spanRemaining_):
                            d_35_symBudget_ = d_33_remaining_
                        elif True:
                            d_35_symBudget_ = d_34_spanRemaining_
                        if (d_35_symBudget_) == (0):
                            d_35_symBudget_ = 1
                        if (d_35_symBudget_) > (4):
                            d_35_symBudget_ = 4
                        d_36_symbolGenerated_: _dafny.Seq
                        d_37_symbolOut_: _dafny.Seq
                        d_38_hitEos_: bool
                        d_39_stepsUsed_: int
                        out18_: _dafny.Seq
                        out19_: _dafny.Seq
                        out20_: bool
                        out21_: int
                        out18_, out19_, out20_, out21_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_32_constrainedPrompt_, generated, currentConstrainedOut, d_35_symBudget_, eosToken)
                        d_36_symbolGenerated_ = out18_
                        d_37_symbolOut_ = out19_
                        d_38_hitEos_ = out20_
                        d_39_stepsUsed_ = out21_
                        generated = d_36_symbolGenerated_
                        currentConstrainedOut = d_37_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_39_stepsUsed_)
                        if d_38_hitEos_:
                            raise _dafny.Break("0")
                        if (d_39_stepsUsed_) == (0):
                            d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


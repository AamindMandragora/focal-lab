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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are solving a grade-school math word problem. Read carefully, identify the quantities, and reason step by step in plain prose. EVERY arithmetic computation MUST be written as <<expression=value>>, for example <<3*4=12>>, <<100-25=75>>, or <<48/6=8>>. At the very end, write '#### ' followed by the final numeric answer wrapped as <<value>>, e.g. '#### <<42>>'. Double-check each computation before committing to it.\n\nWorked example:\nQuestion: A bakery has 4 trays with 6 muffins each. After selling 9 muffins, how many remain?\nAnswer: The bakery starts with <<4*6=24>> muffins. After selling 9, they have <<24-9=15>> muffins remaining. #### <<15>>\n\nNow solve the user's problem in the same format. Be precise with the arithmetic.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 16
        d_3_rollbackLimit_: int
        d_3_rollbackLimit_ = 48
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_1_steps_)
                        d_5_chunkBudget_: int
                        if (d_4_remaining_) < (32):
                            d_5_chunkBudget_ = d_4_remaining_
                        elif True:
                            d_5_chunkBudget_ = 32
                        if (d_5_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        d_6_chunkedG_: _dafny.Seq
                        d_7_stoppedOpen_: bool
                        d_8_stoppedEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_chunkedG_ = out0_
                        d_7_stoppedOpen_ = out1_
                        d_8_stoppedEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        generated = d_6_chunkedG_
                        d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                        if d_8_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_7_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif (d_9_stepsUsed_) == (0):
                            d_10_next_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_next_ = out4_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                                if (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_11_closedGenerated_: _dafny.Seq
                        d_12_closedInside_: bool
                        d_13_closedCurrent_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_11_closedGenerated_ = out5_
                        d_12_closedInside_ = out6_
                        d_13_closedCurrent_ = out7_
                        generated = d_11_closedGenerated_
                        insideConstrainedOut = d_12_closedInside_
                        currentConstrainedOut = d_13_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_3_rollbackLimit_):
                        d_14_rolledGenerated_: _dafny.Seq
                        d_15_rolledCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: _dafny.Seq
                        out8_, out9_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_14_rolledGenerated_ = out8_
                        d_15_rolledCurrent_ = out9_
                        generated = d_14_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_15_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_16_stablePrefix_: _dafny.Seq
                        d_16_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (d_16_stablePrefix_)
                        d_18_validCount_: int
                        out10_: int
                        out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_18_validCount_ = out10_
                        if (d_18_validCount_) <= (d_2_narrowThreshold_):
                            d_19_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                            d_19_next_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_19_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_20_appendedGenerated_: _dafny.Seq
                                d_21_appendedInside_: bool
                                d_22_appendedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                d_20_appendedGenerated_ = out12_
                                d_21_appendedInside_ = out13_
                                d_22_appendedCurrent_ = out14_
                                generated = d_20_appendedGenerated_
                                insideConstrainedOut = d_21_appendedInside_
                                currentConstrainedOut = d_22_appendedCurrent_
                        elif True:
                            d_23_remaining2_: int
                            d_23_remaining2_ = (maxSteps) - (d_1_steps_)
                            d_24_symbolBudget_: int
                            if (d_23_remaining2_) < (8):
                                d_24_symbolBudget_ = d_23_remaining2_
                            elif True:
                                d_24_symbolBudget_ = 8
                            if (d_24_symbolBudget_) == (0):
                                d_1_steps_ = maxSteps
                            elif True:
                                d_25_symbolGenerated_: _dafny.Seq
                                d_26_symbolOut_: _dafny.Seq
                                d_27_hitEos_: bool
                                d_28_stepsUsed_: int
                                out15_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: int
                                out15_, out16_, out17_, out18_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_17_constrainedPrompt_, generated, currentConstrainedOut, d_24_symbolBudget_, eosToken)
                                d_25_symbolGenerated_ = out15_
                                d_26_symbolOut_ = out16_
                                d_27_hitEos_ = out17_
                                d_28_stepsUsed_ = out18_
                                generated = d_25_symbolGenerated_
                                currentConstrainedOut = d_26_symbolOut_
                                d_1_steps_ = (d_1_steps_) + (d_28_stepsUsed_)
                                if d_27_hitEos_:
                                    raise _dafny.Break("0")
                                elif (d_28_stepsUsed_) == (0):
                                    d_29_next_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out19_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_29_next_ = out19_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_29_next_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_30_appendedGenerated_: _dafny.Seq
                                        d_31_appendedInside_: bool
                                        d_32_appendedCurrent_: _dafny.Seq
                                        out20_: _dafny.Seq
                                        out21_: bool
                                        out22_: _dafny.Seq
                                        out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_next_)
                                        d_30_appendedGenerated_ = out20_
                                        d_31_appendedInside_ = out21_
                                        d_32_appendedCurrent_ = out22_
                                        generated = d_30_appendedGenerated_
                                        insideConstrainedOut = d_31_appendedInside_
                                        currentConstrainedOut = d_32_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


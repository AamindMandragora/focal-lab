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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step. Wrap every intermediate arithmetic expression and the final numeric answer inside << ... >> markers. Close every << with a >> as soon as the expression is complete.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_chunkBudget_: int
                        d_2_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_2_chunkBudget_) > (16):
                            d_2_chunkBudget_ = 16
                        if (d_2_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        d_3_chunkedG_: _dafny.Seq
                        d_4_stoppedOpen_: bool
                        d_5_stoppedEos_: bool
                        d_6_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_3_chunkedG_ = out0_
                        d_4_stoppedOpen_ = out1_
                        d_5_stoppedEos_ = out2_
                        d_6_stepsUsed_ = out3_
                        generated = d_3_chunkedG_
                        d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                        if d_5_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_4_stoppedOpen_:
                            d_7_enteredG_: _dafny.Seq
                            d_8_enteredInside_: bool
                            d_9_enteredCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_7_enteredG_ = out4_
                            d_8_enteredInside_ = out5_
                            d_9_enteredCurrent_ = out6_
                            generated = d_7_enteredG_
                            insideConstrainedOut = d_8_enteredInside_
                            currentConstrainedOut = d_9_enteredCurrent_
                        elif (d_6_stepsUsed_) == (0):
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_10_closedGenerated_: _dafny.Seq
                        d_11_closedInside_: bool
                        d_12_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_10_closedGenerated_ = out7_
                        d_11_closedInside_ = out8_
                        d_12_closedCurrent_ = out9_
                        generated = d_10_closedGenerated_
                        insideConstrainedOut = d_11_closedInside_
                        currentConstrainedOut = d_12_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_13_stablePrefix_: _dafny.Seq
                        d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_14_constrainedPrompt_: _dafny.Seq
                        d_14_constrainedPrompt_ = (prompt) + (d_13_stablePrefix_)
                        d_15_validCount_: int
                        out10_: int
                        out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_15_validCount_ = out10_
                        if (d_15_validCount_) <= (4):
                            d_16_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_16_next_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_16_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_17_valid_: bool
                                out12_: bool
                                out12_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_16_next_)
                                d_17_valid_ = out12_
                                if d_17_valid_:
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
                            d_21_remaining_: int
                            d_21_remaining_ = (maxSteps) - (d_1_steps_)
                            d_22_symbolBudget_: int
                            if (d_21_remaining_) > (8):
                                d_22_symbolBudget_ = 8
                            elif True:
                                d_22_symbolBudget_ = d_21_remaining_
                            if (d_22_symbolBudget_) == (0):
                                raise _dafny.Break("0")
                            d_23_symbolGenerated_: _dafny.Seq
                            d_24_symbolOut_: _dafny.Seq
                            d_25_hitEos_: bool
                            d_26_stepsUsed_: int
                            out16_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: int
                            out16_, out17_, out18_, out19_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_14_constrainedPrompt_, generated, currentConstrainedOut, d_22_symbolBudget_, eosToken)
                            d_23_symbolGenerated_ = out16_
                            d_24_symbolOut_ = out17_
                            d_25_hitEos_ = out18_
                            d_26_stepsUsed_ = out19_
                            generated = d_23_symbolGenerated_
                            currentConstrainedOut = d_24_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_26_stepsUsed_)
                            if d_25_hitEos_:
                                raise _dafny.Break("0")
                            if (d_26_stepsUsed_) == (0):
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


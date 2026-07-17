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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the word problem step by step. Use visible GSM calculator spans exactly like <<expression=result>> for arithmetic, and finish with a final line exactly of the form #### answer. If a calculator span has already been opened, complete that span briefly and then continue the solution normally.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_chunkLimit_: int
        d_2_chunkLimit_ = 12
        d_3_forcedSpanDone_: bool
        d_3_forcedSpanDone_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (not(insideConstrainedOut)) and (not(d_3_forcedSpanDone_)):
                        d_4_openedGenerated_: _dafny.Seq
                        d_5_openedInside_: bool
                        d_6_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_4_openedGenerated_ = out0_
                        d_5_openedInside_ = out1_
                        d_6_openedCurrent_ = out2_
                        generated = d_4_openedGenerated_
                        insideConstrainedOut = d_5_openedInside_
                        currentConstrainedOut = d_6_openedCurrent_
                        d_3_forcedSpanDone_ = True
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif not(insideConstrainedOut):
                        d_7_remaining_: int
                        d_7_remaining_ = (maxSteps) - (d_1_steps_)
                        d_8_chunkBudget_: int
                        if (d_7_remaining_) < (d_2_chunkLimit_):
                            d_8_chunkBudget_ = d_7_remaining_
                        elif True:
                            d_8_chunkBudget_ = d_2_chunkLimit_
                        d_9_chunkedGenerated_: _dafny.Seq
                        d_10_stoppedOnOpenSpan_: bool
                        d_11_stoppedOnEos_: bool
                        d_12_stepsUsed_: int
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: bool
                        out6_: int
                        out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_9_chunkedGenerated_ = out3_
                        d_10_stoppedOnOpenSpan_ = out4_
                        d_11_stoppedOnEos_ = out5_
                        d_12_stepsUsed_ = out6_
                        generated = d_9_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                        if d_10_stoppedOnOpenSpan_:
                            d_13_enteredGenerated_: _dafny.Seq
                            d_14_enteredInside_: bool
                            d_15_enteredCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_13_enteredGenerated_ = out7_
                            d_14_enteredInside_ = out8_
                            d_15_enteredCurrent_ = out9_
                            generated = d_13_enteredGenerated_
                            insideConstrainedOut = d_14_enteredInside_
                            currentConstrainedOut = d_15_enteredCurrent_
                            d_3_forcedSpanDone_ = True
                        elif d_11_stoppedOnEos_:
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_closedGenerated_: _dafny.Seq
                        d_17_closedInside_: bool
                        d_18_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_closedGenerated_ = out10_
                        d_17_closedInside_ = out11_
                        d_18_closedCurrent_ = out12_
                        generated = d_16_closedGenerated_
                        insideConstrainedOut = d_17_closedInside_
                        currentConstrainedOut = d_18_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_20_candidates_: _dafny.Seq
                        out13_: _dafny.Seq
                        out13_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                        d_20_candidates_ = out13_
                        d_21_next_: _dafny.Seq
                        d_21_next_ = (d_20_candidates_)[0]
                        if ((d_21_next_) == (eosToken)) and ((len(d_20_candidates_)) > (1)):
                            d_21_next_ = (d_20_candidates_)[1]
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_21_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_22_appendedGenerated_: _dafny.Seq
                            d_23_appendedInside_: bool
                            d_24_appendedCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                            d_22_appendedGenerated_ = out14_
                            d_23_appendedInside_ = out15_
                            d_24_appendedCurrent_ = out16_
                            generated = d_22_appendedGenerated_
                            insideConstrainedOut = d_23_appendedInside_
                            currentConstrainedOut = d_24_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


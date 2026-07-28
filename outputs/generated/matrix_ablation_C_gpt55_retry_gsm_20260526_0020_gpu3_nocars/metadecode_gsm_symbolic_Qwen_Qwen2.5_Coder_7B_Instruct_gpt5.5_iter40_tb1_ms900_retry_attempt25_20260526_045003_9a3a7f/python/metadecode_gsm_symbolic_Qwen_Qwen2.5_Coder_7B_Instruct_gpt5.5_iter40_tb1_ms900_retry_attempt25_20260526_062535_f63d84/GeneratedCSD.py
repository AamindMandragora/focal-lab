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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the GSM word problem accurately. Use visible calculator annotations of the form <<arithmetic expression=result>>. If no annotation has begun yet, start with one short valid annotation, then continue the solution and finish with a final line exactly like #### answer.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_initialOpenCount_: int
        out0_: int
        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generatedPrefix, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        d_2_initialOpenCount_ = out0_
        d_3_seenSpan_: bool
        d_3_seenSpan_ = (insideConstrained) or ((d_2_initialOpenCount_) > (0))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_3_seenSpan_):
                            d_4_openedGenerated_: _dafny.Seq
                            d_5_openedInside_: bool
                            d_6_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openedGenerated_ = out1_
                            d_5_openedInside_ = out2_
                            d_6_openedCurrent_ = out3_
                            generated = d_4_openedGenerated_
                            insideConstrainedOut = d_5_openedInside_
                            currentConstrainedOut = d_6_openedCurrent_
                            d_3_seenSpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_remaining_: int
                            d_7_remaining_ = (maxSteps) - (d_1_steps_)
                            d_8_chunkedGenerated_: _dafny.Seq
                            d_9_stoppedOpen_: bool
                            d_10_stoppedEos_: bool
                            d_11_stepsUsed_: int
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: bool
                            out7_: int
                            out4_, out5_, out6_, out7_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_remaining_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_8_chunkedGenerated_ = out4_
                            d_9_stoppedOpen_ = out5_
                            d_10_stoppedEos_ = out6_
                            d_11_stepsUsed_ = out7_
                            generated = d_8_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_11_stepsUsed_)
                            if d_10_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_9_stoppedOpen_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_3_seenSpan_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
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
                        d_3_seenSpan_ = True
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_16_candidates_: _dafny.Seq
                        out11_: _dafny.Seq
                        out11_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                        d_16_candidates_ = out11_
                        d_17_next_: _dafny.Seq
                        d_17_next_ = (d_16_candidates_)[0]
                        if ((d_17_next_) == (eosToken)) and ((len(d_16_candidates_)) > (1)):
                            d_17_next_ = (d_16_candidates_)[1]
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_17_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_18_appendedGenerated_: _dafny.Seq
                            d_19_appendedInside_: bool
                            d_20_appendedCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                            d_18_appendedGenerated_ = out12_
                            d_19_appendedInside_ = out13_
                            d_20_appendedCurrent_ = out14_
                            generated = d_18_appendedGenerated_
                            insideConstrainedOut = d_19_appendedInside_
                            currentConstrainedOut = d_20_appendedCurrent_
                            d_3_seenSpan_ = True
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


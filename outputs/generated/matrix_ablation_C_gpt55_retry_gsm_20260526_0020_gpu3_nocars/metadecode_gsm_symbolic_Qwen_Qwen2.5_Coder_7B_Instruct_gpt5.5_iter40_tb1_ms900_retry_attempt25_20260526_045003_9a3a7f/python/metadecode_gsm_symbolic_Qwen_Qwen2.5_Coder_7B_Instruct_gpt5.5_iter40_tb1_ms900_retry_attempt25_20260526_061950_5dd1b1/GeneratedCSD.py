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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the word problem accurately. Use at least one visible calculator annotation of the exact form <<arithmetic expression=result>> for arithmetic, and finish with a final line exactly like #### answer.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openCount_: int
        out0_: int
        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generatedPrefix, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        d_2_openCount_ = out0_
        d_3_seenSpan_: bool
        d_3_seenSpan_ = (insideConstrained) or ((d_2_openCount_) > (0))
        d_4_boundaryArmed_: bool
        d_4_boundaryArmed_ = False
        d_5_forceAfter_: int
        d_5_forceAfter_ = 24
        d_6_boundaryAfter_: int
        d_6_boundaryAfter_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_3_seenSpan_)) and ((d_4_boundaryArmed_) or ((d_1_steps_) >= (d_5_forceAfter_))):
                            d_7_openedGenerated_: _dafny.Seq
                            d_8_openedInside_: bool
                            d_9_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_openedGenerated_ = out1_
                            d_8_openedInside_ = out2_
                            d_9_openedCurrent_ = out3_
                            generated = d_7_openedGenerated_
                            insideConstrainedOut = d_8_openedInside_
                            currentConstrainedOut = d_9_openedCurrent_
                            d_3_seenSpan_ = True
                            d_4_boundaryArmed_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif not(d_3_seenSpan_):
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
                                    d_3_seenSpan_ = True
                                    d_4_boundaryArmed_ = False
                                elif ((d_1_steps_) >= (d_6_boundaryAfter_)) and (((((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))))):
                                    d_4_boundaryArmed_ = True
                        elif True:
                            d_11_remaining_: int
                            d_11_remaining_ = (maxSteps) - (d_1_steps_)
                            d_12_chunkedGenerated_: _dafny.Seq
                            d_13_stoppedOpen_: bool
                            d_14_stoppedEos_: bool
                            d_15_stepsUsed_: int
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: bool
                            out8_: int
                            out5_, out6_, out7_, out8_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_11_remaining_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_12_chunkedGenerated_ = out5_
                            d_13_stoppedOpen_ = out6_
                            d_14_stoppedEos_ = out7_
                            d_15_stepsUsed_ = out8_
                            generated = d_12_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_15_stepsUsed_)
                            if d_14_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_13_stoppedOpen_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_3_seenSpan_ = True
                                d_4_boundaryArmed_ = False
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_closedGenerated_: _dafny.Seq
                        d_17_closedInside_: bool
                        d_18_closedCurrent_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: bool
                        out11_: _dafny.Seq
                        out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_closedGenerated_ = out9_
                        d_17_closedInside_ = out10_
                        d_18_closedCurrent_ = out11_
                        generated = d_16_closedGenerated_
                        insideConstrainedOut = d_17_closedInside_
                        currentConstrainedOut = d_18_closedCurrent_
                        d_3_seenSpan_ = True
                        d_4_boundaryArmed_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_20_candidates_: _dafny.Seq
                        out12_: _dafny.Seq
                        out12_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                        d_20_candidates_ = out12_
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
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                            d_22_appendedGenerated_ = out13_
                            d_23_appendedInside_ = out14_
                            d_24_appendedCurrent_ = out15_
                            generated = d_22_appendedGenerated_
                            insideConstrainedOut = d_23_appendedInside_
                            currentConstrainedOut = d_24_appendedCurrent_
                            d_3_seenSpan_ = True
                            d_4_boundaryArmed_ = False
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


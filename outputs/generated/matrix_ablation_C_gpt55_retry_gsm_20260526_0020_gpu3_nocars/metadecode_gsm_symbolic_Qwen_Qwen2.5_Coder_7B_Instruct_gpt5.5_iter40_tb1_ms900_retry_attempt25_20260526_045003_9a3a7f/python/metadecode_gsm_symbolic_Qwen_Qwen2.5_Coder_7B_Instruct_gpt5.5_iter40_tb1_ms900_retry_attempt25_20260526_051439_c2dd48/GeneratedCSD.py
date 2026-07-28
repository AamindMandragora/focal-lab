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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Use visible calculator spans exactly in the form <<expression=result>> for arithmetic, and finish with a final line exactly like #### answer.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openCount_: int
        out0_: int
        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        d_2_openCount_ = out0_
        d_3_closeCount_: int
        out1_: int
        out1_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
        d_3_closeCount_ = out1_
        d_4_haveOpenedSpan_: bool
        d_4_haveOpenedSpan_ = (insideConstrainedOut) or ((d_2_openCount_) > (0))
        d_5_haveClosedSpan_: bool
        d_5_haveClosedSpan_ = (d_3_closeCount_) > (0)
        d_6_forceSpanNow_: bool
        d_6_forceSpanNow_ = False
        d_7_naturalPrefixLimit_: int
        d_7_naturalPrefixLimit_ = 48
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_4_haveOpenedSpan_)) and ((d_6_forceSpanNow_) or ((d_1_steps_) >= (d_7_naturalPrefixLimit_))):
                            d_8_openedGenerated_: _dafny.Seq
                            d_9_openedInside_: bool
                            d_10_openedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_openedGenerated_ = out2_
                            d_9_openedInside_ = out3_
                            d_10_openedCurrent_ = out4_
                            generated = d_8_openedGenerated_
                            insideConstrainedOut = d_9_openedInside_
                            currentConstrainedOut = d_10_openedCurrent_
                            d_4_haveOpenedSpan_ = True
                            d_6_forceSpanNow_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif d_5_haveClosedSpan_:
                            d_11_remaining_: int
                            d_11_remaining_ = (maxSteps) - (d_1_steps_)
                            d_12_chunkedGenerated_: _dafny.Seq
                            d_13_stoppedOnOpenSpan_: bool
                            d_14_stoppedOnEos_: bool
                            d_15_stepsUsed_: int
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: bool
                            out8_: int
                            out5_, out6_, out7_, out8_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_11_remaining_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_12_chunkedGenerated_ = out5_
                            d_13_stoppedOnOpenSpan_ = out6_
                            d_14_stoppedOnEos_ = out7_
                            d_15_stepsUsed_ = out8_
                            generated = d_12_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_15_stepsUsed_)
                            if d_13_stoppedOnOpenSpan_:
                                d_16_enteredGenerated_: _dafny.Seq
                                d_17_enteredInside_: bool
                                d_18_enteredCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_16_enteredGenerated_ = out9_
                                d_17_enteredInside_ = out10_
                                d_18_enteredCurrent_ = out11_
                                generated = d_16_enteredGenerated_
                                insideConstrainedOut = d_17_enteredInside_
                                currentConstrainedOut = d_18_enteredCurrent_
                                d_4_haveOpenedSpan_ = True
                            elif d_14_stoppedOnEos_:
                                raise _dafny.Break("0")
                        elif True:
                            d_19_next_: _dafny.Seq
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_19_next_ = out12_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_19_next_) == (eosToken):
                                d_6_forceSpanNow_ = True
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_19_next_]))
                                if (d_19_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_4_haveOpenedSpan_ = True
                                    d_6_forceSpanNow_ = False
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_20_closedGenerated_: _dafny.Seq
                        d_21_closedInside_: bool
                        d_22_closedCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_20_closedGenerated_ = out13_
                        d_21_closedInside_ = out14_
                        d_22_closedCurrent_ = out15_
                        generated = d_20_closedGenerated_
                        insideConstrainedOut = d_21_closedInside_
                        currentConstrainedOut = d_22_closedCurrent_
                        d_5_haveClosedSpan_ = True
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_23_constrainedPrompt_: _dafny.Seq
                        d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_24_candidates_: _dafny.Seq
                        out16_: _dafny.Seq
                        out16_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, 8, eosToken)
                        d_24_candidates_ = out16_
                        d_25_next_: _dafny.Seq
                        d_25_next_ = (d_24_candidates_)[0]
                        if ((d_25_next_) == (eosToken)) and ((len(d_24_candidates_)) > (1)):
                            d_25_next_ = (d_24_candidates_)[1]
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_25_next_) == (eosToken):
                            d_6_forceSpanNow_ = False
                            raise _dafny.Break("0")
                        elif True:
                            d_26_appendedGenerated_: _dafny.Seq
                            d_27_appendedInside_: bool
                            d_28_appendedCurrent_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: _dafny.Seq
                            out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                            d_26_appendedGenerated_ = out17_
                            d_27_appendedInside_ = out18_
                            d_28_appendedCurrent_ = out19_
                            generated = d_26_appendedGenerated_
                            insideConstrainedOut = d_27_appendedInside_
                            currentConstrainedOut = d_28_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


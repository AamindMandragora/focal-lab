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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Produce exactly one valid SMILES molecule as the answer. Use a single constrained span only, keep generating inside that span until the molecule is complete, then close it immediately and stop. Avoid malformed or repeated degenerate continuations.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_repetitionStart_: int
        d_3_repetitionStart_ = 8
        d_4_openedHere_: bool
        d_4_openedHere_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_4_openedHere_:
                            d_5_nextOutside_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_5_nextOutside_ = out0_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_5_nextOutside_) != (eosToken):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_nextOutside_]))
                            raise _dafny.Break("0")
                        elif True:
                            d_6_openedGenerated_: _dafny.Seq
                            d_7_openedInside_: bool
                            d_8_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openedGenerated_ = out1_
                            d_7_openedInside_ = out2_
                            d_8_openedCurrent_ = out3_
                            generated = d_6_openedGenerated_
                            insideConstrainedOut = d_7_openedInside_
                            currentConstrainedOut = d_8_openedCurrent_
                            d_4_openedHere_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_9_isComplete_: bool
                        d_9_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_9_isComplete_:
                            d_10_closedGenerated_: _dafny.Seq
                            d_11_closedInside_: bool
                            d_12_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_closedGenerated_ = out4_
                            d_11_closedInside_ = out5_
                            d_12_closedCurrent_ = out6_
                            generated = d_10_closedGenerated_
                            insideConstrainedOut = d_11_closedInside_
                            currentConstrainedOut = d_12_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_14_next_: _dafny.Seq
                            d_14_next_ = eosToken
                            d_15_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_15_validCount_ = out7_
                            if (len(currentConstrainedOut)) >= (d_3_repetitionStart_):
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_14_next_ = out8_
                            elif (d_15_validCount_) <= (3):
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_14_next_ = out9_
                            elif (d_15_validCount_) <= (d_2_narrowThreshold_):
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                                d_14_next_ = out10_
                            elif True:
                                d_16_nextSoft_: _dafny.Seq
                                d_17_usedFallback_: bool
                                out11_: _dafny.Seq
                                out12_: bool
                                out11_, out12_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                                d_16_nextSoft_ = out11_
                                d_17_usedFallback_ = out12_
                                d_14_next_ = d_16_nextSoft_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_14_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_18_appendedGenerated_: _dafny.Seq
                                d_19_appendedInside_: bool
                                d_20_appendedCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                d_18_appendedGenerated_ = out13_
                                d_19_appendedInside_ = out14_
                                d_20_appendedCurrent_ = out15_
                                generated = d_18_appendedGenerated_
                                insideConstrainedOut = d_19_appendedInside_
                                currentConstrainedOut = d_20_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


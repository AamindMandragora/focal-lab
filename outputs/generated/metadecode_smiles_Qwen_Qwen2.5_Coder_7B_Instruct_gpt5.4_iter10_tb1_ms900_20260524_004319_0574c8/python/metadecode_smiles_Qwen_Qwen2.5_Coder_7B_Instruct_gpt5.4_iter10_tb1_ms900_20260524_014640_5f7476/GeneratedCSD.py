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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one valid SMILES string inside a constrained span. Start the span promptly, stay within SMILES syntax, avoid repetitive degenerate continuations, and close the span as soon as the molecule is complete.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_repetitionStart_: int
        d_3_repetitionStart_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
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
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_7_closedGenerated_: _dafny.Seq
                        d_8_closedInside_: bool
                        d_9_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_7_closedGenerated_ = out3_
                        d_8_closedInside_ = out4_
                        d_9_closedCurrent_ = out5_
                        generated = d_7_closedGenerated_
                        insideConstrainedOut = d_8_closedInside_
                        currentConstrainedOut = d_9_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_11_next_: _dafny.Seq
                        d_11_next_ = eosToken
                        d_12_validCount_: int
                        out6_: int
                        out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_12_validCount_ = out6_
                        if (len(currentConstrainedOut)) >= (d_3_repetitionStart_):
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_11_next_ = out7_
                        elif (d_12_validCount_) <= (3):
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_11_next_ = out8_
                        elif (d_12_validCount_) <= (d_2_narrowThreshold_):
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                            d_11_next_ = out9_
                        elif True:
                            d_13_nextSoft_: _dafny.Seq
                            d_14_usedFallback_: bool
                            out10_: _dafny.Seq
                            out11_: bool
                            out10_, out11_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                            d_13_nextSoft_ = out10_
                            d_14_usedFallback_ = out11_
                            d_11_next_ = d_13_nextSoft_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_11_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_15_appendedGenerated_: _dafny.Seq
                            d_16_appendedInside_: bool
                            d_17_appendedCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                            d_15_appendedGenerated_ = out12_
                            d_16_appendedInside_ = out13_
                            d_17_appendedCurrent_ = out14_
                            generated = d_15_appendedGenerated_
                            insideConstrainedOut = d_16_appendedInside_
                            currentConstrainedOut = d_17_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


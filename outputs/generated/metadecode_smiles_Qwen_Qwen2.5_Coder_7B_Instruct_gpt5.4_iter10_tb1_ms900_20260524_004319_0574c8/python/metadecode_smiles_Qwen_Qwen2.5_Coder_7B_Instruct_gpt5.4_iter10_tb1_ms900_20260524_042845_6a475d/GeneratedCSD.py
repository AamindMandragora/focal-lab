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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one valid SMILES string inside a constrained span. Start the span promptly, keep the earliest molecule tokens strictly syntactic, continue with a valid non-degenerate SMILES, and close the span immediately when the molecule is complete.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_repetitionStart_: int
        d_3_repetitionStart_ = 8
        d_4_hardPrefixLen_: int
        d_4_hardPrefixLen_ = 3
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_openedGenerated_: _dafny.Seq
                        d_6_openedInside_: bool
                        d_7_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_5_openedGenerated_ = out0_
                        d_6_openedInside_ = out1_
                        d_7_openedCurrent_ = out2_
                        generated = d_5_openedGenerated_
                        insideConstrainedOut = d_6_openedInside_
                        currentConstrainedOut = d_7_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_8_closedGenerated_: _dafny.Seq
                        d_9_closedInside_: bool
                        d_10_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_8_closedGenerated_ = out3_
                        d_9_closedInside_ = out4_
                        d_10_closedCurrent_ = out5_
                        generated = d_8_closedGenerated_
                        insideConstrainedOut = d_9_closedInside_
                        currentConstrainedOut = d_10_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_12_next_: _dafny.Seq
                        d_12_next_ = eosToken
                        d_13_validCount_: int
                        out6_: int
                        out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_13_validCount_ = out6_
                        if (len(currentConstrainedOut)) < (d_4_hardPrefixLen_):
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_12_next_ = out7_
                        elif (len(currentConstrainedOut)) >= (d_3_repetitionStart_):
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_12_next_ = out8_
                        elif (d_13_validCount_) <= (3):
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_12_next_ = out9_
                        elif (d_13_validCount_) <= (d_2_narrowThreshold_):
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                            d_12_next_ = out10_
                        elif True:
                            d_14_nextSoft_: _dafny.Seq
                            d_15_usedFallback_: bool
                            out11_: _dafny.Seq
                            out12_: bool
                            out11_, out12_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                            d_14_nextSoft_ = out11_
                            d_15_usedFallback_ = out12_
                            d_12_next_ = d_14_nextSoft_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_12_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_16_appendedGenerated_: _dafny.Seq
                            d_17_appendedInside_: bool
                            d_18_appendedCurrent_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                            d_16_appendedGenerated_ = out13_
                            d_17_appendedInside_ = out14_
                            d_18_appendedCurrent_ = out15_
                            generated = d_16_appendedGenerated_
                            insideConstrainedOut = d_17_appendedInside_
                            currentConstrainedOut = d_18_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


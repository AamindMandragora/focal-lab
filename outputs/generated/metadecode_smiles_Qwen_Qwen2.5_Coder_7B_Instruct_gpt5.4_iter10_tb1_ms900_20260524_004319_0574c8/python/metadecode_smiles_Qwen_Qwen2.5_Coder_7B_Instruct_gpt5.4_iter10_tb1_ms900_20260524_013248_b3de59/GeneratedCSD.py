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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "When producing a constrained span, emit a valid SMILES string for the requested molecular class, avoid repetitive degenerate continuations, and close the span as soon as the SMILES is complete.")))
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
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_5_closedGenerated_: _dafny.Seq
                        d_6_closedInside_: bool
                        d_7_closedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_5_closedGenerated_ = out1_
                        d_6_closedInside_ = out2_
                        d_7_closedCurrent_ = out3_
                        generated = d_5_closedGenerated_
                        insideConstrainedOut = d_6_closedInside_
                        currentConstrainedOut = d_7_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_8_constrainedPrompt_: _dafny.Seq
                        d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_9_next_: _dafny.Seq
                        d_9_next_ = eosToken
                        d_10_validCount_: int
                        out4_: int
                        out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_10_validCount_ = out4_
                        if (len(currentConstrainedOut)) >= (d_3_repetitionStart_):
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_9_next_ = out5_
                        elif (d_10_validCount_) <= (d_2_narrowThreshold_):
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                            d_9_next_ = out6_
                        elif True:
                            d_11_nextSoft_: _dafny.Seq
                            d_12_usedFallback_: bool
                            out7_: _dafny.Seq
                            out8_: bool
                            out7_, out8_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                            d_11_nextSoft_ = out7_
                            d_12_usedFallback_ = out8_
                            d_9_next_ = d_11_nextSoft_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_9_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_13_appendedGenerated_: _dafny.Seq
                            d_14_appendedInside_: bool
                            d_15_appendedCurrent_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                            d_13_appendedGenerated_ = out9_
                            d_14_appendedInside_ = out10_
                            d_15_appendedCurrent_ = out11_
                            generated = d_13_appendedGenerated_
                            insideConstrainedOut = d_14_appendedInside_
                            currentConstrainedOut = d_15_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


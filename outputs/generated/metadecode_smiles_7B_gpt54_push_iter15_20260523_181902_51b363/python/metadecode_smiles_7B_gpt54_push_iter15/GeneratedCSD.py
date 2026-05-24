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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "When producing the constrained molecule span, generate a valid SMILES string matching the requested molecular class. Prefer chemically plausible continuations and keep the constrained prefix parser-valid at every step.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_4_closedGenerated_: _dafny.Seq
                        d_5_closedInside_: bool
                        d_6_closedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_4_closedGenerated_ = out1_
                        d_5_closedInside_ = out2_
                        d_6_closedCurrent_ = out3_
                        generated = d_4_closedGenerated_
                        insideConstrainedOut = d_5_closedInside_
                        currentConstrainedOut = d_6_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_7_stablePrefix_: _dafny.Seq
                        d_7_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_8_constrainedPrompt_: _dafny.Seq
                        d_8_constrainedPrompt_ = (prompt) + (d_7_stablePrefix_)
                        d_9_validCount_: int
                        out4_: int
                        out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_9_validCount_ = out4_
                        d_10_openParenCount_: int
                        out5_: int
                        out5_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")))
                        d_10_openParenCount_ = out5_
                        d_11_bondCount_: int
                        out6_: int
                        out6_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                        d_11_bondCount_ = out6_
                        d_12_next_: _dafny.Seq
                        d_12_next_ = eosToken
                        if (d_9_validCount_) <= (d_2_narrowThreshold_):
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                            d_12_next_ = out7_
                        elif ((d_10_openParenCount_) > (1)) or ((d_11_bondCount_) > (1)):
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_12_next_ = out8_
                        elif True:
                            d_13_nextSoft_: _dafny.Seq
                            d_14_usedFallback_: bool
                            out9_: _dafny.Seq
                            out10_: bool
                            out9_, out10_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                            d_13_nextSoft_ = out9_
                            d_14_usedFallback_ = out10_
                            d_12_next_ = d_13_nextSoft_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_12_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_15_appendedGenerated_: _dafny.Seq
                            d_16_appendedInside_: bool
                            d_17_appendedCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                            d_15_appendedGenerated_ = out11_
                            d_16_appendedInside_ = out12_
                            d_17_appendedCurrent_ = out13_
                            generated = d_15_appendedGenerated_
                            insideConstrainedOut = d_16_appendedInside_
                            currentConstrainedOut = d_17_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


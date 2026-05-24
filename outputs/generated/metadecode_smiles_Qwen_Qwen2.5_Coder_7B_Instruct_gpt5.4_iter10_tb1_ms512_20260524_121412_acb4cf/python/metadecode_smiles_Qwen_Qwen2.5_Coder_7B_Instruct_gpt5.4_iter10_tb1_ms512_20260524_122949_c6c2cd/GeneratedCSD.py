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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output a valid SMILES string in one constrained span. Start the constrained span immediately, keep every token parser-valid, and close the span as soon as the SMILES is complete. Prefer class-matching, chemically plausible molecules and avoid repetitive continuations.")))
        (d_0_helpers_).SetNonDeterministic(lm, True)
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_openedGenerated_: _dafny.Seq
                        d_3_openedInside_: bool
                        d_4_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_2_openedGenerated_ = out0_
                        d_3_openedInside_ = out1_
                        d_4_openedCurrent_ = out2_
                        generated = d_2_openedGenerated_
                        insideConstrainedOut = d_3_openedInside_
                        currentConstrainedOut = d_4_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_5_closedGenerated_: _dafny.Seq
                        d_6_closedInside_: bool
                        d_7_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_5_closedGenerated_ = out3_
                        d_6_closedInside_ = out4_
                        d_7_closedCurrent_ = out5_
                        generated = d_5_closedGenerated_
                        insideConstrainedOut = d_6_closedInside_
                        currentConstrainedOut = d_7_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_8_stablePrefix_: _dafny.Seq
                        d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                        d_10_nextIn_: _dafny.Seq
                        d_10_nextIn_ = eosToken
                        if (len(currentConstrainedOut)) == (0):
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_10_nextIn_ = out6_
                        elif (len(currentConstrainedOut)) < (3):
                            d_11_nextSoft_: _dafny.Seq
                            d_12_usedFallback_: bool
                            out7_: _dafny.Seq
                            out8_: bool
                            out7_, out8_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                            d_11_nextSoft_ = out7_
                            d_12_usedFallback_ = out8_
                            d_10_nextIn_ = d_11_nextSoft_
                        elif True:
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_10_nextIn_ = out9_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_10_nextIn_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_13_appendedGenerated_: _dafny.Seq
                            d_14_appendedInside_: bool
                            d_15_appendedCurrent_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_nextIn_)
                            d_13_appendedGenerated_ = out10_
                            d_14_appendedInside_ = out11_
                            d_15_appendedCurrent_ = out12_
                            generated = d_13_appendedGenerated_
                            insideConstrainedOut = d_14_appendedInside_
                            currentConstrainedOut = d_15_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


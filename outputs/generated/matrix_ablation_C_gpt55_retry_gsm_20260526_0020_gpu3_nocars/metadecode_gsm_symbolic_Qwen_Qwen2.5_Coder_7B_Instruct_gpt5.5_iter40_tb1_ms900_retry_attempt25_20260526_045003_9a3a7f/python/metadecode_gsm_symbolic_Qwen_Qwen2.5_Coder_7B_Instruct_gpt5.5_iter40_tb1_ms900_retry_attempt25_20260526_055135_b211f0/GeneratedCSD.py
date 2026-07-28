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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem carefully. Show concise reasoning. Use visible calculator spans exactly like <<expression=result>> for arithmetic computations when helpful. Finish with a final line exactly of the form #### answer.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_markerEmitted_: bool
        d_2_markerEmitted_ = False
        d_3_marker_: _dafny.Seq
        d_3_marker_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_1_steps_)
                        if (not(d_2_markerEmitted_)) and ((5) <= (d_4_remaining_)):
                            d_5_before_: _dafny.Seq
                            d_5_before_ = generated
                            d_6_oldSteps_: int
                            d_6_oldSteps_ = d_1_steps_
                            generated = (d_5_before_) + (d_3_marker_)
                            d_2_markerEmitted_ = True
                            d_1_steps_ = (d_6_oldSteps_) + (5)
                        elif True:
                            d_7_before_: _dafny.Seq
                            d_7_before_ = generated
                            d_8_oldSteps_: int
                            d_8_oldSteps_ = d_1_steps_
                            d_9_budget_: int
                            d_9_budget_ = d_4_remaining_
                            d_10_minReasoning_: int
                            if (d_9_budget_) < (8):
                                d_10_minReasoning_ = d_9_budget_
                            elif True:
                                d_10_minReasoning_ = 8
                            d_11_continuation_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).CraneGeneration(lm, parser, (prompt) + (d_7_before_), d_9_budget_, d_10_minReasoning_, eosToken)
                            d_11_continuation_ = out0_
                            generated = (d_7_before_) + (d_11_continuation_)
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_8_oldSteps_) + (d_9_budget_)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_closedGenerated_: _dafny.Seq
                        d_13_closedInside_: bool
                        d_14_closedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_closedGenerated_ = out1_
                        d_13_closedInside_ = out2_
                        d_14_closedCurrent_ = out3_
                        generated = d_12_closedGenerated_
                        insideConstrainedOut = d_13_closedInside_
                        currentConstrainedOut = d_14_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_16_next_: _dafny.Seq
                        d_17_wasConstrained_: bool
                        out4_: _dafny.Seq
                        out5_: bool
                        out4_, out5_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_16_next_ = out4_
                        d_17_wasConstrained_ = out5_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_16_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_18_appendedGenerated_: _dafny.Seq
                            d_19_appendedInside_: bool
                            d_20_appendedCurrent_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                            d_18_appendedGenerated_ = out6_
                            d_19_appendedInside_ = out7_
                            d_20_appendedCurrent_ = out8_
                            generated = d_18_appendedGenerated_
                            insideConstrainedOut = d_19_appendedInside_
                            currentConstrainedOut = d_20_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query in the required visible constrained span format. Put the full query inside one << >> span and do not add explanation before or after it.")))
        (d_0_helpers_).SetNonDeterministic(lm, False)
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_lateOpenThreshold_: int
        d_2_lateOpenThreshold_ = 4
        d_3_forcedOpenUsed_: bool
        d_3_forcedOpenUsed_ = insideConstrainedOut
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_openCount_: int
                        d_4_openCount_ = VerifiedDecoderAgent.CSDHelpers.OccurrencesInRange(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), len(generated))
                        if ((not(d_3_forcedOpenUsed_)) and ((d_4_openCount_) == (0))) and (((maxSteps) - (d_1_steps_)) <= (d_2_lateOpenThreshold_)):
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
                            d_3_forcedOpenUsed_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_9_enteredGenerated_: _dafny.Seq
                                    d_10_enteredInside_: bool
                                    d_11_enteredCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_9_enteredGenerated_ = out4_
                                    d_10_enteredInside_ = out5_
                                    d_11_enteredCurrent_ = out6_
                                    generated = d_9_enteredGenerated_
                                    insideConstrainedOut = d_10_enteredInside_
                                    currentConstrainedOut = d_11_enteredCurrent_
                                    d_3_forcedOpenUsed_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_closedGenerated_: _dafny.Seq
                        d_13_closedInside_: bool
                        d_14_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_closedGenerated_ = out7_
                        d_13_closedInside_ = out8_
                        d_14_closedCurrent_ = out9_
                        generated = d_12_closedGenerated_
                        insideConstrainedOut = d_13_closedInside_
                        currentConstrainedOut = d_14_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_15_stablePrefix_: _dafny.Seq
                        d_15_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_16_constrainedPrompt_: _dafny.Seq
                        d_16_constrainedPrompt_ = (prompt) + (d_15_stablePrefix_)
                        d_17_nextInside_: _dafny.Seq
                        d_18_wasConstrained_: bool
                        out10_: _dafny.Seq
                        out11_: bool
                        out10_, out11_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_17_nextInside_ = out10_
                        d_18_wasConstrained_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_17_nextInside_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_19_appendedGenerated_: _dafny.Seq
                            d_20_appendedInside_: bool
                            d_21_appendedCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_nextInside_)
                            d_19_appendedGenerated_ = out12_
                            d_20_appendedInside_ = out13_
                            d_21_appendedCurrent_ = out14_
                            generated = d_19_appendedGenerated_
                            insideConstrainedOut = d_20_appendedInside_
                            currentConstrainedOut = d_21_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


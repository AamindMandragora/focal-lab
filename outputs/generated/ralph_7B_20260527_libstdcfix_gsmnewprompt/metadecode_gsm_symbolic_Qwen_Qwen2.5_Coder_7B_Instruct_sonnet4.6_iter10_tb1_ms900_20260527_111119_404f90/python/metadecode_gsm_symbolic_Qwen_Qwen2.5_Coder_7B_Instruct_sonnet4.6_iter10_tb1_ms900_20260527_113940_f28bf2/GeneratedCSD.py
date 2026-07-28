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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Wrap every intermediate arithmetic expression and the final answer inside << >> delimiters. Inside << >>, write plain arithmetic expressions using only variable names with NO curly braces (write n1 not {n1}). The very last << >> must contain only the final answer expression.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanLen_: int
        d_2_spanLen_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_spanLen_ = 0
                        d_3_chunkSize_: int
                        d_3_chunkSize_ = (maxSteps) - (d_1_steps_)
                        if (d_3_chunkSize_) > (40):
                            d_3_chunkSize_ = 40
                        d_4_generatedOut_: _dafny.Seq
                        d_5_stoppedOnOpenSpan_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_generatedOut_ = out0_
                        d_5_stoppedOnOpenSpan_ = out1_
                        d_6_stoppedOnEos_ = out2_
                        d_7_stepsUsed_ = out3_
                        generated = d_4_generatedOut_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        if d_5_stoppedOnOpenSpan_:
                            d_8_g2_: _dafny.Seq
                            d_9_i2_: bool
                            d_10_c2_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_8_g2_ = out4_
                            d_9_i2_ = out5_
                            d_10_c2_ = out6_
                            generated = d_8_g2_
                            insideConstrainedOut = d_9_i2_
                            currentConstrainedOut = d_10_c2_
                        elif d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_11_closedGenerated_: _dafny.Seq
                        d_12_closedInside_: bool
                        d_13_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_11_closedGenerated_ = out7_
                        d_12_closedInside_ = out8_
                        d_13_closedCurrent_ = out9_
                        generated = d_11_closedGenerated_
                        insideConstrainedOut = d_12_closedInside_
                        currentConstrainedOut = d_13_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanLen_ = 0
                    elif True:
                        d_14_constrainedPrompt_: _dafny.Seq
                        d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_15_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (d_2_spanLen_) >= (20):
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                            d_15_next_ = out10_
                        elif True:
                            d_16_wasConstrained_: bool = False
                            out11_: _dafny.Seq
                            out12_: bool
                            out11_, out12_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_15_next_ = out11_
                            d_16_wasConstrained_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanLen_ = (d_2_spanLen_) + (1)
                        if (d_15_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_17_appendedGenerated_: _dafny.Seq
                            d_18_appendedInside_: bool
                            d_19_appendedCurrent_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                            d_17_appendedGenerated_ = out13_
                            d_18_appendedInside_ = out14_
                            d_19_appendedCurrent_ = out15_
                            generated = d_17_appendedGenerated_
                            insideConstrainedOut = d_18_appendedInside_
                            currentConstrainedOut = d_19_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


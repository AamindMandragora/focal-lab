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
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out3_: int
        out0_, out1_, out2_, out3_ = default__.AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken)
        generated = out0_
        insideConstrainedOut = out1_
        currentConstrainedOut = out2_
        cost = out3_
        if ((maxSteps) > (0)) and ((cost) <= (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

    @staticmethod
    def AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. Write variable names WITHOUT curly braces (write n not {n}, write price not {price}). For EVERY calculation step and the final answer, you MUST use << >> delimiters. Inside << >> use ONLY: numbers, variable names (no braces), +, -, *, /, //, %, (), int(). Example: <<n * price>> or <<int((length + space) / (plant_width + space))>>. The LAST << >> contains the final answer expression only.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_chunkSize_: int
        d_2_chunkSize_ = 48
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remainingSteps_: int
                        d_3_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        d_4_reserveForSpan_: int
                        d_4_reserveForSpan_ = 80
                        d_5_freeSteps_: int
                        if (d_3_remainingSteps_) > (d_4_reserveForSpan_):
                            d_5_freeSteps_ = (d_3_remainingSteps_) - (d_4_reserveForSpan_)
                        elif True:
                            d_5_freeSteps_ = d_3_remainingSteps_
                        d_6_chunkTokens_: int
                        if (d_5_freeSteps_) < (d_2_chunkSize_):
                            d_6_chunkTokens_ = d_5_freeSteps_
                        elif True:
                            d_6_chunkTokens_ = d_2_chunkSize_
                        if (d_6_chunkTokens_) == (0):
                            raise _dafny.Break("0")
                        d_7_chunkGenerated_: _dafny.Seq
                        d_8_stoppedOnOpenSpan_: bool
                        d_9_stoppedOnEos_: bool
                        d_10_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkTokens_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_7_chunkGenerated_ = out0_
                        d_8_stoppedOnOpenSpan_ = out1_
                        d_9_stoppedOnEos_ = out2_
                        d_10_stepsUsed_ = out3_
                        generated = d_7_chunkGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                        if d_9_stoppedOnEos_:
                            raise _dafny.Break("0")
                        if d_8_stoppedOnOpenSpan_:
                            d_11_eg_: _dafny.Seq
                            d_12_ei_: bool
                            d_13_ec_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_11_eg_ = out4_
                            d_12_ei_ = out5_
                            d_13_ec_ = out6_
                            generated = d_11_eg_
                            insideConstrainedOut = d_12_ei_
                            currentConstrainedOut = d_13_ec_
                    elif True:
                        d_14_remainingSteps_: int
                        d_14_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_14_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        d_15_closeBudget_: int
                        if (d_14_remainingSteps_) < (80):
                            d_15_closeBudget_ = d_14_remainingSteps_
                        elif True:
                            d_15_closeBudget_ = 80
                        d_16_cg_: _dafny.Seq
                        d_17_ci_: bool
                        d_18_cc_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudget_)
                        d_16_cg_ = out7_
                        d_17_ci_ = out8_
                        d_18_cc_ = out9_
                        generated = d_16_cg_
                        insideConstrainedOut = d_17_ci_
                        currentConstrainedOut = d_18_cc_
                        d_1_steps_ = (d_1_steps_) + (d_15_closeBudget_)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


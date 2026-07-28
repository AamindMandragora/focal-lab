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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Reason in plain text first. Before the final answer, explicitly identify the target quantity, write the symbolic relationship using only variables from the problem, check the operation order and units, then put only the final compact symbolic expression inside exactly one << >> span. Do not put intermediate work inside << >>.")))
        d_2_finalReserve_: int
        d_2_finalReserve_ = 64
        if (d_2_finalReserve_) > (maxSteps):
            d_2_finalReserve_ = _dafny.euclidian_division(maxSteps, 2)
        d_3_freeCap_: int
        d_3_freeCap_ = (maxSteps) - (d_2_finalReserve_)
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_3_freeCap_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_4_remaining_: int
                    d_4_remaining_ = (d_3_freeCap_) - (d_1_steps_)
                    d_5_chunkSize_: int
                    d_5_chunkSize_ = d_4_remaining_
                    if (d_5_chunkSize_) > (48):
                        d_5_chunkSize_ = 48
                    if (d_5_chunkSize_) == (0):
                        raise _dafny.Break("0")
                    d_6_genOut_: _dafny.Seq
                    d_7_stoppedOnOpen_: bool
                    d_8_stoppedOnEos_: bool
                    d_9_stepsUsed_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_6_genOut_ = out0_
                    d_7_stoppedOnOpen_ = out1_
                    d_8_stoppedOnEos_ = out2_
                    d_9_stepsUsed_ = out3_
                    generated = d_6_genOut_
                    d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                    if d_7_stoppedOnOpen_:
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        generated = out4_
                        insideConstrainedOut = out5_
                        currentConstrainedOut = out6_
                    elif d_8_stoppedOnEos_:
                        raise _dafny.Break("0")
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            generated = out7_
            insideConstrainedOut = out8_
            currentConstrainedOut = out9_
            d_1_steps_ = (d_1_steps_) + (1)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_10_closeBudget_: int
            d_10_closeBudget_ = (maxSteps) - (d_1_steps_)
            if (d_10_closeBudget_) > (d_2_finalReserve_):
                d_10_closeBudget_ = d_2_finalReserve_
            d_11_cg_: _dafny.Seq
            d_12_ci_: bool
            d_13_cc_: _dafny.Seq
            out10_: _dafny.Seq
            out11_: bool
            out12_: _dafny.Seq
            out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_10_closeBudget_)
            d_11_cg_ = out10_
            d_12_ci_ = out11_
            d_13_cc_ = out12_
            generated = d_11_cg_
            insideConstrainedOut = d_12_ci_
            currentConstrainedOut = d_13_cc_
            d_1_steps_ = (d_1_steps_) + (d_10_closeBudget_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
        d_2_effectiveMax_: int
        if (maxSteps) > (150):
            d_2_effectiveMax_ = 150
        elif True:
            d_2_effectiveMax_ = maxSteps
        with _dafny.label("0"):
            while (d_1_steps_) < (d_2_effectiveMax_):
                with _dafny.c_label("0"):
                    if (d_1_steps_) >= (d_2_effectiveMax_):
                        raise _dafny.Break("0")
                    if not(insideConstrainedOut):
                        d_3_remaining_: int
                        d_3_remaining_ = (d_2_effectiveMax_) - (d_1_steps_)
                        d_4_chunkBudget_: int = int(0)
                        if (d_3_remaining_) <= (22):
                            d_4_chunkBudget_ = 0
                        elif True:
                            d_5_availForChunk_: int
                            d_5_availForChunk_ = (d_3_remaining_) - (22)
                            if (d_5_availForChunk_) > (15):
                                d_4_chunkBudget_ = 15
                            elif True:
                                d_4_chunkBudget_ = d_5_availForChunk_
                        if (d_4_chunkBudget_) > (0):
                            d_6_chunkGenerated_: _dafny.Seq
                            d_7_stoppedOnOpenSpan_: bool
                            d_8_stoppedOnEos_: bool
                            d_9_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_6_chunkGenerated_ = out0_
                            d_7_stoppedOnOpenSpan_ = out1_
                            d_8_stoppedOnEos_ = out2_
                            d_9_stepsUsed_ = out3_
                            d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                            generated = d_6_chunkGenerated_
                            if d_7_stoppedOnOpenSpan_:
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                generated = out4_
                                insideConstrainedOut = out5_
                                currentConstrainedOut = out6_
                            elif d_8_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif True:
                                if (d_1_steps_) < (d_2_effectiveMax_):
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    generated = out7_
                                    insideConstrainedOut = out8_
                                    currentConstrainedOut = out9_
                                    d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if (d_1_steps_) < (d_2_effectiveMax_):
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                generated = out10_
                                insideConstrainedOut = out11_
                                currentConstrainedOut = out12_
                                d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_10_remaining_: int
                        d_10_remaining_ = (d_2_effectiveMax_) - (d_1_steps_)
                        if (d_10_remaining_) == (0):
                            raise _dafny.Break("0")
                        d_11_spanBudget_: int
                        if (d_10_remaining_) > (25):
                            d_11_spanBudget_ = 25
                        elif True:
                            d_11_spanBudget_ = d_10_remaining_
                        d_12_cg_: _dafny.Seq
                        d_13_ci_: bool
                        d_14_cc_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_spanBudget_)
                        d_12_cg_ = out13_
                        d_13_ci_ = out14_
                        d_14_cc_ = out15_
                        generated = d_12_cg_
                        insideConstrainedOut = d_13_ci_
                        currentConstrainedOut = d_14_cc_
                        d_1_steps_ = (d_1_steps_) + (d_11_spanBudget_)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


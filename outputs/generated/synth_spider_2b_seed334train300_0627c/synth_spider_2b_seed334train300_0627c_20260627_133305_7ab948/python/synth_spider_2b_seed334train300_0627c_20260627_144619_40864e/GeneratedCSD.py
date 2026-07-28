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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<SELECT ... FROM ...>> using only the provided schema table and column names. Do not add any text after >>.")))
        d_2_phase0Budget_: int
        if (maxSteps) < (40):
            d_2_phase0Budget_ = maxSteps
        elif True:
            d_2_phase0Budget_ = 40
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_3_chunkBudget_: int
            if ((maxSteps) - (d_1_steps_)) < (d_2_phase0Budget_):
                d_3_chunkBudget_ = (maxSteps) - (d_1_steps_)
            elif True:
                d_3_chunkBudget_ = d_2_phase0Budget_
            d_4_chunkGenerated_: _dafny.Seq
            d_5_stoppedOnOpenSpan_: bool
            d_6_stoppedOnEos_: bool
            d_7_chunkSteps_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_4_chunkGenerated_ = out0_
            d_5_stoppedOnOpenSpan_ = out1_
            d_6_stoppedOnEos_ = out2_
            d_7_chunkSteps_ = out3_
            d_1_steps_ = (d_1_steps_) + (d_7_chunkSteps_)
            generated = d_4_chunkGenerated_
            if d_5_stoppedOnOpenSpan_:
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                generated = out4_
                insideConstrainedOut = out5_
                currentConstrainedOut = out6_
            elif not(d_6_stoppedOnEos_):
                if (d_1_steps_) < (maxSteps):
                    d_8_og_: _dafny.Seq
                    d_9_oi_: bool
                    d_10_oc_: _dafny.Seq
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_8_og_ = out7_
                    d_9_oi_ = out8_
                    d_10_oc_ = out9_
                    generated = d_8_og_
                    insideConstrainedOut = d_9_oi_
                    currentConstrainedOut = d_10_oc_
                    d_1_steps_ = (d_1_steps_) + (1)
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_11_og_: _dafny.Seq
            d_12_oi_: bool
            d_13_oc_: _dafny.Seq
            out10_: _dafny.Seq
            out11_: bool
            out12_: _dafny.Seq
            out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_11_og_ = out10_
            d_12_oi_ = out11_
            d_13_oc_ = out12_
            generated = d_11_og_
            insideConstrainedOut = d_12_oi_
            currentConstrainedOut = d_13_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_14_closeBudget_: int
            d_14_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_15_cg2_: _dafny.Seq
            d_16_ci2_: bool
            d_17_cc2_: _dafny.Seq
            out13_: _dafny.Seq
            out14_: bool
            out15_: _dafny.Seq
            out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudget_)
            d_15_cg2_ = out13_
            d_16_ci2_ = out14_
            d_17_cc2_ = out15_
            generated = d_15_cg2_
            insideConstrainedOut = d_16_ci2_
            currentConstrainedOut = d_17_cc2_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


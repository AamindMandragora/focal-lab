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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<SELECT ...>> where the content is a single valid SQL SELECT statement. Use only table and column names from the schema. Be concise and direct. No explanation or markdown."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_3_chunkBudget_: int
            if (15) <= ((maxSteps) - (d_2_steps_)):
                d_3_chunkBudget_ = 15
            elif True:
                d_3_chunkBudget_ = (maxSteps) - (d_2_steps_)
            if (d_3_chunkBudget_) >= (1):
                d_4_cg_: _dafny.Seq
                d_5_stoppedOnOpenSpan_: bool
                d_6_stoppedOnEos_: bool
                d_7_stepsUsed_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_4_cg_ = out0_
                d_5_stoppedOnOpenSpan_ = out1_
                d_6_stoppedOnEos_ = out2_
                d_7_stepsUsed_ = out3_
                d_2_steps_ = (d_2_steps_) + (d_7_stepsUsed_)
                generated = d_4_cg_
                if d_6_stoppedOnEos_:
                    cost = d_2_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                if d_5_stoppedOnOpenSpan_:
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    generated = out4_
                    insideConstrainedOut = out5_
                    currentConstrainedOut = out6_
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            generated = out7_
            insideConstrainedOut = out8_
            currentConstrainedOut = out9_
            d_2_steps_ = (d_2_steps_) + (1)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_8_rem_: int
            d_8_rem_ = (maxSteps) - (d_2_steps_)
            d_9_closeReserve_: int
            if (d_8_rem_) >= (35):
                d_9_closeReserve_ = 35
            elif (d_8_rem_) >= (10):
                d_9_closeReserve_ = 10
            elif (d_8_rem_) >= (2):
                d_9_closeReserve_ = 2
            elif True:
                d_9_closeReserve_ = d_8_rem_
            d_10_fillBudget_: int
            if (d_8_rem_) > (d_9_closeReserve_):
                d_10_fillBudget_ = (d_8_rem_) - (d_9_closeReserve_)
            elif True:
                d_10_fillBudget_ = 0
            if (d_10_fillBudget_) >= (1):
                d_11_promptStr_: _dafny.Seq
                d_11_promptStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(prompt)
                d_12_flatGroups_: _dafny.Seq
                out10_: _dafny.Seq
                out10_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                d_12_flatGroups_ = out10_
                d_13_allowedUnits_: _dafny.Seq
                d_13_allowedUnits_ = _dafny.SeqWithoutIsStrInference([])
                d_14_pi_: int
                d_14_pi_ = 0
                while (d_14_pi_) < (len(prompt)):
                    d_13_allowedUnits_ = (d_13_allowedUnits_) + (_dafny.SeqWithoutIsStrInference([(prompt)[d_14_pi_]]))
                    d_14_pi_ = (d_14_pi_) + (1)
                d_15_fi_: int
                d_15_fi_ = 0
                while (d_15_fi_) < (len(d_12_flatGroups_)):
                    d_13_allowedUnits_ = (d_13_allowedUnits_) + (_dafny.SeqWithoutIsStrInference([(d_12_flatGroups_)[d_15_fi_]]))
                    d_15_fi_ = (d_15_fi_) + (1)
                d_16_stable_: _dafny.Seq
                d_16_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_17_constrainedPrompt_: _dafny.Seq
                d_17_constrainedPrompt_ = (prompt) + (d_16_stable_)
                d_18_maxStepsPerUnit_: int
                if (d_10_fillBudget_) >= (20):
                    d_18_maxStepsPerUnit_ = 20
                elif True:
                    d_18_maxStepsPerUnit_ = d_10_fillBudget_
                d_19_maxRetries_: int
                d_19_maxRetries_ = 3
                d_20_maxRollbackBudget_: int
                if (d_10_fillBudget_) >= (10):
                    d_20_maxRollbackBudget_ = 10
                elif True:
                    d_20_maxRollbackBudget_ = d_10_fillBudget_
                d_21_filled_: _dafny.Seq
                out11_: _dafny.Seq
                out11_ = (d_0_helpers_).RegenerateUnitOnCheckFailure(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, eosToken, d_18_maxStepsPerUnit_, d_19_maxRetries_, d_20_maxRollbackBudget_, d_13_allowedUnits_)
                d_21_filled_ = out11_
                generated = (d_16_stable_) + (d_21_filled_)
                currentConstrainedOut = d_21_filled_
                d_2_steps_ = (d_2_steps_) + (d_10_fillBudget_)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_22_closeBudget_: int
            d_22_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_23_cg_: _dafny.Seq
            d_24_ci_: bool
            d_25_cc_: _dafny.Seq
            out12_: _dafny.Seq
            out13_: bool
            out14_: _dafny.Seq
            out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_closeBudget_)
            d_23_cg_ = out12_
            d_24_ci_ = out13_
            d_25_cc_ = out14_
            generated = d_23_cg_
            insideConstrainedOut = d_24_ci_
            currentConstrainedOut = d_25_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


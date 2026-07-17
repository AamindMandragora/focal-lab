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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a SQL SELECT statement that answers EXACTLY the question asked. SELECT only the specific columns explicitly mentioned in the question - do not add extra columns like 'name' or 'id' unless the question specifically asks for them. Use only exact table and column names from the schema. No aliases. Output: SQL: <<query>>")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_2_chunkBudget_: int
            if ((maxSteps) - (d_1_steps_)) <= (8):
                d_2_chunkBudget_ = (maxSteps) - (d_1_steps_)
            elif True:
                d_2_chunkBudget_ = 8
            d_3_generatedOut_: _dafny.Seq
            d_4_stoppedOnOpenSpan_: bool
            d_5_stoppedOnEos_: bool
            d_6_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_3_generatedOut_ = out0_
            d_4_stoppedOnOpenSpan_ = out1_
            d_5_stoppedOnEos_ = out2_
            d_6_stepsUsed_ = out3_
            generated = d_3_generatedOut_
            d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
            if d_4_stoppedOnOpenSpan_:
                d_7_og_: _dafny.Seq
                d_8_oi_: bool
                d_9_oc_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_7_og_ = out4_
                d_8_oi_ = out5_
                d_9_oc_ = out6_
                generated = d_7_og_
                insideConstrainedOut = d_8_oi_
                currentConstrainedOut = d_9_oc_
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_10_og_: _dafny.Seq
            d_11_oi_: bool
            d_12_oc_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_10_og_ = out7_
            d_11_oi_ = out8_
            d_12_oc_ = out9_
            generated = d_10_og_
            insideConstrainedOut = d_11_oi_
            currentConstrainedOut = d_12_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_13_remaining_: int
            d_13_remaining_ = (maxSteps) - (d_1_steps_)
            if (d_13_remaining_) >= (2):
                d_14_sqlBudget_: int = int(0)
                if ((d_13_remaining_) - (1)) <= (300):
                    d_14_sqlBudget_ = (d_13_remaining_) - (1)
                elif True:
                    d_14_sqlBudget_ = 300
                d_15_constrainedPrompt_: _dafny.Seq
                d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_16_resultConstrained_: _dafny.Seq
                out10_: _dafny.Seq
                out10_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken, d_14_sqlBudget_, 5, 50)
                d_16_resultConstrained_ = out10_
                d_17_stablePrefix_: _dafny.Seq
                d_17_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                generated = (d_17_stablePrefix_) + (d_16_resultConstrained_)
                currentConstrainedOut = d_16_resultConstrained_
                d_1_steps_ = (d_1_steps_) + (d_14_sqlBudget_)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_18_cg_: _dafny.Seq
                d_19_ci_: bool
                d_20_cc_: _dafny.Seq
                out11_: _dafny.Seq
                out12_: bool
                out13_: _dafny.Seq
                out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_18_cg_ = out11_
                d_19_ci_ = out12_
                d_20_cc_ = out13_
                generated = d_18_cg_
                insideConstrainedOut = d_19_ci_
                currentConstrainedOut = d_20_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_21_remaining_: int
                d_21_remaining_ = (maxSteps) - (d_1_steps_)
                d_22_closeBudget_: int = int(0)
                if (d_21_remaining_) <= (120):
                    d_22_closeBudget_ = d_21_remaining_
                elif True:
                    d_22_closeBudget_ = 120
                d_23_cg_: _dafny.Seq
                d_24_ci_: bool
                d_25_cc_: _dafny.Seq
                out14_: _dafny.Seq
                out15_: bool
                out16_: _dafny.Seq
                out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_closeBudget_)
                d_23_cg_ = out14_
                d_24_ci_ = out15_
                d_25_cc_ = out16_
                generated = d_23_cg_
                insideConstrainedOut = d_24_ci_
                currentConstrainedOut = d_25_cc_
                d_1_steps_ = (d_1_steps_) + (d_22_closeBudget_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


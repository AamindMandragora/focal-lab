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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write exactly one SQL query. Format: SQL: <<SELECT columns FROM tables WHERE conditions>>. Use only table and column names from the schema. Output nothing after >>.")))
        d_2_phase0Limit_: int
        if (maxSteps) > (15):
            d_2_phase0Limit_ = 15
        elif True:
            d_2_phase0Limit_ = maxSteps
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_2_phase0Limit_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_3_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_3_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_3_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                    if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_4_og_: _dafny.Seq
            d_5_oi_: bool
            d_6_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_4_og_ = out1_
            d_5_oi_ = out2_
            d_6_oc_ = out3_
            generated = d_4_og_
            insideConstrainedOut = d_5_oi_
            currentConstrainedOut = d_6_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_7_rem_: int
            d_7_rem_ = (maxSteps) - (d_1_steps_)
            d_8_rollBudget_: int
            d_8_rollBudget_ = _dafny.euclidian_division((d_7_rem_) * (3), 4)
            if (d_8_rollBudget_) >= (1):
                d_9_constrainedPrompt_: _dafny.Seq
                d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_10_penalties_: _dafny.Seq
                d_10_penalties_ = _dafny.SeqWithoutIsStrInference([])
                d_11_rolloutGen_: _dafny.Seq
                d_12_rolloutSteps_: int
                d_13_rolloutEos_: bool
                out4_: _dafny.Seq
                out5_: int
                out6_: bool
                out4_, out5_, out6_ = (d_0_helpers_).RolloutConstrainedWithPenalties(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, d_8_rollBudget_, d_10_penalties_, _dafny.BigRational('2e0'), eosToken)
                d_11_rolloutGen_ = out4_
                d_12_rolloutSteps_ = out5_
                d_13_rolloutEos_ = out6_
                d_14_stable_: _dafny.Seq
                d_14_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                generated = (d_14_stable_) + (d_11_rolloutGen_)
                currentConstrainedOut = d_11_rolloutGen_
                d_1_steps_ = (d_1_steps_) + (d_12_rolloutSteps_)
                if d_13_rolloutEos_:
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    cost = d_1_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_15_closeBudget_: int
            d_15_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_16_cg2_: _dafny.Seq
            d_17_ci2_: bool
            d_18_cc2_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudget_)
            d_16_cg2_ = out7_
            d_17_ci2_ = out8_
            d_18_cc2_ = out9_
            generated = d_16_cg2_
            insideConstrainedOut = d_17_ci2_
            currentConstrainedOut = d_18_cc2_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


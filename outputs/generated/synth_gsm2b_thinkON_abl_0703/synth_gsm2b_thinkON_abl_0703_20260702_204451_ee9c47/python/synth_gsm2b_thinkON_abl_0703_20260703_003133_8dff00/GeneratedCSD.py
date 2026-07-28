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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. The {variable} placeholders are symbolic - keep them as variables. Write the final answer as a complete Python arithmetic expression using those variable names, inside << >> delimiters. Example: if n things cost c each, write <<n * c>>. Write a full expression, not just a single variable name.")))
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_preambleBudget_: int
            if (maxSteps) > (200):
                d_2_preambleBudget_ = (maxSteps) - (150)
            elif True:
                d_2_preambleBudget_ = _dafny.euclidian_division((maxSteps) * (2), 3)
            with _dafny.label("1_0"):
                while ((d_1_steps_) < (d_2_preambleBudget_)) and (not(insideConstrainedOut)):
                    with _dafny.c_label("1_0"):
                        d_3_chunkBudget_: int
                        if ((d_2_preambleBudget_) - (d_1_steps_)) > (32):
                            d_3_chunkBudget_ = 32
                        elif True:
                            d_3_chunkBudget_ = (d_2_preambleBudget_) - (d_1_steps_)
                        d_4_gOut_: _dafny.Seq
                        d_5_stoppedOnOpen_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_gOut_ = out0_
                        d_5_stoppedOnOpen_ = out1_
                        d_6_stoppedOnEos_ = out2_
                        d_7_stepsUsed_ = out3_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        generated = d_4_gOut_
                        if d_6_stoppedOnEos_:
                            raise _dafny.Break("1_0")
                        if d_5_stoppedOnOpen_:
                            d_8_og_: _dafny.Seq
                            d_9_oi_: bool
                            d_10_oc_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_8_og_ = out4_
                            d_9_oi_ = out5_
                            d_10_oc_ = out6_
                            generated = d_8_og_
                            insideConstrainedOut = d_9_oi_
                            currentConstrainedOut = d_10_oc_
                        pass
                pass
            if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_11_og_: _dafny.Seq
                d_12_oi_: bool
                d_13_oc_: _dafny.Seq
                out7_: _dafny.Seq
                out8_: bool
                out9_: _dafny.Seq
                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_11_og_ = out7_
                d_12_oi_ = out8_
                d_13_oc_ = out9_
                generated = d_11_og_
                insideConstrainedOut = d_12_oi_
                currentConstrainedOut = d_13_oc_
                d_1_steps_ = (d_1_steps_) + (1)
            d_14_closureReserve_: int
            d_14_closureReserve_ = 30
            with _dafny.label("1_1"):
                while ((d_1_steps_) < ((maxSteps) - (d_14_closureReserve_))) and (insideConstrainedOut):
                    with _dafny.c_label("1_1"):
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_16_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_16_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_16_next_) == (eosToken):
                            raise _dafny.Break("1_1")
                        elif True:
                            d_17_ag_: _dafny.Seq
                            d_18_ai_: bool
                            d_19_ac_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                            d_17_ag_ = out11_
                            d_18_ai_ = out12_
                            d_19_ac_ = out13_
                            generated = d_17_ag_
                            insideConstrainedOut = d_18_ai_
                            currentConstrainedOut = d_19_ac_
                            if not(insideConstrainedOut):
                                raise _dafny.Break("1_1")
                        pass
                pass
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_20_remainBudget_: int
                d_20_remainBudget_ = (maxSteps) - (d_1_steps_)
                d_21_cg_: _dafny.Seq
                d_22_ci_: bool
                d_23_cc_: _dafny.Seq
                out14_: _dafny.Seq
                out15_: bool
                out16_: _dafny.Seq
                out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_20_remainBudget_)
                d_21_cg_ = out14_
                d_22_ci_ = out15_
                d_23_cc_ = out16_
                generated = d_21_cg_
                insideConstrainedOut = d_22_ci_
                currentConstrainedOut = d_23_cc_
                d_1_steps_ = maxSteps
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


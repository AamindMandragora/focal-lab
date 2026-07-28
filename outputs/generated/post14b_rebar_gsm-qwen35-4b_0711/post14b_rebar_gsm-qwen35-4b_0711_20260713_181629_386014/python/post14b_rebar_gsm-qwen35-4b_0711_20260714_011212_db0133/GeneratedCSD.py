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
        if (maxSteps) == (0):
            pass
        elif True:
            d_1_guidance_: _dafny.Seq
            d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step. At the very end, write your final answer as a single arithmetic expression inside << >>. IMPORTANT: Use plain variable names without curly braces. Write n not {n}, write total not {total}. Allowed: variable names, numbers, +, -, *, /, //, %, (, ), int(). Example: <<n * (m + k) // 12>>. Do NOT write << >> anywhere except for the final answer."))
            (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
            d_2_steps_: int
            d_2_steps_ = 0
            d_3_hitEos_: bool
            d_3_hitEos_ = False
            d_4_reserve_: int
            d_4_reserve_ = 10
            if (d_4_reserve_) >= (maxSteps):
                d_4_reserve_ = 0
            d_5_freeLimit_: int
            d_5_freeLimit_ = (maxSteps) - (d_4_reserve_)
            if (d_5_freeLimit_) == (0):
                d_5_freeLimit_ = 1
            if (d_5_freeLimit_) > (maxSteps):
                d_5_freeLimit_ = maxSteps
            while (((d_2_steps_) < (d_5_freeLimit_)) and (not(insideConstrainedOut))) and (not(d_3_hitEos_)):
                d_6_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_6_next_ = out0_
                d_2_steps_ = (d_2_steps_) + (1)
                if (d_6_next_) == (eosToken):
                    d_3_hitEos_ = True
                elif True:
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                    if ((d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and (not(insideConstrainedOut)):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            if ((insideConstrainedOut) and ((d_2_steps_) < (maxSteps))) and (not(d_3_hitEos_)):
                d_7_closeBudget_: int
                d_7_closeBudget_ = (maxSteps) - (d_2_steps_)
                d_8_cg_: _dafny.Seq
                d_9_ci_: bool
                d_10_cc_: _dafny.Seq
                out1_: _dafny.Seq
                out2_: bool
                out3_: _dafny.Seq
                out1_, out2_, out3_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_7_closeBudget_)
                d_8_cg_ = out1_
                d_9_ci_ = out2_
                d_10_cc_ = out3_
                generated = d_8_cg_
                insideConstrainedOut = d_9_ci_
                currentConstrainedOut = d_10_cc_
                d_2_steps_ = maxSteps
            elif ((not(insideConstrainedOut)) and (not(d_3_hitEos_))) and ((d_2_steps_) < (maxSteps)):
                while (((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut))) and (not(d_3_hitEos_)):
                    d_11_next2_: _dafny.Seq
                    out4_: _dafny.Seq
                    out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_11_next2_ = out4_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_11_next2_) == (eosToken):
                        d_3_hitEos_ = True
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next2_]))
                        if ((d_11_next2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and (not(insideConstrainedOut)):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                if ((insideConstrainedOut) and ((d_2_steps_) < (maxSteps))) and (not(d_3_hitEos_)):
                    d_12_closeBudget2_: int
                    d_12_closeBudget2_ = (maxSteps) - (d_2_steps_)
                    d_13_cg2_: _dafny.Seq
                    d_14_ci2_: bool
                    d_15_cc2_: _dafny.Seq
                    out5_: _dafny.Seq
                    out6_: bool
                    out7_: _dafny.Seq
                    out5_, out6_, out7_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_12_closeBudget2_)
                    d_13_cg2_ = out5_
                    d_14_ci2_ = out6_
                    d_15_cc2_ = out7_
                    generated = d_13_cg2_
                    insideConstrainedOut = d_14_ci2_
                    currentConstrainedOut = d_15_cc2_
                    d_2_steps_ = maxSteps
            cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


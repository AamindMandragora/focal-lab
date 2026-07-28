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
            d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Use the EXACT variable names from the problem (keep underscores, e.g. frac_1 not frac1, ball_type_1 not balltype1). At the very end write your final answer as <<expr>> where expr uses ONLY: exact variable names, numbers, +, -, *, /, //, %, (, ), int(). Do NOT use { or } in expr. Do NOT use **. Use int() for integer counts. Wait until your final answer to open <<. Example: <<int(n * frac_1 * frac_2)>>. Write exactly ONE <<expr>> at the end."))
            (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
            d_2_freeBudget_: int
            d_2_freeBudget_ = _dafny.euclidian_division((maxSteps) * (90), 100)
            if (d_2_freeBudget_) == (0):
                d_2_freeBudget_ = 1
            if (d_2_freeBudget_) >= (maxSteps):
                d_2_freeBudget_ = (maxSteps) - (1)
            d_3_steps_: int
            d_3_steps_ = 0
            d_4_hitEos_: bool
            d_4_hitEos_ = False
            while ((d_3_steps_) < (d_2_freeBudget_)) and (not(d_4_hitEos_)):
                d_5_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_5_next_ = out0_
                d_3_steps_ = (d_3_steps_) + (1)
                if (d_5_next_) == (eosToken):
                    d_4_hitEos_ = True
                elif True:
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                    if ((d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and (not(insideConstrainedOut)):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            if ((insideConstrainedOut) and ((d_3_steps_) < (maxSteps))) and (not(d_4_hitEos_)):
                d_6_remaining_: int
                d_6_remaining_ = (maxSteps) - (d_3_steps_)
                d_7_cap15_: int
                d_7_cap15_ = 15
                d_8_closeBudget_: int = int(0)
                if (d_6_remaining_) < (d_7_cap15_):
                    d_8_closeBudget_ = d_6_remaining_
                elif True:
                    d_8_closeBudget_ = d_7_cap15_
                d_9_cg_: _dafny.Seq
                d_10_ci_: bool
                d_11_cc_: _dafny.Seq
                out1_: _dafny.Seq
                out2_: bool
                out3_: _dafny.Seq
                out1_, out2_, out3_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_8_closeBudget_)
                d_9_cg_ = out1_
                d_10_ci_ = out2_
                d_11_cc_ = out3_
                generated = d_9_cg_
                insideConstrainedOut = d_10_ci_
                currentConstrainedOut = d_11_cc_
                d_3_steps_ = (d_3_steps_) + (d_8_closeBudget_)
            if ((not(insideConstrainedOut)) and ((d_3_steps_) < (maxSteps))) and (not(d_4_hitEos_)):
                d_12_openG_: _dafny.Seq
                d_13_openI_: bool
                d_14_openC_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_12_openG_ = out4_
                d_13_openI_ = out5_
                d_14_openC_ = out6_
                generated = d_12_openG_
                insideConstrainedOut = d_13_openI_
                currentConstrainedOut = d_14_openC_
                d_3_steps_ = (d_3_steps_) + (1)
                if (d_3_steps_) < (maxSteps):
                    d_15_closeBudget2_: int
                    d_15_closeBudget2_ = (maxSteps) - (d_3_steps_)
                    d_16_cg2_: _dafny.Seq
                    d_17_ci2_: bool
                    d_18_cc2_: _dafny.Seq
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudget2_)
                    d_16_cg2_ = out7_
                    d_17_ci2_ = out8_
                    d_18_cc2_ = out9_
                    generated = d_16_cg2_
                    insideConstrainedOut = d_17_ci2_
                    currentConstrainedOut = d_18_cc2_
                    d_3_steps_ = maxSteps
            cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


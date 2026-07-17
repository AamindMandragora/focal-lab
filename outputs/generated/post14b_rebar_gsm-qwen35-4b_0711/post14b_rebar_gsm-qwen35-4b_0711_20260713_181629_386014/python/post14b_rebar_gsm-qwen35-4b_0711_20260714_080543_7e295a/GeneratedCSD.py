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
            d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Work step by step. At the very end write ONLY <<expr>> where expr uses variable names, numbers, +, -, *, /, //, %, (, ), int(). No ** and no { }. Examples: <<int(n*k+m)>>, <<(a+b)*c//12>>, <<n1*c1+n2*c2+c3>>. One final <<expr>> only."))
            (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
            d_2_reserveBudget_: int
            d_2_reserveBudget_ = 20
            d_3_freeBudget_: int = int(0)
            if (maxSteps) <= (d_2_reserveBudget_):
                d_3_freeBudget_ = (maxSteps) - (1)
                if (d_3_freeBudget_) == (0):
                    d_3_freeBudget_ = 1
            elif True:
                d_3_freeBudget_ = (maxSteps) - (d_2_reserveBudget_)
            if (d_3_freeBudget_) >= (maxSteps):
                d_3_freeBudget_ = (maxSteps) - (1)
            d_4_steps_: int
            d_4_steps_ = 0
            d_5_hitEos_: bool
            d_5_hitEos_ = False
            while ((d_4_steps_) < (d_3_freeBudget_)) and (not(d_5_hitEos_)):
                d_6_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_6_next_ = out0_
                d_4_steps_ = (d_4_steps_) + (1)
                if (d_6_next_) == (eosToken):
                    d_5_hitEos_ = True
                elif True:
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                    if ((d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and (not(insideConstrainedOut)):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            if ((insideConstrainedOut) and ((d_4_steps_) < (maxSteps))) and (not(d_5_hitEos_)):
                d_7_remaining_: int
                d_7_remaining_ = (maxSteps) - (d_4_steps_)
                d_8_cap10_: int
                d_8_cap10_ = 10
                d_9_closeBudget10_: int = int(0)
                if (d_7_remaining_) < (d_8_cap10_):
                    d_9_closeBudget10_ = d_7_remaining_
                elif True:
                    d_9_closeBudget10_ = d_8_cap10_
                if (d_9_closeBudget10_) > (0):
                    d_10_cg_: _dafny.Seq
                    d_11_ci_: bool
                    d_12_cc_: _dafny.Seq
                    out1_: _dafny.Seq
                    out2_: bool
                    out3_: _dafny.Seq
                    out1_, out2_, out3_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_9_closeBudget10_)
                    d_10_cg_ = out1_
                    d_11_ci_ = out2_
                    d_12_cc_ = out3_
                    generated = d_10_cg_
                    insideConstrainedOut = d_11_ci_
                    currentConstrainedOut = d_12_cc_
                    d_4_steps_ = (d_4_steps_) + (d_9_closeBudget10_)
            if ((not(insideConstrainedOut)) and ((d_4_steps_) < (maxSteps))) and (not(d_5_hitEos_)):
                d_13_openG_: _dafny.Seq
                d_14_openI_: bool
                d_15_openC_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_13_openG_ = out4_
                d_14_openI_ = out5_
                d_15_openC_ = out6_
                generated = d_13_openG_
                insideConstrainedOut = d_14_openI_
                currentConstrainedOut = d_15_openC_
                d_4_steps_ = (d_4_steps_) + (1)
                if (d_4_steps_) < (maxSteps):
                    d_16_remaining2_: int
                    d_16_remaining2_ = (maxSteps) - (d_4_steps_)
                    d_17_cap18_: int
                    d_17_cap18_ = 18
                    d_18_closeBudget2_: int = int(0)
                    if (d_16_remaining2_) < (d_17_cap18_):
                        d_18_closeBudget2_ = d_16_remaining2_
                    elif True:
                        d_18_closeBudget2_ = d_17_cap18_
                    if (d_18_closeBudget2_) > (0):
                        d_19_cg2_: _dafny.Seq
                        d_20_ci2_: bool
                        d_21_cc2_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_closeBudget2_)
                        d_19_cg2_ = out7_
                        d_20_ci2_ = out8_
                        d_21_cc2_ = out9_
                        generated = d_19_cg2_
                        insideConstrainedOut = d_20_ci2_
                        currentConstrainedOut = d_21_cc2_
                        d_4_steps_ = (d_4_steps_) + (d_18_closeBudget2_)
            cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


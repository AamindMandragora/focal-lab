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
            cost = 0
        elif True:
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_hasClosedSpan_: bool
            d_2_hasClosedSpan_ = False
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_3_spanBudget_: int
                d_3_spanBudget_ = (maxSteps) - (d_1_steps_)
                if (d_3_spanBudget_) > (64):
                    d_3_spanBudget_ = 64
                d_4_cg_: _dafny.Seq
                d_5_ci_: bool
                d_6_cc_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_3_spanBudget_)
                d_4_cg_ = out0_
                d_5_ci_ = out1_
                d_6_cc_ = out2_
                generated = d_4_cg_
                insideConstrainedOut = d_5_ci_
                currentConstrainedOut = d_6_cc_
                d_1_steps_ = (d_1_steps_) + (d_3_spanBudget_)
                if not(insideConstrainedOut):
                    d_2_hasClosedSpan_ = True
            with _dafny.label("1_0"):
                while ((d_1_steps_) < (maxSteps)) and (not(d_2_hasClosedSpan_)):
                    with _dafny.c_label("1_0"):
                        if not(insideConstrainedOut):
                            d_7_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("1_0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_8_g2_: _dafny.Seq
                                    d_9_ic2_: bool
                                    d_10_cc2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_8_g2_ = out4_
                                    d_9_ic2_ = out5_
                                    d_10_cc2_ = out6_
                                    generated = d_8_g2_
                                    insideConstrainedOut = d_9_ic2_
                                    currentConstrainedOut = d_10_cc2_
                        elif True:
                            d_11_spanBudget_: int
                            d_11_spanBudget_ = (maxSteps) - (d_1_steps_)
                            if (d_11_spanBudget_) > (64):
                                d_11_spanBudget_ = 64
                            d_12_remaining_: int
                            d_12_remaining_ = (maxSteps) - (d_1_steps_)
                            if (d_11_spanBudget_) > (d_12_remaining_):
                                d_11_spanBudget_ = d_12_remaining_
                            d_13_cg_: _dafny.Seq
                            d_14_ci_: bool
                            d_15_cc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_spanBudget_)
                            d_13_cg_ = out7_
                            d_14_ci_ = out8_
                            d_15_cc_ = out9_
                            generated = d_13_cg_
                            insideConstrainedOut = d_14_ci_
                            currentConstrainedOut = d_15_cc_
                            d_1_steps_ = (d_1_steps_) + (d_11_spanBudget_)
                            if not(insideConstrainedOut):
                                d_2_hasClosedSpan_ = True
                            elif True:
                                raise _dafny.Break("1_0")
                        pass
                pass
            if ((not(d_2_hasClosedSpan_)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
                d_16_g2_: _dafny.Seq
                d_17_ic2_: bool
                d_18_cc2_: _dafny.Seq
                out10_: _dafny.Seq
                out11_: bool
                out12_: _dafny.Seq
                out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_16_g2_ = out10_
                d_17_ic2_ = out11_
                d_18_cc2_ = out12_
                generated = d_16_g2_
                insideConstrainedOut = d_17_ic2_
                currentConstrainedOut = d_18_cc2_
                d_1_steps_ = (d_1_steps_) + (1)
                if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                    d_19_spanBudget_: int
                    d_19_spanBudget_ = (maxSteps) - (d_1_steps_)
                    if (d_19_spanBudget_) > (64):
                        d_19_spanBudget_ = 64
                    d_20_cg_: _dafny.Seq
                    d_21_ci_: bool
                    d_22_cc_: _dafny.Seq
                    out13_: _dafny.Seq
                    out14_: bool
                    out15_: _dafny.Seq
                    out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_spanBudget_)
                    d_20_cg_ = out13_
                    d_21_ci_ = out14_
                    d_22_cc_ = out15_
                    generated = d_20_cg_
                    insideConstrainedOut = d_21_ci_
                    currentConstrainedOut = d_22_cc_
                    d_1_steps_ = (d_1_steps_) + (d_19_spanBudget_)
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


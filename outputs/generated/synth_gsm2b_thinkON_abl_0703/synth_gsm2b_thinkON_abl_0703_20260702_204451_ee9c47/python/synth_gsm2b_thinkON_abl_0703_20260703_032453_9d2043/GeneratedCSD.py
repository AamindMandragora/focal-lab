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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. At the very end, write ONLY the final arithmetic expression inside << >> like this: <<n * (mult + 1)>>. Use plain variable names (n, m, k, frac, mult, etc.) NOT curly braces. Use Python operators +, -, *, /. Example: <<n - frac * n>>. Put ONLY the expression inside << >>, nothing else.")))
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_closureReserve_: int
            d_2_closureReserve_ = 150
            d_3_preambleBudget_: int
            if (maxSteps) > (d_2_closureReserve_):
                d_3_preambleBudget_ = (maxSteps) - (d_2_closureReserve_)
            elif True:
                d_3_preambleBudget_ = 0
            with _dafny.label("1_0"):
                while ((d_1_steps_) < (d_3_preambleBudget_)) and (not(insideConstrainedOut)):
                    with _dafny.c_label("1_0"):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("1_0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_5_og_: _dafny.Seq
                                d_6_oi_: bool
                                d_7_oc_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_5_og_ = out1_
                                d_6_oi_ = out2_
                                d_7_oc_ = out3_
                                generated = d_5_og_
                                insideConstrainedOut = d_6_oi_
                                currentConstrainedOut = d_7_oc_
                        pass
                pass
            if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_8_extraBudget_: int
                if ((maxSteps) - (d_1_steps_)) > (20):
                    d_8_extraBudget_ = 20
                elif True:
                    d_8_extraBudget_ = (maxSteps) - (d_1_steps_)
                d_9_extraSteps_: int
                d_9_extraSteps_ = 0
                while ((d_9_extraSteps_) < (d_8_extraBudget_)) and (not(insideConstrainedOut)):
                    d_10_next_: _dafny.Seq
                    out4_: _dafny.Seq
                    out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_10_next_ = out4_
                    d_9_extraSteps_ = (d_9_extraSteps_) + (1)
                    if (d_10_next_) == (eosToken):
                        d_9_extraSteps_ = d_8_extraBudget_
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                        if (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_11_og_: _dafny.Seq
                            d_12_oi_: bool
                            d_13_oc_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_11_og_ = out5_
                            d_12_oi_ = out6_
                            d_13_oc_ = out7_
                            generated = d_11_og_
                            insideConstrainedOut = d_12_oi_
                            currentConstrainedOut = d_13_oc_
                d_1_steps_ = (d_1_steps_) + (d_9_extraSteps_)
            if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_14_og_: _dafny.Seq
                d_15_oi_: bool
                d_16_oc_: _dafny.Seq
                out8_: _dafny.Seq
                out9_: bool
                out10_: _dafny.Seq
                out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_14_og_ = out8_
                d_15_oi_ = out9_
                d_16_oc_ = out10_
                generated = d_14_og_
                insideConstrainedOut = d_15_oi_
                currentConstrainedOut = d_16_oc_
                d_1_steps_ = (d_1_steps_) + (1)
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_17_closeBudget_: int
                d_17_closeBudget_ = (maxSteps) - (d_1_steps_)
                d_18_cg_: _dafny.Seq
                d_19_ci_: bool
                d_20_cc_: _dafny.Seq
                out11_: _dafny.Seq
                out12_: bool
                out13_: _dafny.Seq
                out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget_)
                d_18_cg_ = out11_
                d_19_ci_ = out12_
                d_20_cc_ = out13_
                generated = d_18_cg_
                insideConstrainedOut = d_19_ci_
                currentConstrainedOut = d_20_cc_
                d_1_steps_ = maxSteps
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


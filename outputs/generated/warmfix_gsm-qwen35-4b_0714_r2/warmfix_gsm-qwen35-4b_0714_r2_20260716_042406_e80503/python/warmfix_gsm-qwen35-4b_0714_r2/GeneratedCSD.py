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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Wrap each intermediate result and the final answer in <<expr>>. CRITICAL inside << >>: bare variable names with NO curly braces. Write n not {n}, frac_1 not {frac_1}, mult not {mult}. Allowed: variable names, numbers, +, -, *, /, //, %, (, ), int(). Use // for integer division. Wrap whole-number results in int(). No LaTeX, no {}, no $, no **. Each answer: one <<expr>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanEverOpened_: bool
        d_2_spanEverOpened_ = insideConstrained
        d_3_forceThreshold_: int
        if (maxSteps) > (200):
            d_3_forceThreshold_ = (maxSteps) - (200)
        elif True:
            d_3_forceThreshold_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_1_steps_) >= (d_3_forceThreshold_)) and (not(d_2_spanEverOpened_)):
                            d_4_og_: _dafny.Seq
                            d_5_oi_: bool
                            d_6_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_og_ = out0_
                            d_5_oi_ = out1_
                            d_6_oc_ = out2_
                            generated = d_4_og_
                            insideConstrainedOut = d_5_oi_
                            currentConstrainedOut = d_6_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanEverOpened_ = True
                        elif True:
                            d_7_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_spanEverOpened_ = True
                    elif True:
                        d_8_cg_: _dafny.Seq
                        d_9_ci_: bool
                        d_10_cc_: _dafny.Seq
                        d_11_closed_: bool
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_8_cg_ = out4_
                        d_9_ci_ = out5_
                        d_10_cc_ = out6_
                        d_11_closed_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_11_closed_:
                            generated = d_8_cg_
                            insideConstrainedOut = d_9_ci_
                            currentConstrainedOut = d_10_cc_
                        elif True:
                            d_12_constrainedPrompt_: _dafny.Seq
                            d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_13_next_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_13_next_ = out8_
                            if (d_13_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_14_valid_: bool
                                out9_: bool
                                out9_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_13_next_)
                                d_14_valid_ = out9_
                                if d_14_valid_:
                                    d_15_ag_: _dafny.Seq
                                    d_16_ai_: bool
                                    d_17_ac_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                    d_15_ag_ = out10_
                                    d_16_ai_ = out11_
                                    d_17_ac_ = out12_
                                    generated = d_15_ag_
                                    insideConstrainedOut = d_16_ai_
                                    currentConstrainedOut = d_17_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_18_closeBudget_: int
            d_18_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_19_cg_: _dafny.Seq
            d_20_ci_: bool
            d_21_cc_: _dafny.Seq
            out13_: _dafny.Seq
            out14_: bool
            out15_: _dafny.Seq
            out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_closeBudget_)
            d_19_cg_ = out13_
            d_20_ci_ = out14_
            d_21_cc_ = out15_
            generated = d_19_cg_
            insideConstrainedOut = d_20_ci_
            currentConstrainedOut = d_21_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


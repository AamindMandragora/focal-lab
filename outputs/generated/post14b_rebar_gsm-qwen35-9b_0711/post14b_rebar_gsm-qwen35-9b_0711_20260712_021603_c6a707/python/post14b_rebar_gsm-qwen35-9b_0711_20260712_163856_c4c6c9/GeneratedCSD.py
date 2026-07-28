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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. End with exactly one << >> containing a symbolic math expression using only the variable names from the problem (without {} braces). Use // for integer division when converting units (e.g. minutes to hours). Use int() around the entire expression when the result must be a whole number (counts, people, money). Use * for multiplication only, never **. No {} braces, no literals except 0 or 1 if needed."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_reserveSteps_: int
        d_3_reserveSteps_ = 120
        d_4_preambleLimit_: int
        if (maxSteps) > (d_3_reserveSteps_):
            d_4_preambleLimit_ = (maxSteps) - (d_3_reserveSteps_)
        elif True:
            d_4_preambleLimit_ = 0
        if not(insideConstrainedOut):
            with _dafny.label("0_0"):
                while (d_2_steps_) < (d_4_preambleLimit_):
                    with _dafny.c_label("0_0"):
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0_0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        pass
                pass
        if ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_6_og_: _dafny.Seq
            d_7_oi_: bool
            d_8_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_6_og_ = out1_
            d_7_oi_ = out2_
            d_8_oc_ = out3_
            generated = d_6_og_
            insideConstrainedOut = d_7_oi_
            currentConstrainedOut = d_8_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        d_9_minTokensBeforeClose_: int
        d_9_minTokensBeforeClose_ = 2
        d_10_preCloseCount_: int
        d_10_preCloseCount_ = 0
        while (((d_10_preCloseCount_) < (d_9_minTokensBeforeClose_)) and ((d_2_steps_) < (maxSteps))) and (insideConstrainedOut):
            d_11_constrainedPrompt_: _dafny.Seq
            d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
            d_12_next_: _dafny.Seq
            out4_: _dafny.Seq
            out4_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
            d_12_next_ = out4_
            d_2_steps_ = (d_2_steps_) + (1)
            d_10_preCloseCount_ = (d_10_preCloseCount_) + (1)
            if (d_12_next_) == (eosToken):
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            elif True:
                d_13_ag_: _dafny.Seq
                d_14_ai_: bool
                d_15_ac_: _dafny.Seq
                out5_: _dafny.Seq
                out6_: bool
                out7_: _dafny.Seq
                out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                d_13_ag_ = out5_
                d_14_ai_ = out6_
                d_15_ac_ = out7_
                generated = d_13_ag_
                insideConstrainedOut = d_14_ai_
                currentConstrainedOut = d_15_ac_
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_16_closeBudget_: int
            d_16_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_17_cg_: _dafny.Seq
            d_18_ci_: bool
            d_19_cc_: _dafny.Seq
            out8_: _dafny.Seq
            out9_: bool
            out10_: _dafny.Seq
            out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_16_closeBudget_)
            d_17_cg_ = out8_
            d_18_ci_ = out9_
            d_19_cc_ = out10_
            generated = d_17_cg_
            insideConstrainedOut = d_18_ci_
            currentConstrainedOut = d_19_cc_
            d_2_steps_ = (d_2_steps_) + (d_16_closeBudget_)
            if (d_2_steps_) > (maxSteps):
                d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


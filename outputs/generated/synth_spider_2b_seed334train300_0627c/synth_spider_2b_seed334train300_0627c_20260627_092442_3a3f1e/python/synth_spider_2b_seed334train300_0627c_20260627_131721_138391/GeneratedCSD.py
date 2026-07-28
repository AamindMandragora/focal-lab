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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write the simplest SQL that answers the question. NO table aliases. NO column aliases. NO dot notation. Use plain column names. Use the exact table and column names from the schema. Output format: SQL: <<SELECT col FROM table WHERE ...>>"))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_preambleLimit_: int
        d_3_preambleLimit_ = 10
        with _dafny.label("0"):
            while (((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut))) and ((d_2_steps_) < (d_3_preambleLimit_)):
                with _dafny.c_label("0"):
                    d_4_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_4_next_ = out0_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_4_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                        if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_5_og_: _dafny.Seq
            d_6_oi_: bool
            d_7_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_5_og_ = out1_
            d_6_oi_ = out2_
            d_7_oc_ = out3_
            generated = d_5_og_
            insideConstrainedOut = d_6_oi_
            currentConstrainedOut = d_7_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_8_rem_: int
            d_8_rem_ = (maxSteps) - (d_2_steps_)
            d_9_closeReserve_: int
            if (d_8_rem_) >= (20):
                d_9_closeReserve_ = 10
            elif True:
                d_9_closeReserve_ = _dafny.euclidian_division(d_8_rem_, 2)
            d_10_fillBudget_: int
            d_10_fillBudget_ = (d_8_rem_) - (d_9_closeReserve_)
            if (d_10_fillBudget_) >= (1):
                d_11_stableLen_: int
                d_11_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
                d_12_stable_: _dafny.Seq
                d_12_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:d_11_stableLen_:])
                d_13_filled_: _dafny.Seq
                out4_: _dafny.Seq
                out4_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, (prompt) + (d_12_stable_), currentConstrainedOut, eosToken, d_10_fillBudget_, 3, d_10_fillBudget_)
                d_13_filled_ = out4_
                generated = (d_12_stable_) + (d_13_filled_)
                currentConstrainedOut = d_13_filled_
                d_2_steps_ = (d_2_steps_) + (d_10_fillBudget_)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_14_closeBudget_: int
            d_14_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_15_cg_: _dafny.Seq
            d_16_ci_: bool
            d_17_cc_: _dafny.Seq
            out5_: _dafny.Seq
            out6_: bool
            out7_: _dafny.Seq
            out5_, out6_, out7_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudget_)
            d_15_cg_ = out5_
            d_16_ci_ = out6_
            d_17_cc_ = out7_
            generated = d_15_cg_
            insideConstrainedOut = d_16_ci_
            currentConstrainedOut = d_17_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


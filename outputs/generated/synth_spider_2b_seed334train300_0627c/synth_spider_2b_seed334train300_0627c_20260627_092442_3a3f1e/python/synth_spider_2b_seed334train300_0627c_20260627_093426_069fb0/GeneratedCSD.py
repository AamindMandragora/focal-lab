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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query answering the question. Format: SQL: <<YOUR SQL QUERY HERE>> Use only the tables and columns from the provided schema. No explanation, no markdown, just the SQL query inside the delimiters."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        with _dafny.label("0"):
            while ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_3_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_3_next_ = out0_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_3_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                    if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_4_rem_: int
            d_4_rem_ = (maxSteps) - (d_2_steps_)
            d_5_fillBudget_: int
            d_5_fillBudget_ = _dafny.euclidian_division(d_4_rem_, 2)
            if (d_5_fillBudget_) >= (1):
                d_6_stableLen_: int
                d_6_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
                d_7_stable_: _dafny.Seq
                d_7_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:d_6_stableLen_:])
                d_8_constrainedPrompt_: _dafny.Seq
                d_8_constrainedPrompt_ = (prompt) + (d_7_stable_)
                d_9_filled_: _dafny.Seq
                out1_: _dafny.Seq
                out1_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, eosToken, d_5_fillBudget_, 3, d_5_fillBudget_)
                d_9_filled_ = out1_
                generated = (d_7_stable_) + (d_9_filled_)
                currentConstrainedOut = d_9_filled_
                d_2_steps_ = (d_2_steps_) + (d_5_fillBudget_)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_10_closeBudget_: int
            d_10_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_11_cg_: _dafny.Seq
            d_12_ci_: bool
            d_13_cc_: _dafny.Seq
            out2_: _dafny.Seq
            out3_: bool
            out4_: _dafny.Seq
            out2_, out3_, out4_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_10_closeBudget_)
            d_11_cg_ = out2_
            d_12_ci_ = out3_
            d_13_cc_ = out4_
            generated = d_11_cg_
            insideConstrainedOut = d_12_ci_
            currentConstrainedOut = d_13_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


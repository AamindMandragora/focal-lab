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
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step in plain text. At the very end, write exactly ONE expression inside << >> using only: variable names (letters/underscores), numbers, +, -, *, /, //, %, (, ), int(). NO curly braces, NO **, NO LaTeX. Example: <<n * price - base>>. Write << >> ONCE at the END."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_spanDone_: bool
        d_3_spanDone_ = False
        while (((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut))) and (not(d_3_spanDone_)):
            d_4_next_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_4_next_ = out0_
            d_2_steps_ = (d_2_steps_) + (1)
            if (d_4_next_) == (eosToken):
                d_3_spanDone_ = True
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                    d_5_eg_: _dafny.Seq
                    d_6_ei_: bool
                    d_7_ec_: _dafny.Seq
                    out1_: _dafny.Seq
                    out2_: bool
                    out3_: _dafny.Seq
                    out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_5_eg_ = out1_
                    d_6_ei_ = out2_
                    d_7_ec_ = out3_
                    generated = d_5_eg_
                    insideConstrainedOut = d_6_ei_
                    currentConstrainedOut = d_7_ec_
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_8_closeBudget_: int
            d_8_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_9_cg_: _dafny.Seq
            d_10_ci_: bool
            d_11_cc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_8_closeBudget_)
            d_9_cg_ = out4_
            d_10_ci_ = out5_
            d_11_cc_ = out6_
            generated = d_9_cg_
            insideConstrainedOut = d_10_ci_
            currentConstrainedOut = d_11_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step in plain text. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "At the very end of your solution, write the COMPLETE combined final answer as ONE ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "arithmetic expression inside a single << >> delimiter. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Put ALL variables and operations together in that one expression. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Example: <<(w1 + w2 + w3) * price>> or <<n - n_1 - 3*n_2>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Do NOT write multiple << >> for sub-expressions; reserve << >> for the final answer only."))))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_RESERVE_: int
        d_2_RESERVE_ = 80
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_3_avail0_: int
            d_3_avail0_ = (maxSteps) - (d_1_steps_)
            d_4_closeB0_: int
            if (d_3_avail0_) <= (d_2_RESERVE_):
                d_4_closeB0_ = d_3_avail0_
            elif True:
                d_4_closeB0_ = d_2_RESERVE_
            d_5_cg0_: _dafny.Seq
            d_6_ci0_: bool
            d_7_cc0_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_4_closeB0_)
            d_5_cg0_ = out0_
            d_6_ci0_ = out1_
            d_7_cc0_ = out2_
            generated = d_5_cg0_
            insideConstrainedOut = d_6_ci0_
            currentConstrainedOut = d_7_cc0_
            d_1_steps_ = (d_1_steps_) + (d_4_closeB0_)
        with _dafny.label("0"):
            while (((d_1_steps_) + (d_2_RESERVE_)) < (maxSteps)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_8_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_8_next_ = out3_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_8_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                    pass
            pass
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_9_og_: _dafny.Seq
            d_10_oi_: bool
            d_11_oc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_9_og_ = out4_
            d_10_oi_ = out5_
            d_11_oc_ = out6_
            generated = d_9_og_
            insideConstrainedOut = d_10_oi_
            currentConstrainedOut = d_11_oc_
            d_1_steps_ = (d_1_steps_) + (1)
            if (d_1_steps_) < (maxSteps):
                d_12_closeB2_: int
                d_12_closeB2_ = (maxSteps) - (d_1_steps_)
                d_13_cg2_: _dafny.Seq
                d_14_ci2_: bool
                d_15_cc2_: _dafny.Seq
                out7_: _dafny.Seq
                out8_: bool
                out9_: _dafny.Seq
                out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_12_closeB2_)
                d_13_cg2_ = out7_
                d_14_ci2_ = out8_
                d_15_cc2_ = out9_
                generated = d_13_cg2_
                insideConstrainedOut = d_14_ci2_
                currentConstrainedOut = d_15_cc2_
                d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


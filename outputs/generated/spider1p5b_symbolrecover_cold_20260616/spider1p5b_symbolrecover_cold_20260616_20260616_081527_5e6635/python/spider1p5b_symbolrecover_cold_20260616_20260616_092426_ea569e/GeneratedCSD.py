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
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You must respond with exactly: SQL: <<SELECT ...>> where the query uses only the tables and columns in the schema. Output nothing else.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_preambleBudget_: int
        d_2_preambleBudget_ = 8
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and ((d_1_steps_) < (d_2_preambleBudget_))) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_3_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_3_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_3_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                    if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_4_og_: _dafny.Seq
            d_5_oi_: bool
            d_6_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_4_og_ = out1_
            d_5_oi_ = out2_
            d_6_oc_ = out3_
            d_1_steps_ = (d_1_steps_) + (1)
            generated = d_4_og_
            insideConstrainedOut = d_5_oi_
            currentConstrainedOut = d_6_oc_
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_7_closeBudget_: int
            d_7_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_8_cg_: _dafny.Seq
            d_9_ci_: bool
            d_10_cc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_7_closeBudget_)
            d_8_cg_ = out4_
            d_9_ci_ = out5_
            d_10_cc_ = out6_
            generated = d_8_cg_
            insideConstrainedOut = d_9_ci_
            currentConstrainedOut = d_10_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


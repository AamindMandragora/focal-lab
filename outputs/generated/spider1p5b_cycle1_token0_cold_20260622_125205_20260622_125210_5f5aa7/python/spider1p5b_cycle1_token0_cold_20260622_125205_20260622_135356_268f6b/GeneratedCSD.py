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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: followed by a single valid SQL query using only the provided schema tables and columns. No explanation, no markdown, no extra text. Example: SQL: SELECT column FROM table WHERE condition")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_2_remainingBudget_: int
            d_2_remainingBudget_ = (maxSteps) - (d_1_steps_)
            d_3_constrainedResult_: _dafny.Seq
            d_4_terminatedByEos_: bool
            out0_: _dafny.Seq
            out1_: bool
            out0_, out1_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, prompt, d_2_remainingBudget_, eosToken)
            d_3_constrainedResult_ = out0_
            d_4_terminatedByEos_ = out1_
            d_5_resultLen_: int
            d_5_resultLen_ = len(d_3_constrainedResult_)
            generated = (generated) + (d_3_constrainedResult_)
            d_1_steps_ = (d_1_steps_) + (d_5_resultLen_)
            if (d_4_terminatedByEos_) and ((d_1_steps_) < (maxSteps)):
                d_1_steps_ = (d_1_steps_) + (1)
            if (d_1_steps_) > (maxSteps):
                d_1_steps_ = maxSteps
        d_6_narrowThreshold_: int
        d_6_narrowThreshold_ = 12
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_7_cg_: _dafny.Seq
                    d_8_ci_: bool
                    d_9_cc_: _dafny.Seq
                    d_10_closed_: bool
                    out2_: _dafny.Seq
                    out3_: bool
                    out4_: _dafny.Seq
                    out5_: bool
                    out2_, out3_, out4_, out5_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_7_cg_ = out2_
                    d_8_ci_ = out3_
                    d_9_cc_ = out4_
                    d_10_closed_ = out5_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_10_closed_:
                        generated = d_7_cg_
                        insideConstrainedOut = d_8_ci_
                        currentConstrainedOut = d_9_cc_
                        raise _dafny.Break("0")
                    elif True:
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_12_next_: _dafny.Seq
                        out6_: _dafny.Seq
                        out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_6_narrowThreshold_, eosToken)
                        d_12_next_ = out6_
                        if (d_12_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_13_ag_: _dafny.Seq
                            d_14_ai_: bool
                            d_15_ac_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                            d_13_ag_ = out7_
                            d_14_ai_ = out8_
                            d_15_ac_ = out9_
                            generated = d_13_ag_
                            insideConstrainedOut = d_14_ai_
                            currentConstrainedOut = d_15_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a complete, valid SQL query using only the schema tables and columns. Output the full SQL query including SELECT, FROM, WHERE/JOIN clauses as needed. Do not repeat tokens.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if not(insideConstrainedOut):
            insideConstrainedOut = True
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        d_2_closeReserve_: int
        d_2_closeReserve_ = 50
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and (((d_1_steps_) + (d_2_closeReserve_)) < (maxSteps)):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        d_3_cg_: _dafny.Seq
                        d_4_ci_: bool
                        d_5_cc_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_3_cg_ = out0_
                        d_4_ci_ = out1_
                        d_5_cc_ = out2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        generated = d_3_cg_
                        insideConstrainedOut = d_4_ci_
                        currentConstrainedOut = d_5_cc_
                        raise _dafny.Break("0")
                    elif True:
                        d_6_constrainedPrompt_: _dafny.Seq
                        d_6_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_7_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_6_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                        d_7_next_ = out3_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_7_next_) == (eosToken):
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            raise _dafny.Break("0")
                        elif True:
                            d_8_ag_: _dafny.Seq
                            d_9_ai_: bool
                            d_10_ac_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_next_)
                            d_8_ag_ = out4_
                            d_9_ai_ = out5_
                            d_10_ac_ = out6_
                            generated = d_8_ag_
                            insideConstrainedOut = d_9_ai_
                            currentConstrainedOut = d_10_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_11_closeBudget_: int
            d_11_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_12_cg_: _dafny.Seq
            d_13_ci_: bool
            d_14_cc_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_closeBudget_)
            d_12_cg_ = out7_
            d_13_ci_ = out8_
            d_14_cc_ = out9_
            generated = d_12_cg_
            insideConstrainedOut = d_13_ci_
            currentConstrainedOut = d_14_cc_
            d_1_steps_ = maxSteps
        if insideConstrainedOut:
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


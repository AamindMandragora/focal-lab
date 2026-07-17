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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single valid SQL SELECT statement. Use only the schema tables and columns. Output concise SQL with no trailing punctuation or repeated characters. Stop as soon as the query is complete.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if not(insideConstrainedOut):
            insideConstrainedOut = True
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
            d_2_cg_: _dafny.Seq
            d_3_ci_: bool
            d_4_cc_: _dafny.Seq
            d_5_closed_: bool
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out3_: bool
            out0_, out1_, out2_, out3_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
            d_2_cg_ = out0_
            d_3_ci_ = out1_
            d_4_cc_ = out2_
            d_5_closed_ = out3_
            d_1_steps_ = (d_1_steps_) + (1)
            if d_5_closed_:
                generated = d_2_cg_
                insideConstrainedOut = d_3_ci_
                currentConstrainedOut = d_4_cc_
            elif True:
                d_6_constrainedPrompt_: _dafny.Seq
                d_6_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_7_next_: _dafny.Seq
                out4_: _dafny.Seq
                out4_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_6_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                d_7_next_ = out4_
                if (d_7_next_) == (eosToken):
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                elif True:
                    d_8_ag_: _dafny.Seq
                    d_9_ai_: bool
                    d_10_ac_: _dafny.Seq
                    out5_: _dafny.Seq
                    out6_: bool
                    out7_: _dafny.Seq
                    out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_next_)
                    d_8_ag_ = out5_
                    d_9_ai_ = out6_
                    d_10_ac_ = out7_
                    generated = d_8_ag_
                    insideConstrainedOut = d_9_ai_
                    currentConstrainedOut = d_10_ac_
        if insideConstrainedOut:
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


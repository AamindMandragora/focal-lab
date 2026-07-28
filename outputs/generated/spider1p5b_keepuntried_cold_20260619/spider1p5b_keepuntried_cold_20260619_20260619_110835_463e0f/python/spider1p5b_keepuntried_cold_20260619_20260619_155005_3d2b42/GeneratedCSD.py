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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<your SQL query here>>. Use only tables and columns from the provided schema. No explanation, no markdown. Keep the SQL concise and correct.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            generated = out0_
            insideConstrainedOut = out1_
            currentConstrainedOut = out2_
            d_1_steps_ = (d_1_steps_) + (1)
        d_2_badTokens_: _dafny.Seq
        d_2_badTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))])
        d_3_spanTokens_: int
        d_3_spanTokens_ = 0
        d_4_maxSpanTokens_: int
        d_4_maxSpanTokens_ = 300
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_5_cg_: _dafny.Seq
                    d_6_ci_: bool
                    d_7_cc_: _dafny.Seq
                    d_8_closed_: bool
                    out3_: _dafny.Seq
                    out4_: bool
                    out5_: _dafny.Seq
                    out6_: bool
                    out3_, out4_, out5_, out6_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_5_cg_ = out3_
                    d_6_ci_ = out4_
                    d_7_cc_ = out5_
                    d_8_closed_ = out6_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_8_closed_:
                        generated = d_5_cg_
                        insideConstrainedOut = d_6_ci_
                        currentConstrainedOut = d_7_cc_
                        raise _dafny.Break("0")
                    if (d_1_steps_) >= (maxSteps):
                        raise _dafny.Break("0")
                    if (d_3_spanTokens_) >= (d_4_maxSpanTokens_):
                        d_9_constrainedPrompt2_: _dafny.Seq
                        d_9_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_10_next2_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_9_constrainedPrompt2_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_10_next2_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_10_next2_) == (eosToken):
                            raise _dafny.Break("0")
                        d_11_ag2_: _dafny.Seq
                        d_12_ai2_: bool
                        d_13_ac2_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next2_)
                        d_11_ag2_ = out8_
                        d_12_ai2_ = out9_
                        d_13_ac2_ = out10_
                        generated = d_11_ag2_
                        insideConstrainedOut = d_12_ai2_
                        currentConstrainedOut = d_13_ac2_
                        d_3_spanTokens_ = (d_3_spanTokens_) + (1)
                    elif True:
                        d_14_constrainedPrompt_: _dafny.Seq
                        d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_15_next_: _dafny.Seq
                        out11_: _dafny.Seq
                        out11_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, d_2_badTokens_, _dafny.BigRational('8e0'), eosToken)
                        d_15_next_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_15_next_) == (eosToken):
                            raise _dafny.Break("0")
                        d_16_ag_: _dafny.Seq
                        d_17_ai_: bool
                        d_18_ac_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                        d_16_ag_ = out12_
                        d_17_ai_ = out13_
                        d_18_ac_ = out14_
                        generated = d_16_ag_
                        insideConstrainedOut = d_17_ai_
                        currentConstrainedOut = d_18_ac_
                        d_3_spanTokens_ = (d_3_spanTokens_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


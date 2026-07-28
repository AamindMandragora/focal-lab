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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are a SQL expert. Output exactly one line: SQL: followed by a complete, syntactically valid SQL query answering the question. Use only the tables and columns from the schema. Include SELECT, FROM, and any necessary WHERE/JOIN/GROUP BY clauses.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeSteps_: int
        d_2_freeSteps_ = 10
        if (d_2_freeSteps_) > (maxSteps):
            d_2_freeSteps_ = maxSteps
        with _dafny.label("0"):
            while (d_1_steps_) < (d_2_freeSteps_):
                with _dafny.c_label("0"):
                    d_3_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_3_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_3_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_4_remainingBudget_: int
            d_4_remainingBudget_ = (maxSteps) - (d_1_steps_)
            d_5_constrainedGenerated_: _dafny.Seq
            d_6_terminatedByEos_: bool
            out1_: _dafny.Seq
            out2_: bool
            out1_, out2_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, (prompt) + (generated), d_4_remainingBudget_, eosToken)
            d_5_constrainedGenerated_ = out1_
            d_6_terminatedByEos_ = out2_
            generated = (generated) + (d_5_constrainedGenerated_)
            d_1_steps_ = maxSteps
        elif insideConstrainedOut:
            with _dafny.label("3_0_0"):
                while (d_1_steps_) < (maxSteps):
                    with _dafny.c_label("3_0_0"):
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_7_cg_: _dafny.Seq
                            d_8_ci_: bool
                            d_9_cc_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_cg_ = out3_
                            d_8_ci_ = out4_
                            d_9_cc_ = out5_
                            generated = d_7_cg_
                            insideConstrainedOut = d_8_ci_
                            currentConstrainedOut = d_9_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("3_0_0")
                        elif True:
                            d_10_stableLen_: int
                            d_10_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
                            d_11_constrainedPrompt_: _dafny.Seq
                            d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:d_10_stableLen_:]))
                            d_12_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_12_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                raise _dafny.Break("3_0_0")
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


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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write a simple, direct SQL query using exact table and column names from the schema. Do not use column aliases (no AS keyword for columns). Use simple JOINs, WHERE clauses, GROUP BY, ORDER BY, LIMIT. For 'both X and Y' questions use INTERSECT. For 'either X or Y' questions use UNION.")))
        if (insideConstrainedOut) and ((maxSteps) > (0)):
            d_1_closeBudget_: int
            d_1_closeBudget_ = maxSteps
            d_2_cg_: _dafny.Seq
            d_3_ci_: bool
            d_4_cc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_1_closeBudget_)
            d_2_cg_ = out0_
            d_3_ci_ = out1_
            d_4_cc_ = out2_
            generated = d_2_cg_
            insideConstrainedOut = d_3_ci_
            currentConstrainedOut = d_4_cc_
            cost = maxSteps
        elif (not(insideConstrainedOut)) and ((maxSteps) > (0)):
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            insideConstrainedOut = False
            d_5_constrainedPrompt_: _dafny.Seq
            d_5_constrainedPrompt_ = prompt
            d_6_steps_: int
            d_6_steps_ = 0
            with _dafny.label("1_0_0"):
                while (d_6_steps_) < (maxSteps):
                    with _dafny.c_label("1_0_0"):
                        d_7_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_5_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('3e-1'), eosToken)
                        d_7_next_ = out3_
                        d_6_steps_ = (d_6_steps_) + (1)
                        if (d_7_next_) == (eosToken):
                            raise _dafny.Break("1_0_0")
                        elif True:
                            d_8_isComplete_: bool
                            d_8_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_8_isComplete_:
                                raise _dafny.Break("1_0_0")
                            elif True:
                                d_9_g2_: _dafny.Seq
                                d_10_i2_: bool
                                d_11_c2_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_next_)
                                d_9_g2_ = out4_
                                d_10_i2_ = out5_
                                d_11_c2_ = out6_
                                generated = d_9_g2_
                                insideConstrainedOut = d_10_i2_
                                currentConstrainedOut = d_11_c2_
                        pass
                pass
            cost = d_6_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


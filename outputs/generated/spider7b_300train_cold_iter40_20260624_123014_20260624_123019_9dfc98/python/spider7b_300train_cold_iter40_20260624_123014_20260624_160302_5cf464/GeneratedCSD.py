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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a complete SQL query for the question using the schema. Use WHERE col NOT IN (SELECT ...) for exclusion/negation questions. Use GROUP BY for aggregate questions (max, count per group). Write full table names without aliases. Do not use AS keyword. Join tables with JOIN...ON. Output only the SQL query.")))
        if not(insideConstrainedOut):
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
            generated = out0_
            insideConstrainedOut = out1_
            currentConstrainedOut = out2_
        if (d_1_steps_) < (maxSteps):
            d_2_rem_: int
            d_2_rem_ = (maxSteps) - (d_1_steps_)
            d_3_fillBudget_: int
            d_3_fillBudget_ = _dafny.euclidian_division((d_2_rem_) * (17), 20)
            d_4_loopSteps_: int
            d_4_loopSteps_ = 0
            with _dafny.label("1_0"):
                while (d_4_loopSteps_) < (d_3_fillBudget_):
                    with _dafny.c_label("1_0"):
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            raise _dafny.Break("1_0")
                        d_5_stable_: _dafny.Seq
                        d_5_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_6_constrainedPrompt_: _dafny.Seq
                        d_6_constrainedPrompt_ = (prompt) + (d_5_stable_)
                        d_7_next_: _dafny.Seq
                        d_8_wasConstrained_: bool
                        out3_: _dafny.Seq
                        out4_: bool
                        out3_, out4_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_6_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_7_next_ = out3_
                        d_8_wasConstrained_ = out4_
                        d_4_loopSteps_ = (d_4_loopSteps_) + (1)
                        if (d_7_next_) == (eosToken):
                            raise _dafny.Break("1_0")
                        d_9_ng_: _dafny.Seq
                        d_10_ni_: bool
                        d_11_nc_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_next_)
                        d_9_ng_ = out5_
                        d_10_ni_ = out6_
                        d_11_nc_ = out7_
                        generated = d_9_ng_
                        insideConstrainedOut = d_10_ni_
                        currentConstrainedOut = d_11_nc_
                        pass
                pass
            d_1_steps_ = (d_1_steps_) + (d_3_fillBudget_)
        if (d_1_steps_) < (maxSteps):
            d_12_closeBudget_: int
            d_12_closeBudget_ = (maxSteps) - (d_1_steps_)
            out8_: _dafny.Seq
            out9_: bool
            out10_: _dafny.Seq
            out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_12_closeBudget_)
            generated = out8_
            insideConstrainedOut = out9_
            currentConstrainedOut = out10_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


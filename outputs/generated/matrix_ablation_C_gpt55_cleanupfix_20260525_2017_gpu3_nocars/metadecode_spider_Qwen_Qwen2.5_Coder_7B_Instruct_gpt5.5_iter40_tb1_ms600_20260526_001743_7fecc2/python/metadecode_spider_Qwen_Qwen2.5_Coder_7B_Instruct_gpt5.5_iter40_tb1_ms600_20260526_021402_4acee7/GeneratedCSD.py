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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Return exactly one answer in the form SQL: <<query>> and nothing else. Generate a single SQLite query for the given Spider schema and question. Use only schema table and column names. Prefer Spider canonical SQL: lowercase keywords, explicit joins, minimal aliases unless needed, no trailing semicolon. The decoder will force the visible SQL: << prefix; continue with the SQL query content and close only after all required filters, joins, grouping, ordering, limits, set operations, and subqueries are included.")))
        if (maxSteps) == (0):
            cost = 0
        elif (maxSteps) == (1):
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:"))]))
            cost = 1
        elif (maxSteps) == (2):
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))]))
            cost = 2
        elif True:
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))]))
            d_1_openedGenerated_: _dafny.Seq
            d_2_openedInside_: bool
            d_3_openedCurrent_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_1_openedGenerated_ = out0_
            d_2_openedInside_ = out1_
            d_3_openedCurrent_ = out2_
            generated = d_1_openedGenerated_
            insideConstrainedOut = d_2_openedInside_
            currentConstrainedOut = d_3_openedCurrent_
            d_4_steps_: int
            d_4_steps_ = 3
            d_5_hitEos_: bool
            d_5_hitEos_ = False
            d_6_extensionSteps_: int
            d_6_extensionSteps_ = 0
            d_7_extensionLimit_: int
            d_7_extensionLimit_ = 80
            while ((((((d_4_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut)) and ((len(currentConstrainedOut)) < (420))) and (not(d_5_hitEos_))) and ((not((parser).IsCompletePrefix(currentConstrainedOut))) or ((d_6_extensionSteps_) < (d_7_extensionLimit_))):
                d_8_stablePrefix_: _dafny.Seq
                d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_9_constrainedPrompt_: _dafny.Seq
                d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                if (parser).IsCompletePrefix(currentConstrainedOut):
                    d_10_nextComplete_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('8e0'), 18, eosToken)
                    d_10_nextComplete_ = out3_
                    d_4_steps_ = (d_4_steps_) + (1)
                    if (d_10_nextComplete_) == (eosToken):
                        d_5_hitEos_ = True
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_nextComplete_]))
                        currentConstrainedOut = (currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([d_10_nextComplete_]))
                        insideConstrainedOut = True
                        d_6_extensionSteps_ = (d_6_extensionSteps_) + (1)
                elif True:
                    d_11_nextIncomplete_: _dafny.Seq
                    out4_: _dafny.Seq
                    out4_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('12e0'), 16, eosToken)
                    d_11_nextIncomplete_ = out4_
                    d_4_steps_ = (d_4_steps_) + (1)
                    if (d_11_nextIncomplete_) == (eosToken):
                        d_5_hitEos_ = True
                    elif True:
                        d_12_appendedGenerated_: _dafny.Seq
                        d_13_appendedInside_: bool
                        d_14_appendedCurrent_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_nextIncomplete_)
                        d_12_appendedGenerated_ = out5_
                        d_13_appendedInside_ = out6_
                        d_14_appendedCurrent_ = out7_
                        generated = d_12_appendedGenerated_
                        insideConstrainedOut = d_13_appendedInside_
                        currentConstrainedOut = d_14_appendedCurrent_
                        d_6_extensionSteps_ = 0
            if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_4_steps_) < (maxSteps)):
                d_15_closedGenerated_: _dafny.Seq
                d_16_closedInside_: bool
                d_17_closedCurrent_: _dafny.Seq
                out8_: _dafny.Seq
                out9_: bool
                out10_: _dafny.Seq
                out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_15_closedGenerated_ = out8_
                d_16_closedInside_ = out9_
                d_17_closedCurrent_ = out10_
                generated = d_15_closedGenerated_
                insideConstrainedOut = d_16_closedInside_
                currentConstrainedOut = d_17_closedCurrent_
                d_4_steps_ = (d_4_steps_) + (1)
            cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


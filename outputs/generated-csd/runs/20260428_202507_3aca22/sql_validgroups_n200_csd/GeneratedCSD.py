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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_activeGroupIdx_: int
        d_2_activeGroupIdx_ = -1
        d_3_lastClauseToken_: _dafny.Seq
        d_3_lastClauseToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_activeGroupIdx_ = -1
                                d_3_lastClauseToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                    elif True:
                        d_5_complete_: bool
                        d_5_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_complete_:
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out1_
                            d_7_closedInside_ = out2_
                            d_8_closedCurrent_ = out3_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_2_activeGroupIdx_ = -1
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_stablePrefix_: _dafny.Seq
                            d_9_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_10_constrainedPrompt_: _dafny.Seq
                            d_10_constrainedPrompt_ = (prompt) + (d_9_stablePrefix_)
                            d_11_tokBeforeComma_: _dafny.Seq
                            d_12_foundBeforeComma_: bool
                            out4_: _dafny.Seq
                            out5_: bool
                            out4_, out5_ = (d_0_helpers_).LastTokenBefore(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")))
                            d_11_tokBeforeComma_ = out4_
                            d_12_foundBeforeComma_ = out5_
                            if d_12_foundBeforeComma_:
                                d_3_lastClauseToken_ = d_11_tokBeforeComma_
                            (lm).GenerateLogits((d_10_constrainedPrompt_) + (currentConstrainedOut))
                            if (len(validTokenGroups)) > (0):
                                d_13_flatPreferred_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                d_13_flatPreferred_ = out6_
                                if (len(d_13_flatPreferred_)) > (0):
                                    d_14_anyValidPreferred_: bool
                                    out7_: bool
                                    out7_ = (d_0_helpers_).MaskValidNextInGroup(lm, parser, currentConstrainedOut, d_13_flatPreferred_, eosToken)
                                    d_14_anyValidPreferred_ = out7_
                                if ((0) <= (d_2_activeGroupIdx_)) and ((d_2_activeGroupIdx_) < (len(validTokenGroups))):
                                    d_15_activeGroup_: _dafny.Seq
                                    d_15_activeGroup_ = (validTokenGroups)[d_2_activeGroupIdx_]
                                    if (len(d_15_activeGroup_)) > (0):
                                        d_16_anyValidActive_: bool
                                        out8_: bool
                                        out8_ = (d_0_helpers_).MaskValidNextInGroup(lm, parser, currentConstrainedOut, d_15_activeGroup_, eosToken)
                                        d_16_anyValidActive_ = out8_
                                if (d_3_lastClauseToken_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))):
                                    d_17_clauseIdx_: int
                                    out9_: int
                                    out9_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_3_lastClauseToken_)
                                    d_17_clauseIdx_ = out9_
                                    if ((0) <= (d_17_clauseIdx_)) and ((d_17_clauseIdx_) < (len(validTokenGroups))):
                                        d_18_clauseGroup_: _dafny.Seq
                                        d_18_clauseGroup_ = (validTokenGroups)[d_17_clauseIdx_]
                                        if (len(d_18_clauseGroup_)) > (0):
                                            d_19_anyValidClause_: bool
                                            out10_: bool
                                            out10_ = (d_0_helpers_).MaskValidNextInGroup(lm, parser, currentConstrainedOut, d_18_clauseGroup_, eosToken)
                                            d_19_anyValidClause_ = out10_
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_20_chosen_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (lm).ChooseNextToken()
                            d_20_chosen_ = out11_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_20_chosen_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_21_appendedGenerated_: _dafny.Seq
                                d_22_appendedInside_: bool
                                d_23_appendedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_chosen_)
                                d_21_appendedGenerated_ = out12_
                                d_22_appendedInside_ = out13_
                                d_23_appendedCurrent_ = out14_
                                generated = d_21_appendedGenerated_
                                insideConstrainedOut = d_22_appendedInside_
                                currentConstrainedOut = d_23_appendedCurrent_
                                out15_: int
                                out15_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_20_chosen_)
                                d_2_activeGroupIdx_ = out15_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


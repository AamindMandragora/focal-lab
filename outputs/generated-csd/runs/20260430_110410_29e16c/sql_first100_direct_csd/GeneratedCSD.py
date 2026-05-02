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
        d_2_activeGroup_: int
        d_2_activeGroup_ = -1
        d_3_scopeKeyword_: _dafny.Seq
        d_3_scopeKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_chunkBudget_: int
                        d_4_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_5_chunkedGenerated_: _dafny.Seq
                        d_6_stoppedOnOpenSpan_: bool
                        d_7_stoppedOnEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_chunkedGenerated_ = out0_
                        d_6_stoppedOnOpenSpan_ = out1_
                        d_7_stoppedOnEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        generated = d_5_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        if d_7_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_6_stoppedOnOpenSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_2_activeGroup_ = -1
                    elif True:
                        d_9_completeNow_: bool
                        d_9_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_9_completeNow_:
                            d_10_closedGenerated_: _dafny.Seq
                            d_11_closedInside_: bool
                            d_12_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_closedGenerated_ = out4_
                            d_11_closedInside_ = out5_
                            d_12_closedCurrent_ = out6_
                            generated = d_10_closedGenerated_
                            insideConstrainedOut = d_11_closedInside_
                            currentConstrainedOut = d_12_closedCurrent_
                            d_2_activeGroup_ = -1
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_13_stablePrefix_: _dafny.Seq
                            d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_14_constrainedPrompt_: _dafny.Seq
                            d_14_constrainedPrompt_ = (prompt) + (d_13_stablePrefix_)
                            d_15_semanticContext_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_3_scopeKeyword_)
                            d_15_semanticContext_ = out7_
                            (lm).GenerateLogits((d_14_constrainedPrompt_) + (currentConstrainedOut))
                            if (len(validTokenGroups)) > (0):
                                (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'))
                                if ((d_2_activeGroup_) >= (0)) and ((d_2_activeGroup_) < (len(validTokenGroups))):
                                    d_16_activeHasValid_: bool
                                    out8_: bool
                                    out8_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, (validTokenGroups)[d_2_activeGroup_])
                                    d_16_activeHasValid_ = out8_
                                    if d_16_activeHasValid_:
                                        (d_0_helpers_).BoostTokenLogits(lm, (validTokenGroups)[d_2_activeGroup_], _dafny.BigRational('8e0'))
                                if (len(d_15_semanticContext_)) > (0):
                                    d_17_flatPreferred_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out9_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_17_flatPreferred_ = out9_
                                    d_18_focused_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_17_flatPreferred_, d_15_semanticContext_)
                                    d_18_focused_ = out10_
                                    if (len(d_18_focused_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_18_focused_, _dafny.BigRational('4e0'))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_19_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (lm).ChooseNextToken()
                            d_19_next_ = out11_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_19_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_20_appendedGenerated_: _dafny.Seq
                                d_21_appendedInside_: bool
                                d_22_appendedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                d_20_appendedGenerated_ = out12_
                                d_21_appendedInside_ = out13_
                                d_22_appendedCurrent_ = out14_
                                generated = d_20_appendedGenerated_
                                insideConstrainedOut = d_21_appendedInside_
                                currentConstrainedOut = d_22_appendedCurrent_
                                d_23_groupIdx_: int
                                out15_: int
                                out15_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_19_next_)
                                d_23_groupIdx_ = out15_
                                if (d_23_groupIdx_) >= (0):
                                    d_2_activeGroup_ = d_23_groupIdx_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


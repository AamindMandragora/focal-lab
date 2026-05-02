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
        d_2_openSpanToken_: _dafny.Seq
        d_2_openSpanToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_chunkBudget_: int
                        d_3_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_4_chunkedGenerated_: _dafny.Seq
                        d_5_stoppedOnOpenSpan_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, d_2_openSpanToken_, eosToken)
                        d_4_chunkedGenerated_ = out0_
                        d_5_stoppedOnOpenSpan_ = out1_
                        d_6_stoppedOnEos_ = out2_
                        d_7_stepsUsed_ = out3_
                        generated = d_4_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        if d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_5_stoppedOnOpenSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_8_complete_: bool
                        d_8_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_complete_:
                            d_9_closedGenerated_: _dafny.Seq
                            d_10_closedInside_: bool
                            d_11_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated_ = out4_
                            d_10_closedInside_ = out5_
                            d_11_closedCurrent_ = out6_
                            generated = d_9_closedGenerated_
                            insideConstrainedOut = d_10_closedInside_
                            currentConstrainedOut = d_11_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_12_constrainedPrompt_: _dafny.Seq
                            d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            (lm).GenerateLogits((d_12_constrainedPrompt_) + (currentConstrainedOut))
                            if (len(validTokenGroups)) > (0):
                                d_13_topToken_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = (d_0_helpers_).GetHighestLogitToken(lm)
                                d_13_topToken_ = out7_
                                d_14_groupIdx_: int
                                out8_: int
                                out8_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_13_topToken_)
                                d_14_groupIdx_ = out8_
                                if (d_14_groupIdx_) >= (0):
                                    d_15_safeGroup_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out9_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets((validTokenGroups)[d_14_groupIdx_], (lm).Tokens)
                                    d_15_safeGroup_ = out9_
                                    (d_0_helpers_).BoostTokenLogits(lm, d_15_safeGroup_, _dafny.BigRational('2e0'))
                            if (len(currentConstrainedOut)) > (0):
                                d_16_lastTok_: _dafny.Seq
                                d_16_lastTok_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                                if (d_16_lastTok_) in ((lm).Tokens):
                                    (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_16_lastTok_]), _dafny.BigRational('8e0'))
                                d_17_recent_: _dafny.Seq
                                d_17_recent_ = currentConstrainedOut
                                if (4) < (len(d_17_recent_)):
                                    d_17_recent_ = _dafny.SeqWithoutIsStrInference((d_17_recent_)[(len(d_17_recent_)) - (4)::])
                                d_18_safeRecent_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_17_recent_, (lm).Tokens)
                                d_18_safeRecent_ = out10_
                                (d_0_helpers_).PenalizeTokenLogits(lm, d_18_safeRecent_, _dafny.BigRational('2e0'))
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
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


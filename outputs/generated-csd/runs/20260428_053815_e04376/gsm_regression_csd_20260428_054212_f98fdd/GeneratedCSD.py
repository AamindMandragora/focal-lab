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
        d_2_openSpan_: _dafny.Seq
        d_2_openSpan_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
        d_3_preferredFlat_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_3_preferredFlat_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        (lm).GenerateLogits((prompt) + (generated))
                        (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_2_openSpan_]), _dafny.BigRational('1e2'))
                        d_4_nextOutside_: _dafny.Seq
                        out1_: _dafny.Seq
                        out1_ = (d_0_helpers_).GetHighestLogitToken(lm)
                        d_4_nextOutside_ = out1_
                        d_5_groupIdx_: int
                        out2_: int
                        out2_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_4_nextOutside_)
                        d_5_groupIdx_ = out2_
                        d_6_stepped_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_6_stepped_ = out3_
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_stepped_]))
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_6_stepped_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if VerifiedDecoderAgent.default__.Contains(d_6_stepped_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_7_complete_: bool
                        d_7_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_complete_:
                            d_8_closedGenerated_: _dafny.Seq
                            d_9_closedInside_: bool
                            d_10_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_8_closedGenerated_ = out4_
                            d_9_closedInside_ = out5_
                            d_10_closedCurrent_ = out6_
                            generated = d_8_closedGenerated_
                            insideConstrainedOut = d_9_closedInside_
                            currentConstrainedOut = d_10_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_stablePrefix_: _dafny.Seq
                            d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_12_constrainedPrompt_: _dafny.Seq
                            d_12_constrainedPrompt_ = (prompt) + (d_11_stablePrefix_)
                            (lm).GenerateLogits((d_12_constrainedPrompt_) + (currentConstrainedOut))
                            if (len(d_3_preferredFlat_)) > (0):
                                d_13_candidates_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                d_13_candidates_ = out7_
                                d_14_preferred_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_13_candidates_, d_3_preferredFlat_)
                                d_14_preferred_ = out8_
                            d_15_topTok_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_15_topTok_ = out9_
                            d_16_idx_: int
                            out10_: int
                            out10_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_15_topTok_)
                            d_16_idx_ = out10_
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_17_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (lm).ChooseNextToken()
                            d_17_next_ = out11_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_17_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_18_appendedGenerated_: _dafny.Seq
                                d_19_appendedInside_: bool
                                d_20_appendedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                d_18_appendedGenerated_ = out12_
                                d_19_appendedInside_ = out13_
                                d_20_appendedCurrent_ = out14_
                                generated = d_18_appendedGenerated_
                                insideConstrainedOut = d_19_appendedInside_
                                currentConstrainedOut = d_20_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


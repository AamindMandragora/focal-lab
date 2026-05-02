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
        d_2_fromKeyword_: _dafny.Seq
        d_2_fromKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_semanticContext_: _dafny.Seq
        d_4_semanticContext_ = _dafny.SeqWithoutIsStrInference([])
        d_5_flatPreferred_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_5_flatPreferred_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_next_: _dafny.Seq
                        out1_: _dafny.Seq
                        out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_6_next_ = out1_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_6_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                            if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_4_semanticContext_ = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_7_closedGenerated_: _dafny.Seq
                            d_8_closedInside_: bool
                            d_9_closedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_closedGenerated_ = out2_
                            d_8_closedInside_ = out3_
                            d_9_closedCurrent_ = out4_
                            generated = d_7_closedGenerated_
                            insideConstrainedOut = d_8_closedInside_
                            currentConstrainedOut = d_9_closedCurrent_
                            d_4_semanticContext_ = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_stablePrefix_: _dafny.Seq
                            d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_11_constrainedPrompt_: _dafny.Seq
                            d_11_constrainedPrompt_ = (prompt) + (d_10_stablePrefix_)
                            out5_: _dafny.Seq
                            out5_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_2_fromKeyword_)
                            d_4_semanticContext_ = out5_
                            d_12_validCount_: int
                            out6_: int
                            out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_12_validCount_ = out6_
                            if (d_12_validCount_) <= (d_3_narrowThreshold_):
                                d_13_next_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_13_next_ = out7_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_13_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_14_appendedGenerated_: _dafny.Seq
                                    d_15_appendedInside_: bool
                                    d_16_appendedCurrent_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                    d_14_appendedGenerated_ = out8_
                                    d_15_appendedInside_ = out9_
                                    d_16_appendedCurrent_ = out10_
                                    generated = d_14_appendedGenerated_
                                    insideConstrainedOut = d_15_appendedInside_
                                    currentConstrainedOut = d_16_appendedCurrent_
                            elif True:
                                (lm).GenerateLogits((d_11_constrainedPrompt_) + (currentConstrainedOut))
                                d_17_candidates_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                d_17_candidates_ = out11_
                                if (len(d_4_semanticContext_)) > (0):
                                    d_18_focused_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_17_candidates_, d_4_semanticContext_)
                                    d_18_focused_ = out12_
                                    if (len(d_18_focused_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_18_focused_, _dafny.BigRational('8e0'))
                                if (len(d_5_flatPreferred_)) > (0):
                                    d_19_anyPreferredValid_: bool
                                    out13_: bool
                                    out13_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_5_flatPreferred_)
                                    d_19_anyPreferredValid_ = out13_
                                    if d_19_anyPreferredValid_:
                                        d_20_preferred_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out14_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_17_candidates_, d_5_flatPreferred_)
                                        d_20_preferred_ = out14_
                                        if (len(d_20_preferred_)) > (0):
                                            (d_0_helpers_).BoostTokenLogits(lm, d_20_preferred_, _dafny.BigRational('4e0'))
                                if VerifiedDecoderAgent.default__.Contains(d_2_fromKeyword_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    (d_0_helpers_).ScaleAllLogits(lm, _dafny.BigRational('1e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_21_next_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (lm).ChooseNextToken()
                                d_21_next_ = out15_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_21_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_22_appendedGenerated2_: _dafny.Seq
                                    d_23_appendedInside2_: bool
                                    d_24_appendedCurrent2_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                    d_22_appendedGenerated2_ = out16_
                                    d_23_appendedInside2_ = out17_
                                    d_24_appendedCurrent2_ = out18_
                                    generated = d_22_appendedGenerated2_
                                    insideConstrainedOut = d_23_appendedInside2_
                                    currentConstrainedOut = d_24_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


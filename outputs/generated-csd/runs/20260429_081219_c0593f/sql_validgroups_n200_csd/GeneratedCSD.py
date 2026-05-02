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
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            if VerifiedDecoderAgent.default__.Contains(d_3_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_4_isComplete_: bool
                        d_4_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_4_isComplete_:
                            d_5_closedGenerated_: _dafny.Seq
                            d_6_closedInside_: bool
                            d_7_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_5_closedGenerated_ = out1_
                            d_6_closedInside_ = out2_
                            d_7_closedCurrent_ = out3_
                            generated = d_5_closedGenerated_
                            insideConstrainedOut = d_6_closedInside_
                            currentConstrainedOut = d_7_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_stablePrefix_: _dafny.Seq
                            d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_9_constrainedPrompt_: _dafny.Seq
                            d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                            d_10_validCount_: int
                            out4_: int
                            out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_10_validCount_ = out4_
                            if (d_10_validCount_) <= (d_2_narrowThreshold_):
                                d_11_next_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_11_next_ = out5_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_11_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_12_appendedGenerated_: _dafny.Seq
                                    d_13_appendedInside_: bool
                                    d_14_appendedCurrent_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                                    d_12_appendedGenerated_ = out6_
                                    d_13_appendedInside_ = out7_
                                    d_14_appendedCurrent_ = out8_
                                    generated = d_12_appendedGenerated_
                                    insideConstrainedOut = d_13_appendedInside_
                                    currentConstrainedOut = d_14_appendedCurrent_
                            elif True:
                                d_15_tableContext_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))
                                d_15_tableContext_ = out9_
                                (lm).GenerateLogits((d_9_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(d_15_tableContext_)) > (0):
                                    d_16_candidates1_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                    d_16_candidates1_ = out10_
                                    d_17_inScope_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_16_candidates1_, d_15_tableContext_)
                                    d_17_inScope_ = out11_
                                    if (len(d_17_inScope_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_17_inScope_, _dafny.BigRational('6e0'))
                                if (len(validTokenGroups)) > (0):
                                    d_18_flatPreferred_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_18_flatPreferred_ = out12_
                                    if (len(d_18_flatPreferred_)) > (0):
                                        d_19_anyPreferredValid_: bool
                                        out13_: bool
                                        out13_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_18_flatPreferred_)
                                        d_19_anyPreferredValid_ = out13_
                                        if d_19_anyPreferredValid_:
                                            d_20_candidates2_: _dafny.Seq
                                            out14_: _dafny.Seq
                                            out14_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                            d_20_candidates2_ = out14_
                                            d_21_preferred_: _dafny.Seq
                                            out15_: _dafny.Seq
                                            out15_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_20_candidates2_, d_18_flatPreferred_)
                                            d_21_preferred_ = out15_
                                            if (len(d_21_preferred_)) > (0):
                                                (d_0_helpers_).BoostTokenLogits(lm, d_21_preferred_, _dafny.BigRational('3e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_22_next_: _dafny.Seq
                                out16_: _dafny.Seq
                                out16_ = (lm).ChooseNextToken()
                                d_22_next_ = out16_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_22_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_23_appendedGenerated2_: _dafny.Seq
                                    d_24_appendedInside2_: bool
                                    d_25_appendedCurrent2_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                    d_23_appendedGenerated2_ = out17_
                                    d_24_appendedInside2_ = out18_
                                    d_25_appendedCurrent2_ = out19_
                                    generated = d_23_appendedGenerated2_
                                    insideConstrainedOut = d_24_appendedInside2_
                                    currentConstrainedOut = d_25_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


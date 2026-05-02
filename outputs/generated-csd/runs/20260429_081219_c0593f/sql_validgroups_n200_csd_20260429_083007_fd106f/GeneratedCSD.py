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
        d_3_schemaFocus_: _dafny.Seq
        d_3_schemaFocus_ = _dafny.SeqWithoutIsStrInference([])
        d_4_flatPreferred_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_4_flatPreferred_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_next_: _dafny.Seq
                        out1_: _dafny.Seq
                        out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out1_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            if VerifiedDecoderAgent.default__.Contains(d_5_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_3_schemaFocus_ = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out2_
                            d_7_closedInside_ = out3_
                            d_8_closedCurrent_ = out4_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_3_schemaFocus_ = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_stablePrefix_: _dafny.Seq
                            d_9_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_10_constrainedPrompt_: _dafny.Seq
                            d_10_constrainedPrompt_ = (prompt) + (d_9_stablePrefix_)
                            out5_: _dafny.Seq
                            out5_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))
                            d_3_schemaFocus_ = out5_
                            d_11_validCount_: int
                            out6_: int
                            out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_11_validCount_ = out6_
                            if (d_11_validCount_) > (d_2_narrowThreshold_):
                                d_12_budget_: int
                                d_12_budget_ = stepTokenBudget
                                if (d_12_budget_) > ((maxSteps) - (d_1_steps_)):
                                    d_12_budget_ = (maxSteps) - (d_1_steps_)
                                if (d_12_budget_) > (0):
                                    d_13_symbolOut_: _dafny.Seq
                                    d_14_hitEos_: bool
                                    d_15_stepsUsed_: int
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: int
                                    out7_, out8_, out9_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, d_12_budget_, eosToken)
                                    d_13_symbolOut_ = out7_
                                    d_14_hitEos_ = out8_
                                    d_15_stepsUsed_ = out9_
                                    generated = (d_9_stablePrefix_) + (d_13_symbolOut_)
                                    currentConstrainedOut = d_13_symbolOut_
                                    d_1_steps_ = (d_1_steps_) + (d_15_stepsUsed_)
                                    if d_14_hitEos_:
                                        raise _dafny.Break("0")
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                (lm).GenerateLogits((d_10_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(d_3_schemaFocus_)) > (0):
                                    d_16_candidates1_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                    d_16_candidates1_ = out10_
                                    d_17_focused_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_16_candidates1_, d_3_schemaFocus_)
                                    d_17_focused_ = out11_
                                    if (len(d_17_focused_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_17_focused_, _dafny.BigRational('6e0'))
                                if (len(d_4_flatPreferred_)) > (0):
                                    d_18_anyPreferredValid_: bool
                                    out12_: bool
                                    out12_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_4_flatPreferred_)
                                    d_18_anyPreferredValid_ = out12_
                                    if d_18_anyPreferredValid_:
                                        d_19_candidates2_: _dafny.Seq
                                        out13_: _dafny.Seq
                                        out13_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                        d_19_candidates2_ = out13_
                                        d_20_preferred_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out14_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_19_candidates2_, d_4_flatPreferred_)
                                        d_20_preferred_ = out14_
                                        if (len(d_20_preferred_)) > (0):
                                            (d_0_helpers_).BoostTokenLogits(lm, d_20_preferred_, _dafny.BigRational('4e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_21_next2_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (lm).ChooseNextToken()
                                d_21_next2_ = out15_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_21_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_22_appendedGenerated_: _dafny.Seq
                                    d_23_appendedInside_: bool
                                    d_24_appendedCurrent_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next2_)
                                    d_22_appendedGenerated_ = out16_
                                    d_23_appendedInside_ = out17_
                                    d_24_appendedCurrent_ = out18_
                                    generated = d_22_appendedGenerated_
                                    insideConstrainedOut = d_23_appendedInside_
                                    currentConstrainedOut = d_24_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


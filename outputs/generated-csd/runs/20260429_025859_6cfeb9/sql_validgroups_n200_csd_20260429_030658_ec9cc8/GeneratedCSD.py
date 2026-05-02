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
        d_3_preferredFlat_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_3_preferredFlat_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_next_: _dafny.Seq
                        out1_: _dafny.Seq
                        out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out1_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_5_closedGenerated_: _dafny.Seq
                            d_6_closedInside_: bool
                            d_7_closedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_5_closedGenerated_ = out2_
                            d_6_closedInside_ = out3_
                            d_7_closedCurrent_ = out4_
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
                            out5_: int
                            out5_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_10_validCount_ = out5_
                            d_11_useTight_: bool
                            d_11_useTight_ = (d_10_validCount_) <= (d_2_narrowThreshold_)
                            if (not(d_11_useTight_)) and ((len(d_3_preferredFlat_)) > (0)):
                                d_12_anyPreferredValid_: bool
                                out6_: bool
                                out6_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_3_preferredFlat_)
                                d_12_anyPreferredValid_ = out6_
                                if d_12_anyPreferredValid_:
                                    d_11_useTight_ = True
                            if d_11_useTight_:
                                (lm).GenerateLogits((d_9_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(d_3_preferredFlat_)) > (0):
                                    d_13_candidates_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out7_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                    d_13_candidates_ = out7_
                                    d_14_preferredCandidates_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out8_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_13_candidates_, d_3_preferredFlat_)
                                    d_14_preferredCandidates_ = out8_
                                    if (len(d_14_preferredCandidates_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_14_preferredCandidates_, _dafny.BigRational('6e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_15_next_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (lm).ChooseNextToken()
                                d_15_next_ = out9_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_15_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_16_appendedGenerated_: _dafny.Seq
                                    d_17_appendedInside_: bool
                                    d_18_appendedCurrent_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                    d_16_appendedGenerated_ = out10_
                                    d_17_appendedInside_ = out11_
                                    d_18_appendedCurrent_ = out12_
                                    generated = d_16_appendedGenerated_
                                    insideConstrainedOut = d_17_appendedInside_
                                    currentConstrainedOut = d_18_appendedCurrent_
                            elif True:
                                d_19_remaining_: int
                                d_19_remaining_ = (maxSteps) - (d_1_steps_)
                                d_20_budget_: int
                                if (stepTokenBudget) < (d_19_remaining_):
                                    d_20_budget_ = stepTokenBudget
                                elif True:
                                    d_20_budget_ = d_19_remaining_
                                if (d_20_budget_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_21_symbolOut_: _dafny.Seq
                                    d_22_hitEos_: bool
                                    d_23_stepsUsed_: int
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: int
                                    out13_, out14_, out15_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, d_20_budget_, eosToken)
                                    d_21_symbolOut_ = out13_
                                    d_22_hitEos_ = out14_
                                    d_23_stepsUsed_ = out15_
                                    generated = (d_8_stablePrefix_) + (d_21_symbolOut_)
                                    currentConstrainedOut = d_21_symbolOut_
                                    d_1_steps_ = (d_1_steps_) + (d_23_stepsUsed_)
                                    if d_22_hitEos_:
                                        raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


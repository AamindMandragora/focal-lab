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
        d_2_narrowThreshold_ = 8
        d_3_flatPreferred_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_3_flatPreferred_ = out0_
        d_4_done_: bool
        d_4_done_ = False
        while ((d_1_steps_) < (maxSteps)) and (not(d_4_done_)):
            if not(insideConstrainedOut):
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                insideConstrainedOut = True
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                if (parser).IsCompletePrefix(currentConstrainedOut):
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
                    d_4_done_ = True
                elif True:
                    d_8_constrainedPrompt_: _dafny.Seq
                    d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_9_validCount_: int
                    out4_: int
                    out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                    d_9_validCount_ = out4_
                    if ((d_9_validCount_) <= (d_2_narrowThreshold_)) or ((stepTokenBudget) == (0)):
                        d_10_next_: _dafny.Seq
                        out5_: _dafny.Seq
                        out5_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_10_next_ = out5_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_10_next_) == (eosToken):
                            d_4_done_ = True
                        elif True:
                            d_11_appendedGenerated_: _dafny.Seq
                            d_12_appendedInside_: bool
                            d_13_appendedCurrent_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                            d_11_appendedGenerated_ = out6_
                            d_12_appendedInside_ = out7_
                            d_13_appendedCurrent_ = out8_
                            generated = d_11_appendedGenerated_
                            insideConstrainedOut = d_12_appendedInside_
                            currentConstrainedOut = d_13_appendedCurrent_
                    elif True:
                        (lm).GenerateLogits((d_8_constrainedPrompt_) + (currentConstrainedOut))
                        if (len(d_3_flatPreferred_)) > (0):
                            d_14_candidates_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                            d_14_candidates_ = out9_
                            d_15_preferred_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_14_candidates_, d_3_flatPreferred_)
                            d_15_preferred_ = out10_
                            if (len(d_15_preferred_)) > (0):
                                (d_0_helpers_).BoostTokenLogits(lm, d_15_preferred_, _dafny.BigRational('3e0'))
                        d_16_stablePrefix_: _dafny.Seq
                        d_16_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_17_remainingBudget_: int
                        d_17_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        d_18_localBudget_: int
                        d_18_localBudget_ = stepTokenBudget
                        if (d_17_remainingBudget_) < (d_18_localBudget_):
                            d_18_localBudget_ = d_17_remainingBudget_
                        d_19_symbolOut_: _dafny.Seq
                        d_20_hitEos_: bool
                        d_21_stepsUsed_: int
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: int
                        out11_, out12_, out13_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, d_18_localBudget_, eosToken)
                        d_19_symbolOut_ = out11_
                        d_20_hitEos_ = out12_
                        d_21_stepsUsed_ = out13_
                        generated = (d_16_stablePrefix_) + (d_19_symbolOut_)
                        currentConstrainedOut = d_19_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_21_stepsUsed_)
                        if d_20_hitEos_:
                            d_4_done_ = True
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


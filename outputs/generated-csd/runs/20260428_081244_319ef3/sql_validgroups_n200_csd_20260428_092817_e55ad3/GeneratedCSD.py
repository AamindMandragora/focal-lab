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
        d_2_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_flatGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_generated2_: _dafny.Seq
                        d_4_inside2_: bool
                        d_5_current2_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_3_generated2_ = out1_
                        d_4_inside2_ = out2_
                        d_5_current2_ = out3_
                        generated = d_3_generated2_
                        insideConstrainedOut = d_4_inside2_
                        currentConstrainedOut = d_5_current2_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_6_completeNow_: bool
                        d_6_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_6_completeNow_:
                            d_7_generated3_: _dafny.Seq
                            d_8_inside3_: bool
                            d_9_current3_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_generated3_ = out4_
                            d_8_inside3_ = out5_
                            d_9_current3_ = out6_
                            generated = d_7_generated3_
                            insideConstrainedOut = d_8_inside3_
                            currentConstrainedOut = d_9_current3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_10_deadEnd_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_10_deadEnd_ = out7_
                            if d_10_deadEnd_:
                                d_11_repaired_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_11_repaired_ = out8_
                                if (len(d_11_repaired_)) == (len(currentConstrainedOut)):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_12_stablePrefix_: _dafny.Seq
                                    d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    generated = (d_12_stablePrefix_) + (d_11_repaired_)
                                    currentConstrainedOut = d_11_repaired_
                            elif True:
                                d_13_stablePrefix2_: _dafny.Seq
                                d_13_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_14_constrainedPrompt_: _dafny.Seq
                                d_14_constrainedPrompt_ = (prompt) + (d_13_stablePrefix2_)
                                (lm).GenerateLogits((d_14_constrainedPrompt_) + (currentConstrainedOut))
                                d_15_candidates_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, 16, eosToken)
                                d_15_candidates_ = out9_
                                d_16_hinted_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_15_candidates_, d_2_flatGroups_)
                                d_16_hinted_ = out10_
                                if (len(d_16_hinted_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_16_hinted_, _dafny.BigRational('3e0'))
                                d_17_budget_: int
                                d_17_budget_ = stepTokenBudget
                                if (d_17_budget_) == (0):
                                    d_17_budget_ = 1
                                if ((maxSteps) - (d_1_steps_)) < (d_17_budget_):
                                    d_17_budget_ = (maxSteps) - (d_1_steps_)
                                d_18_currentOut_: _dafny.Seq
                                d_19_hitEos_: bool
                                d_20_stepsUsed_: int
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: int
                                out11_, out12_, out13_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, d_17_budget_, eosToken)
                                d_18_currentOut_ = out11_
                                d_19_hitEos_ = out12_
                                d_20_stepsUsed_ = out13_
                                d_1_steps_ = (d_1_steps_) + (d_20_stepsUsed_)
                                if d_19_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (d_13_stablePrefix2_) + (d_18_currentOut_)
                                    currentConstrainedOut = d_18_currentOut_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


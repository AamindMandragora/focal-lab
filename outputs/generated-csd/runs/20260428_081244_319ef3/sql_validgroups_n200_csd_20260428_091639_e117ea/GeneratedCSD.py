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
                        (lm).GenerateLogits((prompt) + (generated))
                        (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                        d_3_next_: _dafny.Seq
                        out1_: _dafny.Seq
                        out1_ = (lm).ChooseNextTokenUnconstrained()
                        d_3_next_ = out1_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if VerifiedDecoderAgent.default__.Contains(d_3_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                raise _dafny.Break("0")
                    elif True:
                        d_4_completeNow_: bool
                        d_4_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_4_completeNow_:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                            d_5_closeTok_: _dafny.Seq
                            out2_: _dafny.Seq
                            out2_ = (lm).ChooseNextTokenUnconstrained()
                            d_5_closeTok_ = out2_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_closeTok_]))
                            d_1_steps_ = (d_1_steps_) + (1)
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            raise _dafny.Break("0")
                        elif True:
                            d_6_deadEnd_: bool
                            out3_: bool
                            out3_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_6_deadEnd_ = out3_
                            if d_6_deadEnd_:
                                d_7_repaired_: _dafny.Seq
                                out4_: _dafny.Seq
                                out4_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_7_repaired_ = out4_
                                if (len(d_7_repaired_)) == (len(currentConstrainedOut)):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_8_stablePrefix_: _dafny.Seq
                                    d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    generated = (d_8_stablePrefix_) + (d_7_repaired_)
                                    currentConstrainedOut = d_7_repaired_
                            elif True:
                                d_9_stablePrefix2_: _dafny.Seq
                                d_9_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_10_constrainedPrompt_: _dafny.Seq
                                d_10_constrainedPrompt_ = (prompt) + (d_9_stablePrefix2_)
                                (lm).GenerateLogits((d_10_constrainedPrompt_) + (currentConstrainedOut))
                                d_11_candidates_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, 10, eosToken)
                                d_11_candidates_ = out5_
                                d_12_hinted_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_11_candidates_, d_2_flatGroups_)
                                d_12_hinted_ = out6_
                                if (len(d_12_hinted_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_12_hinted_, _dafny.BigRational('3e0'))
                                d_13_budget_: int
                                d_13_budget_ = stepTokenBudget
                                if (d_13_budget_) == (0):
                                    d_13_budget_ = 1
                                if ((maxSteps) - (d_1_steps_)) < (d_13_budget_):
                                    d_13_budget_ = (maxSteps) - (d_1_steps_)
                                d_14_currentOut_: _dafny.Seq
                                d_15_hitEos_: bool
                                d_16_stepsUsed_: int
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: int
                                out7_, out8_, out9_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, d_13_budget_, eosToken)
                                d_14_currentOut_ = out7_
                                d_15_hitEos_ = out8_
                                d_16_stepsUsed_ = out9_
                                d_1_steps_ = (d_1_steps_) + (d_16_stepsUsed_)
                                if d_15_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (d_9_stablePrefix2_) + (d_14_currentOut_)
                                    currentConstrainedOut = d_14_currentOut_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


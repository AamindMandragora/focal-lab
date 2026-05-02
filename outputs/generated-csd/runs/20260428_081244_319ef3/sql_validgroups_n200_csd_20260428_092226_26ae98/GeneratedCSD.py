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
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_3_completeNow_: bool
                        d_3_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_completeNow_:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_4_deadEnd_: bool
                            out1_: bool
                            out1_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_4_deadEnd_ = out1_
                            if d_4_deadEnd_:
                                d_5_repaired_: _dafny.Seq
                                out2_: _dafny.Seq
                                out2_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_5_repaired_ = out2_
                                if (len(d_5_repaired_)) == (len(currentConstrainedOut)):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_6_stablePrefix_: _dafny.Seq
                                    d_6_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    generated = (d_6_stablePrefix_) + (d_5_repaired_)
                                    currentConstrainedOut = d_5_repaired_
                            elif True:
                                d_7_stablePrefix2_: _dafny.Seq
                                d_7_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_8_constrainedPrompt_: _dafny.Seq
                                d_8_constrainedPrompt_ = (prompt) + (d_7_stablePrefix2_)
                                (lm).GenerateLogits((d_8_constrainedPrompt_) + (currentConstrainedOut))
                                d_9_candidates_: _dafny.Seq
                                out3_: _dafny.Seq
                                out3_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, 12, eosToken)
                                d_9_candidates_ = out3_
                                d_10_hinted_: _dafny.Seq
                                out4_: _dafny.Seq
                                out4_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_9_candidates_, d_2_flatGroups_)
                                d_10_hinted_ = out4_
                                if (len(d_10_hinted_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_10_hinted_, _dafny.BigRational('2e0'))
                                d_11_budget_: int
                                d_11_budget_ = stepTokenBudget
                                if (d_11_budget_) == (0):
                                    d_11_budget_ = 1
                                if ((maxSteps) - (d_1_steps_)) < (d_11_budget_):
                                    d_11_budget_ = (maxSteps) - (d_1_steps_)
                                d_12_currentOut_: _dafny.Seq
                                d_13_hitEos_: bool
                                d_14_stepsUsed_: int
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: int
                                out5_, out6_, out7_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, d_11_budget_, eosToken)
                                d_12_currentOut_ = out5_
                                d_13_hitEos_ = out6_
                                d_14_stepsUsed_ = out7_
                                d_1_steps_ = (d_1_steps_) + (d_14_stepsUsed_)
                                if d_13_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (d_7_stablePrefix2_) + (d_12_currentOut_)
                                    currentConstrainedOut = d_12_currentOut_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


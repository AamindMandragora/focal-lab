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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, eosToken):
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
        d_2_narrowThreshold_ = 10
        d_3_done_: bool
        d_3_done_ = False
        while ((d_1_steps_) < (maxSteps)) and (not(d_3_done_)):
            if not(insideConstrainedOut):
                (lm).GenerateLogits((prompt) + (generated))
                d_4_openCandidates_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets((lm).Tokens, (lm).Tokens)
                d_4_openCandidates_ = out0_
                (d_0_helpers_).BoostTokenLogits(lm, d_4_openCandidates_, _dafny.BigRational('0e0'))
                d_5_nextOpen_: _dafny.Seq
                out1_: _dafny.Seq
                out1_ = (lm).ChooseNextTokenUnconstrained()
                d_5_nextOpen_ = out1_
                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                if (d_5_nextOpen_) == (eosToken):
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_3_done_ = True
                elif True:
                    if VerifiedDecoderAgent.default__.Contains(d_5_nextOpen_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_nextOpen_]))
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        (lm).GenerateLogits((prompt) + (generated))
                        (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_5_nextOpen_]), _dafny.BigRational('1e2'))
                        d_6_forcedOpen_: _dafny.Seq
                        out2_: _dafny.Seq
                        out2_ = (lm).ChooseNextTokenUnconstrained()
                        d_6_forcedOpen_ = out2_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        if (d_6_forcedOpen_) == (eosToken):
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_done_ = True
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_forcedOpen_]))
                            d_1_steps_ = (d_1_steps_) + (1)
                            if VerifiedDecoderAgent.default__.Contains(d_6_forcedOpen_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                d_3_done_ = True
            elif True:
                d_7_isComplete_: bool
                d_7_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_7_isComplete_:
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    if (d_1_steps_) < (maxSteps):
                        (lm).GenerateLogits((prompt) + (generated))
                        d_8_closeCandidates_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets((lm).Tokens, (lm).Tokens)
                        d_8_closeCandidates_ = out3_
                        (d_0_helpers_).BoostTokenLogits(lm, d_8_closeCandidates_, _dafny.BigRational('0e0'))
                        d_9_nextClose_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (lm).ChooseNextTokenUnconstrained()
                        d_9_nextClose_ = out4_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        if (d_9_nextClose_) == (eosToken):
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_done_ = True
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_nextClose_]))
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_done_ = True
                    elif True:
                        d_3_done_ = True
                elif True:
                    d_10_stablePrefix_: _dafny.Seq
                    d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_11_constrainedPrompt_: _dafny.Seq
                    d_11_constrainedPrompt_ = (prompt) + (d_10_stablePrefix_)
                    d_12_validCount_: int
                    out5_: int
                    out5_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                    d_12_validCount_ = out5_
                    if (d_12_validCount_) <= (d_2_narrowThreshold_):
                        d_13_next_: _dafny.Seq
                        out6_: _dafny.Seq
                        out6_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_13_next_ = out6_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_13_next_) == (eosToken):
                            d_3_done_ = True
                        elif True:
                            d_14_appendedGenerated_: _dafny.Seq
                            d_15_appendedInside_: bool
                            d_16_appendedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                            d_14_appendedGenerated_ = out7_
                            d_15_appendedInside_ = out8_
                            d_16_appendedCurrent_ = out9_
                            generated = d_14_appendedGenerated_
                            insideConstrainedOut = d_15_appendedInside_
                            currentConstrainedOut = d_16_appendedCurrent_
                    elif True:
                        d_17_symbolBudget_: int
                        d_17_symbolBudget_ = (maxSteps) - (d_1_steps_)
                        d_18_currentOut_: _dafny.Seq
                        d_19_hitEos_: bool
                        d_20_stepsUsed_: int
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: int
                        out10_, out11_, out12_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, d_17_symbolBudget_, eosToken)
                        d_18_currentOut_ = out10_
                        d_19_hitEos_ = out11_
                        d_20_stepsUsed_ = out12_
                        generated = (d_10_stablePrefix_) + (d_18_currentOut_)
                        currentConstrainedOut = d_18_currentOut_
                        d_1_steps_ = (d_1_steps_) + (d_20_stepsUsed_)
                        if d_19_hitEos_:
                            d_3_done_ = True
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


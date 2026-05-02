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
        if True:
            generated = generatedPrefix
            insideConstrainedOut = insideConstrained
            currentConstrainedOut = currentConstrained
            cost = 0
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_done_: bool
            d_2_done_ = False
            while ((d_1_steps_) < (maxSteps)) and (not(d_2_done_)):
                if not(insideConstrainedOut):
                    (lm).GenerateLogits((prompt) + (generated))
                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                    d_3_nextOpen_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (lm).ChooseNextTokenUnconstrained()
                    d_3_nextOpen_ = out0_
                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                    if (d_3_nextOpen_) == (eosToken):
                        d_2_done_ = True
                    elif True:
                        if VerifiedDecoderAgent.default__.Contains(d_3_nextOpen_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_nextOpen_]))
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_2_done_ = True
                elif True:
                    d_4_isComplete_: bool
                    d_4_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if d_4_isComplete_:
                        (lm).GenerateLogits((prompt) + (generated))
                        (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                        d_5_nextClose_: _dafny.Seq
                        out1_: _dafny.Seq
                        out1_ = (lm).ChooseNextTokenUnconstrained()
                        d_5_nextClose_ = out1_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        if (d_5_nextClose_) == (eosToken):
                            d_2_done_ = True
                        elif True:
                            if VerifiedDecoderAgent.default__.Contains(d_5_nextClose_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_nextClose_]))
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_done_ = True
                            elif True:
                                d_2_done_ = True
                    elif True:
                        if (stepTokenBudget) == (0):
                            d_2_done_ = True
                        elif True:
                            d_6_remaining_: int
                            d_6_remaining_ = (maxSteps) - (d_1_steps_)
                            if (d_6_remaining_) == (0):
                                d_2_done_ = True
                            elif True:
                                d_7_localBudget_: int
                                d_7_localBudget_ = stepTokenBudget
                                if (d_6_remaining_) < (d_7_localBudget_):
                                    d_7_localBudget_ = d_6_remaining_
                                d_8_stablePrefix_: _dafny.Seq
                                d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_9_constrainedPrompt_: _dafny.Seq
                                d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                                d_10_currentOut_: _dafny.Seq
                                d_11_hitEos_: bool
                                d_12_stepsUsed_: int
                                out2_: _dafny.Seq
                                out3_: bool
                                out4_: int
                                out2_, out3_, out4_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, d_7_localBudget_, eosToken)
                                d_10_currentOut_ = out2_
                                d_11_hitEos_ = out3_
                                d_12_stepsUsed_ = out4_
                                generated = (d_8_stablePrefix_) + (d_10_currentOut_)
                                currentConstrainedOut = d_10_currentOut_
                                d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                                if (d_11_hitEos_) or ((d_12_stepsUsed_) == (0)):
                                    d_2_done_ = True
            cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


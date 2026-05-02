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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, eosToken):
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
        d_1_steps_: int
        d_1_steps_ = 0
        while (d_1_steps_) < (maxSteps):
            if not(insideConstrainedOut):
                d_2_openedGenerated_: _dafny.Seq
                d_3_openedInside_: bool
                d_4_openedCurrent_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_2_openedGenerated_ = out0_
                d_3_openedInside_ = out1_
                d_4_openedCurrent_ = out2_
                generated = d_2_openedGenerated_
                insideConstrainedOut = d_3_openedInside_
                currentConstrainedOut = d_4_openedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_5_completeNow_: bool
                d_5_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                d_6_validCountNow_: int
                out3_: int
                out3_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                d_6_validCountNow_ = out3_
                if (d_5_completeNow_) and ((d_6_validCountNow_) <= (1)):
                    d_1_steps_ = maxSteps
                elif True:
                    d_7_next_: _dafny.Seq
                    d_7_next_ = eosToken
                    if (d_6_validCountNow_) <= (3):
                        d_8_constrainedPrompt1_: _dafny.Seq
                        d_8_constrainedPrompt1_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_8_constrainedPrompt1_, currentConstrainedOut, eosToken)
                        d_7_next_ = out4_
                    elif True:
                        d_9_constrainedPrompt2_: _dafny.Seq
                        d_9_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        (lm).GenerateLogits((d_9_constrainedPrompt2_) + (currentConstrainedOut))
                        d_10_cands_: _dafny.Seq
                        out5_: _dafny.Seq
                        out5_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 6, eosToken)
                        d_10_cands_ = out5_
                        (d_0_helpers_).BoostTokenLogits(lm, d_10_cands_, _dafny.BigRational('8e0'))
                        if not(d_5_completeNow_):
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                        out6_: _dafny.Seq
                        out6_ = (lm).ChooseNextToken()
                        d_7_next_ = out6_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        d_11_validNext_: bool
                        out7_: bool
                        out7_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_7_next_)
                        d_11_validNext_ = out7_
                        if ((d_7_next_) != (eosToken)) and (not(d_11_validNext_)):
                            d_12_constrainedPrompt3_: _dafny.Seq
                            d_12_constrainedPrompt3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_constrainedPrompt3_, currentConstrainedOut, eosToken)
                            d_7_next_ = out8_
                    if (d_7_next_) == (eosToken):
                        d_1_steps_ = maxSteps
                    elif True:
                        d_1_steps_ = (d_1_steps_) + (1)
                        if not(d_5_completeNow_):
                            d_13_appendedGenerated_: _dafny.Seq
                            d_14_appendedInside_: bool
                            d_15_appendedCurrent_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_next_)
                            d_13_appendedGenerated_ = out9_
                            d_14_appendedInside_ = out10_
                            d_15_appendedCurrent_ = out11_
                            generated = d_13_appendedGenerated_
                            insideConstrainedOut = d_14_appendedInside_
                            currentConstrainedOut = d_15_appendedCurrent_
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_16_completeEnd_: bool
            d_16_completeEnd_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_16_completeEnd_:
                d_17_closedGenerated_: _dafny.Seq
                d_18_closedInside_: bool
                d_19_closedCurrent_: _dafny.Seq
                out12_: _dafny.Seq
                out13_: bool
                out14_: _dafny.Seq
                out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_17_closedGenerated_ = out12_
                d_18_closedInside_ = out13_
                d_19_closedCurrent_ = out14_
                generated = d_17_closedGenerated_
                insideConstrainedOut = d_18_closedInside_
                currentConstrainedOut = d_19_closedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


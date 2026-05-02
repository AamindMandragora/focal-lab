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
        d_2_openedOnce_: bool
        d_2_openedOnce_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_3_completeNow_: bool
                        d_3_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if (d_3_completeNow_) and (((d_1_steps_) + (1)) == (maxSteps)):
                            d_4_closedGenerated1_: _dafny.Seq
                            d_5_closedInside1_: bool
                            d_6_closedCurrent1_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_4_closedGenerated1_ = out0_
                            d_5_closedInside1_ = out1_
                            d_6_closedCurrent1_ = out2_
                            generated = d_4_closedGenerated1_
                            insideConstrainedOut = d_5_closedInside1_
                            currentConstrainedOut = d_6_closedCurrent1_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if (d_3_completeNow_) and (((d_1_steps_) + (1)) < (maxSteps)):
                                d_7_closedGenerated2_: _dafny.Seq
                                d_8_closedInside2_: bool
                                d_9_closedCurrent2_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_7_closedGenerated2_ = out3_
                                d_8_closedInside2_ = out4_
                                d_9_closedCurrent2_ = out5_
                                generated = d_7_closedGenerated2_
                                insideConstrainedOut = d_8_closedInside2_
                                currentConstrainedOut = d_9_closedCurrent2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                if ((d_1_steps_) + (1)) >= (maxSteps):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_10_constrainedPrompt_: _dafny.Seq
                                    d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    (lm).GenerateLogits((d_10_constrainedPrompt_) + (currentConstrainedOut))
                                    d_11_candidates_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out6_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                                    d_11_candidates_ = out6_
                                    (d_0_helpers_).BoostTokenLogits(lm, d_11_candidates_, _dafny.BigRational('8e0'))
                                    d_12_nextC_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_12_nextC_ = out7_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_12_nextC_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_13_appendedGenerated_: _dafny.Seq
                                        d_14_appendedInside_: bool
                                        d_15_appendedCurrent_: _dafny.Seq
                                        out8_: _dafny.Seq
                                        out9_: bool
                                        out10_: _dafny.Seq
                                        out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_nextC_)
                                        d_13_appendedGenerated_ = out8_
                                        d_14_appendedInside_ = out9_
                                        d_15_appendedCurrent_ = out10_
                                        generated = d_13_appendedGenerated_
                                        insideConstrainedOut = d_14_appendedInside_
                                        currentConstrainedOut = d_15_appendedCurrent_
                    elif True:
                        d_16_enoughExplanation_: bool
                        d_16_enoughExplanation_ = (6) <= ((len(generated)) - (len(generatedPrefix)))
                        if ((not(d_2_openedOnce_)) and (d_16_enoughExplanation_)) and (((d_1_steps_) + (3)) <= (maxSteps)):
                            d_17_openedGenerated_: _dafny.Seq
                            d_18_openedInside_: bool
                            d_19_openedCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_17_openedGenerated_ = out11_
                            d_18_openedInside_ = out12_
                            d_19_openedCurrent_ = out13_
                            generated = d_17_openedGenerated_
                            insideConstrainedOut = d_18_openedInside_
                            currentConstrainedOut = d_19_openedCurrent_
                            d_2_openedOnce_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                            d_20_nextU_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (lm).ChooseNextToken()
                            d_20_nextU_ = out14_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_20_nextU_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_20_nextU_]))
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


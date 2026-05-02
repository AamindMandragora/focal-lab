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
                        if (d_3_next_) == (eosToken):
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_4_openedGenerated_: _dafny.Seq
                            d_5_openedInside_: bool
                            d_6_openedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openedGenerated_ = out2_
                            d_5_openedInside_ = out3_
                            d_6_openedCurrent_ = out4_
                            generated = d_4_openedGenerated_
                            insideConstrainedOut = d_5_openedInside_
                            currentConstrainedOut = d_6_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_unconstrainedNext_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_unconstrainedNext_ = out5_
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_unconstrainedNext_]))
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_8_completeNow_: bool
                        d_8_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_completeNow_:
                            d_9_closedGenerated_: _dafny.Seq
                            d_10_closedInside_: bool
                            d_11_closedCurrent_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated_ = out6_
                            d_10_closedInside_ = out7_
                            d_11_closedCurrent_ = out8_
                            generated = d_9_closedGenerated_
                            insideConstrainedOut = d_10_closedInside_
                            currentConstrainedOut = d_11_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_12_stablePrefix_: _dafny.Seq
                            d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                            (lm).GenerateLogits((d_13_constrainedPrompt_) + (currentConstrainedOut))
                            d_14_candidates_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                            d_14_candidates_ = out9_
                            d_15_preferred_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_14_candidates_, d_2_flatGroups_)
                            d_15_preferred_ = out10_
                            if (len(d_15_preferred_)) > (0):
                                (d_0_helpers_).BoostTokenLogits(lm, d_15_preferred_, _dafny.BigRational('8e0'))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_16_nextConstrained_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (lm).ChooseNextToken()
                            d_16_nextConstrained_ = out11_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_16_nextConstrained_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_17_appendedGenerated_: _dafny.Seq
                                d_18_appendedInside_: bool
                                d_19_appendedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_nextConstrained_)
                                d_17_appendedGenerated_ = out12_
                                d_18_appendedInside_ = out13_
                                d_19_appendedCurrent_ = out14_
                                generated = d_17_appendedGenerated_
                                insideConstrainedOut = d_18_appendedInside_
                                currentConstrainedOut = d_19_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


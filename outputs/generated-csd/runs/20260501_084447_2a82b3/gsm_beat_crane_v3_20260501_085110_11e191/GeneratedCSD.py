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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openedAnySpan_: bool
        d_2_openedAnySpan_ = insideConstrained
        d_3_answerBiasTokens_: _dafny.Seq
        d_3_answerBiasTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "The")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "the")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "is")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "So")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "First"))])
        d_4_penalizeTokens_: _dafny.Seq
        d_4_penalizeTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Let")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "let")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Suppose")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "suppose")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Assume")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "assume")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Remora")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "dolphin")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "feet")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "inches")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Bodhi"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_2_openedAnySpan_)) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)):
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                            (d_0_helpers_).BoostTokenLogits(lm, d_3_answerBiasTokens_, _dafny.BigRational('3e0'))
                            d_5_openChoice_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (lm).ChooseNextTokenUnconstrained()
                            d_5_openChoice_ = out0_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_5_openChoice_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if (d_5_openChoice_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_6_openedGenerated_: _dafny.Seq
                                    d_7_openedInside_: bool
                                    d_8_openedCurrent_: _dafny.Seq
                                    out1_: _dafny.Seq
                                    out2_: bool
                                    out3_: _dafny.Seq
                                    out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_6_openedGenerated_ = out1_
                                    d_7_openedInside_ = out2_
                                    d_8_openedCurrent_ = out3_
                                    generated = d_6_openedGenerated_
                                    insideConstrainedOut = d_7_openedInside_
                                    currentConstrainedOut = d_8_openedCurrent_
                                    d_2_openedAnySpan_ = True
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_openChoice_]))
                                    if (d_5_openChoice_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                        insideConstrainedOut = True
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                        d_2_openedAnySpan_ = True
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, d_3_answerBiasTokens_, _dafny.BigRational('2e0'))
                            if (not(d_2_openedAnySpan_)) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)):
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'))
                            d_9_nextOutside_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (lm).ChooseNextTokenUnconstrained()
                            d_9_nextOutside_ = out4_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_nextOutside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_nextOutside_]))
                                if (d_9_nextOutside_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_openedAnySpan_ = True
                    elif True:
                        d_10_completeNow_: bool
                        d_10_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_10_completeNow_:
                            d_11_closedGenerated_: _dafny.Seq
                            d_12_closedInside_: bool
                            d_13_closedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_closedGenerated_ = out5_
                            d_12_closedInside_ = out6_
                            d_13_closedCurrent_ = out7_
                            generated = d_11_closedGenerated_
                            insideConstrainedOut = d_12_closedInside_
                            currentConstrainedOut = d_13_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_14_stablePrefix_: _dafny.Seq
                            d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            (lm).GenerateLogits(((prompt) + (d_14_stablePrefix_)) + (currentConstrainedOut))
                            if (len(validTokenGroups)) > (0):
                                (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'))
                            (d_0_helpers_).PenalizeTokenLogits(lm, d_4_penalizeTokens_, _dafny.BigRational('8e0'))
                            d_15_nextInside_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).ConstrainedSample(lm, parser, currentConstrainedOut, eosToken)
                            d_15_nextInside_ = out8_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_15_nextInside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_16_appendedGenerated_: _dafny.Seq
                                d_17_appendedInside_: bool
                                d_18_appendedCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_nextInside_)
                                d_16_appendedGenerated_ = out9_
                                d_17_appendedInside_ = out10_
                                d_18_appendedCurrent_ = out11_
                                generated = d_16_appendedGenerated_
                                insideConstrainedOut = d_17_appendedInside_
                                currentConstrainedOut = d_18_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


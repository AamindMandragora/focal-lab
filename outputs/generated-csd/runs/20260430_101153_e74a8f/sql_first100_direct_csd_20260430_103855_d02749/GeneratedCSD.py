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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens):
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
                            raise _dafny.Break("0")
                    elif True:
                        d_5_completeNow_: bool
                        d_5_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_completeNow_:
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))) in ((lm).Tokens):
                                d_6_closedGenerated_: _dafny.Seq
                                d_7_closedInside_: bool
                                d_8_closedCurrent_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_6_closedGenerated_ = out3_
                                d_7_closedInside_ = out4_
                                d_8_closedCurrent_ = out5_
                                generated = d_6_closedGenerated_
                                insideConstrainedOut = d_7_closedInside_
                                currentConstrainedOut = d_8_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_9_prevBeforeComma_: _dafny.Seq
                            d_10_foundCommaPrev_: bool
                            out6_: _dafny.Seq
                            out7_: bool
                            out6_, out7_ = (d_0_helpers_).LastTokenBefore(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")))
                            d_9_prevBeforeComma_ = out6_
                            d_10_foundCommaPrev_ = out7_
                            if (d_10_foundCommaPrev_) and ((((((d_9_prevBeforeComma_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")))) or ((d_9_prevBeforeComma_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER"))))) or ((d_9_prevBeforeComma_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY"))))) or ((d_9_prevBeforeComma_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING"))))) or ((d_9_prevBeforeComma_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE"))))):
                                d_11_repaired_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_9_prevBeforeComma_)
                                d_11_repaired_ = out8_
                                d_12_dropCount_: int
                                d_12_dropCount_ = (len(currentConstrainedOut)) - (len(d_11_repaired_))
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (d_12_dropCount_):])
                                currentConstrainedOut = d_11_repaired_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_13_narrow_: bool
                                out9_: bool
                                out9_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                                d_13_narrow_ = out9_
                                if d_13_narrow_:
                                    (lm).GenerateLogits((prompt) + (generated))
                                    if (len(validTokenGroups)) > (0):
                                        (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('8e0'))
                                    d_14_next_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = (d_0_helpers_).ConstrainedSample(lm, parser, currentConstrainedOut, eosToken)
                                    d_14_next_ = out10_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_14_next_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_15_appendedGenerated_: _dafny.Seq
                                        d_16_appendedInside_: bool
                                        d_17_appendedCurrent_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out12_: bool
                                        out13_: _dafny.Seq
                                        out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                        d_15_appendedGenerated_ = out11_
                                        d_16_appendedInside_ = out12_
                                        d_17_appendedCurrent_ = out13_
                                        generated = d_15_appendedGenerated_
                                        insideConstrainedOut = d_16_appendedInside_
                                        currentConstrainedOut = d_17_appendedCurrent_
                                elif True:
                                    if ((stepTokenBudget) > (0)) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)):
                                        d_18_constrainedPrompt_: _dafny.Seq
                                        d_18_constrainedPrompt_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                        d_19_currentOut_: _dafny.Seq
                                        d_20_hitEos_: bool
                                        d_21_stepsUsed_: int
                                        out14_: _dafny.Seq
                                        out15_: bool
                                        out16_: int
                                        out14_, out15_, out16_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, stepTokenBudget, eosToken)
                                        d_19_currentOut_ = out14_
                                        d_20_hitEos_ = out15_
                                        d_21_stepsUsed_ = out16_
                                        if (d_21_stepsUsed_) <= ((maxSteps) - (d_1_steps_)):
                                            generated = (d_18_constrainedPrompt_) + (d_19_currentOut_)
                                            currentConstrainedOut = d_19_currentOut_
                                            d_1_steps_ = (d_1_steps_) + (d_21_stepsUsed_)
                                            if d_20_hitEos_:
                                                raise _dafny.Break("0")
                                        elif True:
                                            raise _dafny.Break("0")
                                    elif True:
                                        raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost


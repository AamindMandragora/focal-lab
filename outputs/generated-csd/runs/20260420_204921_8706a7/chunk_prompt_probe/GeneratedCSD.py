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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_useSingle_: bool
                        d_2_useSingle_ = False
                        if (len(generated)) > (0):
                            d_3_lastTok_: _dafny.Seq
                            d_3_lastTok_ = (generated)[(len(generated)) - (1)]
                            if ((((((((VerifiedDecoderAgent.default__.Contains(d_3_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))) or (VerifiedDecoderAgent.default__.Contains(d_3_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or (VerifiedDecoderAgent.default__.Contains(d_3_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or (VerifiedDecoderAgent.default__.Contains(d_3_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final"))))) or (VerifiedDecoderAgent.default__.Contains(d_3_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))))) or (VerifiedDecoderAgent.default__.Contains(d_3_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "symbol"))))) or (VerifiedDecoderAgent.default__.Contains(d_3_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Symbol"))))) or (VerifiedDecoderAgent.default__.Contains(d_3_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "quantity"))))) or (VerifiedDecoderAgent.default__.Contains(d_3_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Quantity")))):
                                d_2_useSingle_ = True
                        if d_2_useSingle_:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'))
                            d_4_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (lm).ChooseNextTokenUnconstrained()
                            d_4_next_ = out0_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_4_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                                if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            d_5_chunkLimit_: int
                            d_5_chunkLimit_ = (maxSteps) - (d_1_steps_)
                            if (d_5_chunkLimit_) > (8):
                                d_5_chunkLimit_ = 8
                            d_6_chunkGenerated_: _dafny.Seq
                            d_7_stoppedOnOpenSpan_: bool
                            d_8_stoppedOnEos_: bool
                            d_9_stepsUsed_: int
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: bool
                            out4_: int
                            out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkLimit_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_6_chunkGenerated_ = out1_
                            d_7_stoppedOnOpenSpan_ = out2_
                            d_8_stoppedOnEos_ = out3_
                            d_9_stepsUsed_ = out4_
                            generated = d_6_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                            if d_8_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_7_stoppedOnOpenSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_10_closedGenerated_: _dafny.Seq
                            d_11_closedInside_: bool
                            d_12_closedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_closedGenerated_ = out5_
                            d_11_closedInside_ = out6_
                            d_12_closedCurrent_ = out7_
                            generated = d_10_closedGenerated_
                            insideConstrainedOut = d_11_closedInside_
                            currentConstrainedOut = d_12_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_13_stablePrefix_: _dafny.Seq
                            d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_14_constrainedPrompt_: _dafny.Seq
                            d_14_constrainedPrompt_ = (prompt) + (d_13_stablePrefix_)
                            d_15_narrow_: bool
                            out8_: bool
                            out8_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_15_narrow_ = out8_
                            d_16_validCount_: int
                            out9_: int
                            out9_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_16_validCount_ = out9_
                            if (d_15_narrow_) or ((d_16_validCount_) <= (2)):
                                d_17_next_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_17_next_ = out10_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_17_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_18_appendedGenerated_: _dafny.Seq
                                    d_19_appendedInside_: bool
                                    d_20_appendedCurrent_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                    d_18_appendedGenerated_ = out11_
                                    d_19_appendedInside_ = out12_
                                    d_20_appendedCurrent_ = out13_
                                    generated = d_18_appendedGenerated_
                                    insideConstrainedOut = d_19_appendedInside_
                                    currentConstrainedOut = d_20_appendedCurrent_
                            elif True:
                                d_21_next_: _dafny.Seq
                                d_22_isValid_: bool
                                out14_: _dafny.Seq
                                out15_: bool
                                out14_, out15_ = (d_0_helpers_).SoftConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('3e0'), eosToken)
                                d_21_next_ = out14_
                                d_22_isValid_ = out15_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_21_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif d_22_isValid_:
                                    d_23_appendedGenerated_: _dafny.Seq
                                    d_24_appendedInside_: bool
                                    d_25_appendedCurrent_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                    d_23_appendedGenerated_ = out16_
                                    d_24_appendedInside_ = out17_
                                    d_25_appendedCurrent_ = out18_
                                    generated = d_23_appendedGenerated_
                                    insideConstrainedOut = d_24_appendedInside_
                                    currentConstrainedOut = d_25_appendedCurrent_
                                elif True:
                                    d_26_repairedGenerated_: _dafny.Seq
                                    d_27_repairedCurrent_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out19_, out20_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_13_stablePrefix_, generated, currentConstrainedOut)
                                    d_26_repairedGenerated_ = out19_
                                    d_27_repairedCurrent_ = out20_
                                    generated = d_26_repairedGenerated_
                                    currentConstrainedOut = d_27_repairedCurrent_
                                    if (parser).IsValidPrefix(currentConstrainedOut):
                                        insideConstrainedOut = True
                                    elif True:
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


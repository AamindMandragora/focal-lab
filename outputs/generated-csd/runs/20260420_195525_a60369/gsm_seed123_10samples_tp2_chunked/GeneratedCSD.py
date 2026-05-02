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
                        d_2_chunkLimit_: int
                        d_2_chunkLimit_ = (maxSteps) - (d_1_steps_)
                        if (d_2_chunkLimit_) > (2):
                            d_2_chunkLimit_ = 2
                        if (len(generated)) > (0):
                            (lm).GenerateLogits((prompt) + (generated))
                            d_3_lastTok_: _dafny.Seq
                            d_3_lastTok_ = (generated)[(len(generated)) - (1)]
                            if (((((((VerifiedDecoderAgent.default__.Contains(d_3_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))) or (VerifiedDecoderAgent.default__.Contains(d_3_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":"))))) or (VerifiedDecoderAgent.default__.Contains(d_3_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or (VerifiedDecoderAgent.default__.Contains(d_3_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or (VerifiedDecoderAgent.default__.Contains(d_3_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "let"))))) or (VerifiedDecoderAgent.default__.Contains(d_3_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Let"))))) or (VerifiedDecoderAgent.default__.Contains(d_3_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "symbol"))))) or (VerifiedDecoderAgent.default__.Contains(d_3_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Symbol")))):
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
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
                            d_5_chunkGenerated_: _dafny.Seq
                            d_6_stoppedOnOpenSpan_: bool
                            d_7_stoppedOnEos_: bool
                            d_8_stepsUsed_: int
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: bool
                            out4_: int
                            out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkLimit_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_5_chunkGenerated_ = out1_
                            d_6_stoppedOnOpenSpan_ = out2_
                            d_7_stoppedOnEos_ = out3_
                            d_8_stepsUsed_ = out4_
                            generated = d_5_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                            if d_7_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif True:
                                if d_6_stoppedOnOpenSpan_:
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_9_closedGenerated_: _dafny.Seq
                            d_10_closedInside_: bool
                            d_11_closedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated_ = out5_
                            d_10_closedInside_ = out6_
                            d_11_closedCurrent_ = out7_
                            generated = d_9_closedGenerated_
                            insideConstrainedOut = d_10_closedInside_
                            currentConstrainedOut = d_11_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_12_stablePrefix_: _dafny.Seq
                            d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                            d_14_validCount_: int
                            out8_: int
                            out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_14_validCount_ = out8_
                            if (d_14_validCount_) <= (2):
                                d_15_next_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_15_next_ = out9_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_15_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_16_valid_: bool
                                    out10_: bool
                                    out10_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_15_next_)
                                    d_16_valid_ = out10_
                                    if d_16_valid_:
                                        d_17_appendedGenerated_: _dafny.Seq
                                        d_18_appendedInside_: bool
                                        d_19_appendedCurrent_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out12_: bool
                                        out13_: _dafny.Seq
                                        out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                        d_17_appendedGenerated_ = out11_
                                        d_18_appendedInside_ = out12_
                                        d_19_appendedCurrent_ = out13_
                                        generated = d_17_appendedGenerated_
                                        insideConstrainedOut = d_18_appendedInside_
                                        currentConstrainedOut = d_19_appendedCurrent_
                                    elif True:
                                        d_20_repairedGenerated_: _dafny.Seq
                                        d_21_repairedCurrent_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out14_, out15_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_12_stablePrefix_, generated, currentConstrainedOut)
                                        d_20_repairedGenerated_ = out14_
                                        d_21_repairedCurrent_ = out15_
                                        generated = d_20_repairedGenerated_
                                        currentConstrainedOut = d_21_repairedCurrent_
                                        if (parser).IsValidPrefix(currentConstrainedOut):
                                            insideConstrainedOut = True
                                        elif True:
                                            insideConstrainedOut = False
                                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                d_22_next_: _dafny.Seq
                                d_23_isValid_: bool
                                out16_: _dafny.Seq
                                out17_: bool
                                out16_, out17_ = (d_0_helpers_).SoftConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('3e0'), eosToken)
                                d_22_next_ = out16_
                                d_23_isValid_ = out17_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_22_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    if d_23_isValid_:
                                        d_24_appendedGenerated_: _dafny.Seq
                                        d_25_appendedInside_: bool
                                        d_26_appendedCurrent_: _dafny.Seq
                                        out18_: _dafny.Seq
                                        out19_: bool
                                        out20_: _dafny.Seq
                                        out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                        d_24_appendedGenerated_ = out18_
                                        d_25_appendedInside_ = out19_
                                        d_26_appendedCurrent_ = out20_
                                        generated = d_24_appendedGenerated_
                                        insideConstrainedOut = d_25_appendedInside_
                                        currentConstrainedOut = d_26_appendedCurrent_
                                    elif True:
                                        d_27_repairedGenerated_: _dafny.Seq
                                        d_28_repairedCurrent_: _dafny.Seq
                                        out21_: _dafny.Seq
                                        out22_: _dafny.Seq
                                        out21_, out22_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_12_stablePrefix_, generated, currentConstrainedOut)
                                        d_27_repairedGenerated_ = out21_
                                        d_28_repairedCurrent_ = out22_
                                        generated = d_27_repairedGenerated_
                                        currentConstrainedOut = d_28_repairedCurrent_
                                        if (parser).IsValidPrefix(currentConstrainedOut):
                                            insideConstrainedOut = True
                                        elif True:
                                            insideConstrainedOut = False
                                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


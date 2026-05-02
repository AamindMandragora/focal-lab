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
                        (lm).GenerateLogits((prompt) + (generated))
                        if (len(generated)) > (0):
                            d_2_lastTok_: _dafny.Seq
                            d_2_lastTok_ = (generated)[(len(generated)) - (1)]
                            if (((((((VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Let"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "let"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "define"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Define"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "symbol"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "quantity")))):
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('3e0'))
                            if (((VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))):
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('2e0'))
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (lm).ChooseNextTokenUnconstrained()
                        d_3_next_ = out0_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_4_closedGenerated_: _dafny.Seq
                            d_5_closedInside_: bool
                            d_6_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_4_closedGenerated_ = out1_
                            d_5_closedInside_ = out2_
                            d_6_closedCurrent_ = out3_
                            generated = d_4_closedGenerated_
                            insideConstrainedOut = d_5_closedInside_
                            currentConstrainedOut = d_6_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_stablePrefix_: _dafny.Seq
                            d_7_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_8_constrainedPrompt_: _dafny.Seq
                            d_8_constrainedPrompt_ = (prompt) + (d_7_stablePrefix_)
                            d_9_narrow_: bool
                            out4_: bool
                            out4_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_9_narrow_ = out4_
                            if d_9_narrow_:
                                d_10_next_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_10_next_ = out5_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_10_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_11_valid_: bool
                                    out6_: bool
                                    out6_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_10_next_)
                                    d_11_valid_ = out6_
                                    if d_11_valid_:
                                        d_12_appendedGenerated_: _dafny.Seq
                                        d_13_appendedInside_: bool
                                        d_14_appendedCurrent_: _dafny.Seq
                                        out7_: _dafny.Seq
                                        out8_: bool
                                        out9_: _dafny.Seq
                                        out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                                        d_12_appendedGenerated_ = out7_
                                        d_13_appendedInside_ = out8_
                                        d_14_appendedCurrent_ = out9_
                                        generated = d_12_appendedGenerated_
                                        insideConstrainedOut = d_13_appendedInside_
                                        currentConstrainedOut = d_14_appendedCurrent_
                                    elif True:
                                        d_15_repairedGenerated_: _dafny.Seq
                                        d_16_repairedCurrent_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_7_stablePrefix_, generated, currentConstrainedOut)
                                        d_15_repairedGenerated_ = out10_
                                        d_16_repairedCurrent_ = out11_
                                        generated = d_15_repairedGenerated_
                                        currentConstrainedOut = d_16_repairedCurrent_
                                        if (parser).IsValidPrefix(currentConstrainedOut):
                                            insideConstrainedOut = True
                                        elif True:
                                            insideConstrainedOut = False
                                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                d_17_next_: _dafny.Seq
                                d_18_isValid_: bool
                                out12_: _dafny.Seq
                                out13_: bool
                                out12_, out13_ = (d_0_helpers_).SoftConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('25e-1'), eosToken)
                                d_17_next_ = out12_
                                d_18_isValid_ = out13_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_17_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    if d_18_isValid_:
                                        d_19_appendedGenerated_: _dafny.Seq
                                        d_20_appendedInside_: bool
                                        d_21_appendedCurrent_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out15_: bool
                                        out16_: _dafny.Seq
                                        out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                        d_19_appendedGenerated_ = out14_
                                        d_20_appendedInside_ = out15_
                                        d_21_appendedCurrent_ = out16_
                                        generated = d_19_appendedGenerated_
                                        insideConstrainedOut = d_20_appendedInside_
                                        currentConstrainedOut = d_21_appendedCurrent_
                                    elif True:
                                        d_22_repairedGenerated_: _dafny.Seq
                                        d_23_repairedCurrent_: _dafny.Seq
                                        out17_: _dafny.Seq
                                        out18_: _dafny.Seq
                                        out17_, out18_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_7_stablePrefix_, generated, currentConstrainedOut)
                                        d_22_repairedGenerated_ = out17_
                                        d_23_repairedCurrent_ = out18_
                                        generated = d_22_repairedGenerated_
                                        currentConstrainedOut = d_23_repairedCurrent_
                                        if (parser).IsValidPrefix(currentConstrainedOut):
                                            insideConstrainedOut = True
                                        elif True:
                                            insideConstrainedOut = False
                                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost


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
                        (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('3e0'))
                        if (len(generated)) > (0):
                            d_2_lastTok_: _dafny.Seq
                            d_2_lastTok_ = (generated)[(len(generated)) - (1)]
                            if ((((((VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "let"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Let"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "quantity"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Quantity"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "symbol"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Symbol")))):
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'))
                            if (((VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))):
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('3e0'))
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (lm).ChooseNextTokenUnconstrained()
                        d_3_next_ = out0_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_4_g2_: _dafny.Seq
                                d_5_i2_: bool
                                d_6_c2_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_4_g2_ = out1_
                                d_5_i2_ = out2_
                                d_6_c2_ = out3_
                                generated = d_4_g2_
                                insideConstrainedOut = d_5_i2_
                                currentConstrainedOut = d_6_c2_
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_7_g3_: _dafny.Seq
                            d_8_i3_: bool
                            d_9_c3_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_g3_ = out4_
                            d_8_i3_ = out5_
                            d_9_c3_ = out6_
                            generated = d_7_g3_
                            insideConstrainedOut = d_8_i3_
                            currentConstrainedOut = d_9_c3_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_stablePrefix_: _dafny.Seq
                            d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_11_constrainedPrompt_: _dafny.Seq
                            d_11_constrainedPrompt_ = (prompt) + (d_10_stablePrefix_)
                            d_12_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_12_next_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_13_g4_: _dafny.Seq
                                d_14_i4_: bool
                                d_15_c4_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                d_13_g4_ = out8_
                                d_14_i4_ = out9_
                                d_15_c4_ = out10_
                                generated = d_13_g4_
                                insideConstrainedOut = d_14_i4_
                                currentConstrainedOut = d_15_c4_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

